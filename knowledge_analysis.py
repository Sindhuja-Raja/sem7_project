"""
Knowledge Retrieval / Research Analyst module - now a true RAG consumer.

Runs after the main retrieve -> dedupe -> rank -> rerank pipeline has
produced the final list of papers shown as cards. Per-paper extraction
itself (per-field-group semantic retrieval against that paper's already-
built FAISS chunk index, then ONE Groq call over only the retrieved
chunks) is owned by knowledge_extraction_agent.py - this module is a thin
adapter: it calls KnowledgeExtractionPipeline, reshapes the results into
the flat-dict-list tables the rest of the app already expects, and builds
the four cross-paper aggregation tables that were always this module's
own concern (deterministic counting + embedding-based clustering, neither
of which touches the extraction/retrieval layer). The shared per-paper
FAISS index is built once, only for papers that already passed Paper
Validation (app.py's validation-gated collection loop, calling
pdf_extraction.get_full_paper_text -> paper_validation.validate_paper_text
-> rag_pipeline.build_paper_index_from_sections) - never rebuilt here.

Five outputs, all JSON-serializable lists of flat dicts so app.py can
hand them straight to pandas.DataFrame() for st.dataframe() rendering:

    knowledge_table       - one row per paper, ~26 extracted technical fields.
    overall_analysis      - Category / Most Common / Evidence, aggregated
                             deterministically (counting) across papers.
    research_gap_analysis - semantically clustered research gaps (embedding
                             similarity, not string matching) with a
                             paraphrased label, papers, frequency and one
                             practical recommendation per cluster.
    future_work_analysis  - semantically clustered future-work themes.
    research_insights     - 8 fixed-category synthesized insights, grounded
                             only in the four outputs above.
    retrieved_chunks       - which chunks were retrieved for Knowledge
                             Extraction, with similarity score and section
                             name, for the Streamlit "Retrieved Chunks"
                             transparency table shown before knowledge_table.

Every per-paper field defaults to "Not Found in Retrieved Context" when the
retrieved chunks don't explicitly state it - the LLM is never allowed to
fall back on its own knowledge or the un-retrieved rest of the document.
Degrades gracefully to empty tables (never an exception) if GROQ_API_KEY
isn't configured - the paper cards above are unaffected either way.
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import config
import llm
from embedding import cluster_texts
from knowledge_extraction_agent import FIELD_KEYS, FIELD_TO_COLUMN, KnowledgeExtractionPipeline
from rag_pipeline import NOT_FOUND_IN_CONTEXT, PaperIndex
from utils import Paper

logger = logging.getLogger(__name__)

NOT_REPORTED = NOT_FOUND_IN_CONTEXT  # local alias - kept so the rest of this
                                      # file (written against "NOT_REPORTED")
                                      # didn't need a hundred renames; the
                                      # actual string is now the shared RAG
                                      # sentinel from rag_pipeline.py

_MAX_PAPERS_TO_ANALYZE = 15


@dataclass
class KnowledgeAnalysis:
    knowledge_table: List[dict] = field(default_factory=list)
    overall_analysis: List[dict] = field(default_factory=list)
    research_gap_analysis: List[dict] = field(default_factory=list)
    future_work_analysis: List[dict] = field(default_factory=list)
    research_insights: List[dict] = field(default_factory=list)
    retrieved_chunks: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# Field list/column labels and the extraction prompt itself now live in
# knowledge_extraction_agent.py (KNOWLEDGE_FIELDS/FIELD_KEYS/FIELD_TO_COLUMN)
# alongside the per-field-group RAG retrieval that grounds them - this
# module only needs the resulting key->column mapping to build tables.
_FIELDS = FIELD_KEYS
_FIELD_TO_COLUMN = FIELD_TO_COLUMN


def _extract_rows(papers: List[Paper], rag_cache: Dict[str, PaperIndex]) -> List[dict]:
    """Runs KnowledgeExtractionPipeline (knowledge_extraction_agent.py -
    per-field-group RAG retrieval + one Groq call per paper) and reshapes
    each ExtractionResult into the flat-dict row shape every table builder
    below already expects, so nothing downstream needed to change."""
    pipeline = KnowledgeExtractionPipeline()
    results = pipeline.run(papers, rag_cache)
    return [result.to_row() for result in results]


# --------------------------------------------------------------------------
# Table 1: knowledge_table
# --------------------------------------------------------------------------

def _build_knowledge_table(papers: List[Paper], rows: List[dict]) -> List[dict]:
    table = []
    for paper, row in zip(papers, rows):
        entry = {
            "Paper Title": paper.title,
            "Year": paper.year or NOT_REPORTED,
            "Source": paper.source,
            "Evidence Source": row["_evidence_source"],
        }
        for f in _FIELDS:
            entry[_FIELD_TO_COLUMN[f]] = row[f]
        table.append(entry)
    return table


# --------------------------------------------------------------------------
# Table 2: overall_analysis - deterministic Python aggregation, not
# another LLM call, so "most common" is always exactly correct.
# --------------------------------------------------------------------------

_SPLIT_RE = re.compile(r",| / |/| and |;")


def _tokenize(value: str) -> List[str]:
    return [t.strip() for t in _SPLIT_RE.split(value) if t.strip() and t.strip().lower() != NOT_REPORTED.lower()]


def _overall_row(category: str, field_name: str, rows: List[dict]) -> dict:
    """Full frequency breakdown for one category - every distinct value
    extracted for `field_name` across all papers, most-used first, never
    collapsed to a single "Most Common" winner and never hidden behind a
    "No Common Pattern Found" placeholder when nothing repeats. "Most
    Common"/"Evidence" are kept (top item) for callers that only need the
    single winner (e.g. problem_solution_analysis.py); "Items" carries the
    complete per-value breakdown for the UI."""
    total = len(rows)
    counter: Counter = Counter()
    for r in rows:
        counter.update(_tokenize(r.get(field_name, NOT_REPORTED)))
    items = [{"Value": value, "Count": count, "Papers": f"{count} paper{'s' if count != 1 else ''}"}
             for value, count in counter.most_common()]
    if not items:
        return {"Category": category, "Most Common": NOT_REPORTED, "Evidence": NOT_REPORTED, "Items": []}
    top_value, top_count = counter.most_common(1)[0]
    return {
        "Category": category,
        "Most Common": top_value,
        "Evidence": f"{top_count} of {total} papers",
        "Items": items,
    }


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _highest_accuracy_row(rows: List[dict]) -> dict:
    total = len(rows)
    reported = 0
    best_num, best_display = -1.0, None
    for r in rows:
        acc = r.get("accuracy", NOT_REPORTED)
        if acc == NOT_REPORTED:
            continue
        reported += 1
        match = _NUMBER_RE.search(acc)
        if not match:
            continue
        num = float(match.group(0))
        if num > best_num:
            best_num = num
            title = r.get("_paper_title", "")
            best_display = f"{acc} - {title}" if title else acc
    if best_display is None:
        return {"Category": "Highest Reported Accuracy", "Most Common": NOT_REPORTED,
                "Evidence": f"0 of {total} papers report accuracy"}
    return {"Category": "Highest Reported Accuracy", "Most Common": best_display,
            "Evidence": f"Reported in {reported} of {total} papers"}


def _build_overall_analysis(rows: List[dict]) -> List[dict]:
    order = [
        ("Most Used AI Technique", "ai_technique"),
        ("Most Used AI Model", "ai_model"),
        ("Most Used Dataset", "dataset"),
        ("Most Used Framework", "framework"),
        ("Most Used Evaluation Metric", "evaluation_metrics"),
        ("Most Used Programming Language", "programming_language"),
        ("Most Used Hardware", "hardware"),
    ]
    result = [_overall_row(category, field_name, rows) for category, field_name in order]
    result.append(_highest_accuracy_row(rows))
    result.append(_overall_row("Most Common Research Domain", "research_domain", rows))
    return result


# --------------------------------------------------------------------------
# Tables 3 & 4: semantic clustering (dense embeddings, not string
# matching) + one batched LLM call per table to paraphrase/label each
# cluster - "Papers Identified"/"Frequency" are computed from the
# clustering itself, not guessed by the LLM.
# --------------------------------------------------------------------------

def _cluster_texts(items: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
    """items = [(paper_title, text), ...], clustered via the shared
    embedding.cluster_texts() (greedy, running-centroid) after dropping
    "Not Reported" entries."""
    items = [(title, text) for title, text in items if text and text.lower() != NOT_REPORTED.lower()]
    return cluster_texts(items, config.KNOWLEDGE_GAP_CLUSTER_SIMILARITY_THRESHOLD)


def _build_cluster_message(clusters: List[List[Tuple[str, str]]]) -> str:
    blocks = []
    for i, cluster in enumerate(clusters, start=1):
        member_lines = "\n".join(f"- ({title}) {text}" for title, text in cluster)
        blocks.append(f"Group {i}:\n{member_lines}")
    return "\n\n".join(blocks)


def _label_clusters(clusters: List[List[Tuple[str, str]]], system_prompt: str) -> List[dict]:
    if not clusters:
        return []
    if not llm.is_available():
        return [{} for _ in clusters]

    content = llm._call_groq(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_cluster_message(clusters)},
        ],
        max_tokens=min(3000, 220 * len(clusters)),
        temperature=0.2,
        json_mode=True,
    )
    data = llm.parse_json(content) if content else None
    by_index: Dict[int, dict] = {}
    if isinstance(data, dict) and isinstance(data.get("clusters"), list):
        for entry in data["clusters"]:
            if not isinstance(entry, dict):
                continue
            try:
                by_index[int(entry.get("index"))] = entry
            except (TypeError, ValueError):
                continue
    elif content:
        logger.warning("Cluster labeling returned unusable JSON: %s", content[:200])
    return [by_index.get(i, {}) for i in range(1, len(clusters) + 1)]


GAP_LABELING_SYSTEM_PROMPT = """You receive groups of research-gap statements from different \
academic papers, already clustered because they discuss a similar theme. For EACH group, write:
- "gap_label": a concise (5-12 word) PARAPHRASED description of the shared research gap - do NOT
  copy any input sentence verbatim, synthesize your own wording for the shared theme.
- "recommendation": one practical, actionable recommendation for future researchers to address
  this gap.

Base both only on the statements given in that group - never invent a gap not reflected in the
given text.

Respond with ONLY a JSON object: {"clusters": [{"index": 1, "gap_label": "...", "recommendation": "..."}]}
One entry per group given, "index" matching the 1-based position below. No markdown fences, no
commentary."""

FUTURE_WORK_LABELING_SYSTEM_PROMPT = """You receive groups of "future work" statements from \
different academic papers, already clustered because they propose a similar direction. For EACH
group, write "future_work_label": a concise (5-12 word) PARAPHRASED description of the shared
future-work theme - do NOT copy any input sentence verbatim.

Base it only on the statements given in that group.

Respond with ONLY a JSON object: {"clusters": [{"index": 1, "future_work_label": "..."}]}
One entry per group given, "index" matching the 1-based position below. No markdown fences, no
commentary."""


def _build_research_gap_table(rows: List[dict]) -> List[dict]:
    items = [(r.get("_paper_title", ""), r.get("research_gap", NOT_REPORTED)) for r in rows]
    clusters = _cluster_texts(items)
    if not clusters:
        return []

    labels = _label_clusters(clusters, GAP_LABELING_SYSTEM_PROMPT)
    table = []
    for cluster, label_entry in zip(clusters, labels):
        papers = ", ".join(sorted({title for title, _ in cluster}))
        gap_label = str(label_entry.get("gap_label") or "").strip() or cluster[0][1][:120]
        recommendation = str(label_entry.get("recommendation") or "").strip() or NOT_REPORTED
        table.append({
            "Research Gap": gap_label,
            "Papers Identified": papers,
            "Frequency": len(cluster),
            "Recommendation": recommendation,
        })
    table.sort(key=lambda r: r["Frequency"], reverse=True)
    return table


def _build_future_work_table(rows: List[dict]) -> List[dict]:
    items = [(r.get("_paper_title", ""), r.get("future_work", NOT_REPORTED)) for r in rows]
    clusters = _cluster_texts(items)
    if not clusters:
        return []

    labels = _label_clusters(clusters, FUTURE_WORK_LABELING_SYSTEM_PROMPT)
    table = []
    for cluster, label_entry in zip(clusters, labels):
        papers = ", ".join(sorted({title for title, _ in cluster}))
        future_work_label = str(label_entry.get("future_work_label") or "").strip() or cluster[0][1][:120]
        table.append({
            "Future Work": future_work_label,
            "Papers": papers,
            "Frequency": len(cluster),
        })
    table.sort(key=lambda r: r["Frequency"], reverse=True)
    return table


# --------------------------------------------------------------------------
# Table 5: research_insights - grounded in the four outputs above, not
# a fresh re-read of the papers.
# --------------------------------------------------------------------------

INSIGHT_CATEGORIES = [
    "Research Trend", "Emerging AI Technique", "Common Limitation",
    "Most Promising Research Direction", "Recommended AI Model",
    "Recommended Dataset", "Recommended Framework", "Novel Research Opportunity",
]

INSIGHTS_SYSTEM_PROMPT = f"""You are a research analyst producing high-level insights from an \
already-completed literature analysis. You are given aggregated "most common" stats across the
analyzed papers, clustered research gaps with frequency, and clustered future-work themes with
frequency. Using ONLY this given data - never inventing facts beyond it - produce exactly one
insight for each of these 8 categories, in this order: {", ".join(INSIGHT_CATEGORIES)}.

- "Research Trend": the pattern most evident across the papers.
- "Emerging AI Technique": a technique that appears to be gaining traction in this data.
- "Common Limitation": the most frequently shared limitation/gap theme.
- "Most Promising Research Direction": which future-work theme looks most valuable to pursue.
- "Recommended AI Model": which model appears best-supported by the data, with a short reason.
- "Recommended Dataset": which dataset appears most standard/reusable, with a short reason.
- "Recommended Framework": which framework appears most supported, with a short reason.
- "Novel Research Opportunity": a genuine gap in the given data not yet addressed by any paper.

If the given data has no clear answer for a category, say so honestly (e.g. "No clear pattern in
the analyzed papers") rather than inventing one.

Respond with ONLY a JSON object: {{"insights": [{{"category": "...", "insight": "..."}}]}} with
exactly these 8 categories in the order given. No markdown fences, no commentary."""


def _generate_insights(
    overall_analysis: List[dict], research_gap_analysis: List[dict],
    future_work_analysis: List[dict], paper_count: int,
) -> List[dict]:
    if not llm.is_available():
        return [{"Insight Category": c, "Insight": "GROQ_API_KEY not configured - insights unavailable."}
                for c in INSIGHT_CATEGORIES]

    summary = {
        "papers_analyzed": paper_count,
        "overall_analysis": overall_analysis,
        "research_gaps": [
            {"gap": r["Research Gap"], "frequency": r["Frequency"]} for r in research_gap_analysis[:8]
        ],
        "future_work_themes": [
            {"theme": r["Future Work"], "frequency": r["Frequency"]} for r in future_work_analysis[:8]
        ],
    }

    content = llm._call_groq(
        messages=[
            {"role": "system", "content": INSIGHTS_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(summary, indent=2)},
        ],
        max_tokens=900,
        temperature=0.2,
        json_mode=True,
    )
    data = llm.parse_json(content) if content else None
    by_category: Dict[str, str] = {}
    if isinstance(data, dict) and isinstance(data.get("insights"), list):
        for entry in data["insights"]:
            if isinstance(entry, dict) and entry.get("category"):
                by_category[str(entry["category"]).strip()] = str(entry.get("insight") or "").strip()
    elif content:
        logger.warning("Insight generation returned unusable JSON: %s", content[:200])

    return [
        {"Insight Category": c, "Insight": by_category.get(c) or "No clear pattern in the analyzed papers."}
        for c in INSIGHT_CATEGORIES
    ]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _build_retrieved_chunks_table(papers: List[Paper], rows: List[dict]) -> List[dict]:
    """The Streamlit "Retrieved Chunks" transparency table - Paper /
    Section Name / Similarity Score / Chunk Preview - shown before
    knowledge_table so the user can see exactly what evidence Knowledge
    Extraction actually retrieved and reasoned over."""
    table = []
    for paper, row in zip(papers, rows):
        for rc in row.get("_retrieved_chunks", []):
            table.append({
                "Paper Title": paper.title,
                "Section Name": rc.chunk.section_name,
                "Similarity Score": round(rc.similarity_score, 3),
                "Chunk Preview": rc.chunk.chunk_text[:200] + ("..." if len(rc.chunk.chunk_text) > 200 else ""),
            })
    table.sort(key=lambda r: r["Similarity Score"], reverse=True)
    return table


def analyze_papers(papers: List[Paper], rag_cache: Dict[str, PaperIndex]) -> KnowledgeAnalysis:
    """Entry point used by app.py after ranking is finished. `rag_cache`
    is the shared {paper title -> PaperIndex} built once by app.py's
    validation-gated collection loop - reused here, never rebuilt."""
    papers = papers[:_MAX_PAPERS_TO_ANALYZE]
    if not papers:
        return KnowledgeAnalysis()

    rows = _extract_rows(papers, rag_cache)
    for row, paper in zip(rows, papers):
        row["_paper_title"] = paper.title

    knowledge_table = _build_knowledge_table(papers, rows)
    retrieved_chunks = _build_retrieved_chunks_table(papers, rows)
    overall_analysis = _build_overall_analysis(rows)
    research_gap_analysis = _build_research_gap_table(rows)
    future_work_analysis = _build_future_work_table(rows)
    research_insights = _generate_insights(
        overall_analysis, research_gap_analysis, future_work_analysis, len(papers),
    )

    return KnowledgeAnalysis(
        knowledge_table=knowledge_table,
        overall_analysis=overall_analysis,
        research_gap_analysis=research_gap_analysis,
        future_work_analysis=future_work_analysis,
        research_insights=research_insights,
        retrieved_chunks=retrieved_chunks,
    )
