"""
Problem-Solution Analysis Agent.

Runs automatically, immediately after Knowledge Retrieval and before
Contradiction Detection, as part of the same automatic search pipeline
(not on-demand like Contradiction Detection). Performs literature
SYNTHESIS across all displayed papers together - never a per-paper
summary - by:

    1. extracting each paper's main + secondary research problem(s) - one
       batched LLM call over chunks retrieved from each paper's shared
       FAISS index (config.RAG_QUERY_PROBLEM_SOLUTION: "problem statement,
       research problem, methodology, results") instead of the whole
       paper. This is the only new extraction this module performs - AI
       Technique / AI Model / best-reported performance are reused
       directly from knowledge_analysis.py's already-computed
       knowledge_table, at zero extra retrieval/LLM cost.
    2. semantically clustering similar problem phrases (shared embedding-
       based clustering from embedding.cluster_texts - "High Latency" and
       "High Response Time" merge into one row instead of staying two)
    3. mapping each clustered problem to the AI techniques/models used
       across its supporting papers and their best reported performance,
       all via deterministic aggregation over already-extracted data

Output: problem_solution_table + summary, both JSON-serializable lists of
flat dicts for st.dataframe(). Designed so a future "Inventor Agent" can
consume problem_solution_table directly to spot over-solved vs
under-solved problems and recommend research directions.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import config
import llm
import pipeline_status
from embedding import cluster_texts
from rag_pipeline import NOT_FOUND_IN_CONTEXT, PaperIndex, retrieve_context
from utils import Paper

logger = logging.getLogger(__name__)

NOT_REPORTED = NOT_FOUND_IN_CONTEXT  # shared RAG sentinel - see rag_pipeline.py

# knowledge_table columns this module reuses instead of re-extracting.
_COL_AI_TECHNIQUE = "AI Technique"
_COL_AI_MODEL = "AI Model"
_METRIC_COLUMNS = ["Accuracy", "F1 Score", "Precision", "Recall", "AUC", "mAP"]


@dataclass
class PaperProblems:
    main_problem: str = NOT_REPORTED
    secondary_problems: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Step 1: per-paper problem extraction (one batched Groq call - small,
# Introduction-priority text keeps this cheap even for 15 papers at once)
# --------------------------------------------------------------------------

PROBLEM_EXTRACTION_SYSTEM_PROMPT = """You extract the research problem(s) addressed by academic \
papers, for a literature-synthesis system that will group similar problems together across many \
papers. For EACH paper you are given its RETRIEVED CHUNKS (selected by semantic search as the \
passages most relevant to identifying its research problem) - not the whole paper.

For EACH paper given, extract:
- main_problem: the single primary research problem the paper addresses, as a concise 2-5 word
  phrase - never a full sentence. Examples: "High Resource Allocation Cost", "High Latency",
  "Poor Resource Utilization", "SLA Violation", "Energy Consumption", "Load Imbalance", "Limited
  Explainability", "Dynamic Workload Handling".
- secondary_problems: 0-3 additional distinct problems the paper also addresses, same concise
  phrase style. Empty list if there are none.

Base this ONLY on the retrieved chunks given - never invent a problem that isn't reflected in
them, and never use outside knowledge. If no clear research problem is stated in the given
chunks, use "Not Found in Retrieved Context" for main_problem and an empty list for
secondary_problems.

Respond with ONLY a JSON object: {"papers": [{"index": 1, "main_problem": "...",
"secondary_problems": ["...", "..."]}]}
One entry per paper given, "index" matching the 1-based position below. No markdown fences, no
commentary."""


def _retrieve_problem_evidence(papers: List[Paper], rag_cache: Dict[str, PaperIndex]) -> Dict[int, str]:
    """{paper index -> assembled retrieved-chunk context}, via each
    paper's already-built FAISS index - no PDF fetch here at all, this is
    pure (fast, local) retrieval against rag_cache."""
    contexts: Dict[int, str] = {}
    for i, paper in enumerate(papers):
        paper_index = rag_cache.get(paper.title, PaperIndex(paper_id=paper.title))
        context, _ = retrieve_context(
            paper_index,
            queries=[(config.RAG_QUERY_PROBLEM_SOLUTION, config.RAG_TOP_K_PROBLEM_SOLUTION)],
        )
        contexts[i] = context
    return contexts


def _build_problem_extraction_message(papers: List[Paper], contexts: Dict[int, str]) -> str:
    blocks = []
    for i, p in enumerate(papers):
        context = contexts.get(i, "")
        blocks.append(
            f"Paper {i + 1}: {p.title}\nRetrieved chunks:\n{context or '(no relevant chunks retrieved)'}"
        )
    return "\n\n".join(blocks)


def _extract_problems(
    papers: List[Paper], rag_cache: Dict[str, PaperIndex],
) -> Tuple[List[PaperProblems], str, str]:
    """One PaperProblems entry per paper, same order, plus a
    (pipeline_status, reason) pair explaining WHY the batch came back
    empty when it does - so analyze_problems_and_solutions() (and
    ultimately the UI) can tell "quota exhausted" apart from "genuinely
    no evidence" apart from "the LLM call itself failed", instead of
    every one of those collapsing into the same empty result."""
    empty = [PaperProblems() for _ in papers]
    if not papers:
        return empty, pipeline_status.INVALID_INPUT, "No papers were selected."
    if not llm.is_available():
        return empty, pipeline_status.LLM_UNAVAILABLE, pipeline_status.MESSAGES[pipeline_status.LLM_UNAVAILABLE]
    if llm.is_daily_quota_exhausted():
        return empty, pipeline_status.DAILY_QUOTA_EXCEEDED, pipeline_status.MESSAGES[pipeline_status.DAILY_QUOTA_EXCEEDED]

    contexts = _retrieve_problem_evidence(papers, rag_cache)
    if not any(c.strip() for c in contexts.values()):
        logger.warning("Problem extraction: RAG retrieved zero usable chunks for all %d paper(s).", len(papers))
        return empty, pipeline_status.RETRIEVAL_FAILURE, pipeline_status.MESSAGES[pipeline_status.RETRIEVAL_FAILURE]

    try:
        content = llm._call_groq(
            messages=[
                {"role": "system", "content": PROBLEM_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": _build_problem_extraction_message(papers, contexts)},
            ],
            max_tokens=min(4000, 400 * len(papers)),
            temperature=0.1,
            json_mode=True,
            agent_name="ProblemSolutionExtraction",
            paper_id=f"batch[{len(papers)}]",
            cache_version=config.LLM_CACHE_VERSION,
        )
    except llm.GroqDailyQuotaExceeded as exc:
        logger.warning("Problem extraction: daily Groq quota exhausted - %s", exc)
        return empty, pipeline_status.DAILY_QUOTA_EXCEEDED, pipeline_status.MESSAGES[pipeline_status.DAILY_QUOTA_EXCEEDED]
    data = llm.parse_json(content) if content else None
    if not isinstance(data, dict) or not isinstance(data.get("papers"), list):
        if content:
            logger.warning("Problem extraction returned unusable JSON: %s", content[:200])
        return empty, pipeline_status.EXTRACTION_FAILURE, pipeline_status.MESSAGES[pipeline_status.EXTRACTION_FAILURE]

    by_index: Dict[int, dict] = {}
    for entry in data["papers"]:
        if not isinstance(entry, dict):
            continue
        try:
            by_index[int(entry.get("index")) - 1] = entry
        except (TypeError, ValueError):
            continue

    results = list(empty)
    for i in range(len(papers)):
        entry = by_index.get(i)
        if not entry:
            continue
        main = str(entry.get("main_problem") or "").strip() or NOT_REPORTED
        secondary_raw = entry.get("secondary_problems")
        secondary = [str(s).strip() for s in secondary_raw if str(s).strip()] if isinstance(secondary_raw, list) else []
        results[i] = PaperProblems(main_problem=main, secondary_problems=secondary)

    if not any(pp.main_problem != NOT_REPORTED for pp in results):
        # The LLM call succeeded and returned valid JSON, but genuinely
        # found no clear problem statement in the retrieved evidence for
        # ANY paper - a real (if uncommon) outcome, distinct from every
        # failure mode above, and worth telling the UI apart from those.
        return results, pipeline_status.INSUFFICIENT_EVIDENCE, pipeline_status.MESSAGES[pipeline_status.INSUFFICIENT_EVIDENCE]
    return results, pipeline_status.SUCCESS, ""


# --------------------------------------------------------------------------
# Step 2: semantic grouping of problem phrases (shared embedding-based
# clustering, no LLM cost) + one small batched LLM call to give each
# cluster a clean canonical name.
# --------------------------------------------------------------------------

PROBLEM_LABEL_SYSTEM_PROMPT = """You receive groups of short research-problem phrases from \
different papers, already clustered because they refer to the same underlying problem. For EACH \
group, write "problem_label": a concise (2-5 word) canonical Title Case name for the shared \
problem (e.g. "High Latency", "Low Resource Utilization", "SLA Violation") - pick or paraphrase a \
clean representative name, do not just concatenate the inputs.

Respond with ONLY a JSON object: {"clusters": [{"index": 1, "problem_label": "..."}]}
One entry per group given, "index" matching the 1-based position below. No markdown fences, no
commentary."""


def _build_cluster_message(clusters: List[List[Tuple[str, str]]]) -> str:
    blocks = []
    for i, cluster in enumerate(clusters, start=1):
        member_lines = "\n".join(f"- ({label}) {text}" for label, text in cluster)
        blocks.append(f"Group {i}:\n{member_lines}")
    return "\n\n".join(blocks)


def _label_problem_clusters(clusters: List[List[Tuple[str, str]]]) -> List[str]:
    """One label string per cluster, same order. Falls back to the first
    member's raw text if the LLM is unavailable/unparsable."""
    fallback = [cluster[0][1] for cluster in clusters]
    if not clusters or not llm.is_available() or llm.is_daily_quota_exhausted():
        return fallback

    try:
        content = llm._call_groq(
            messages=[
                {"role": "system", "content": PROBLEM_LABEL_SYSTEM_PROMPT},
                {"role": "user", "content": _build_cluster_message(clusters)},
            ],
            max_tokens=min(2000, 60 * len(clusters)),
            temperature=0.2,
            json_mode=True,
            agent_name="ProblemClusterLabeling",
            paper_id=f"clusters[{len(clusters)}]",
            cache_version=config.LLM_CACHE_VERSION,
        )
    except llm.GroqDailyQuotaExceeded as exc:
        logger.warning("Problem cluster labeling: daily Groq quota exhausted - %s", exc)
        return fallback
    data = llm.parse_json(content) if content else None
    by_index: Dict[int, dict] = {}
    if isinstance(data, dict) and isinstance(data.get("clusters"), list):
        for entry in data["clusters"]:
            if isinstance(entry, dict):
                try:
                    by_index[int(entry.get("index"))] = entry
                except (TypeError, ValueError):
                    continue
    elif content:
        logger.warning("Problem cluster labeling returned unusable JSON: %s", content[:200])

    labels = []
    for i, default in enumerate(fallback, start=1):
        entry = by_index.get(i)
        label = str(entry.get("problem_label") or "").strip() if entry else ""
        labels.append(label or default)
    return labels


# --------------------------------------------------------------------------
# Step 3 & 4: solution mapping - pure deterministic aggregation over the
# ALREADY-EXTRACTED knowledge_table (no new fetch, no new LLM call).
# --------------------------------------------------------------------------

_SPLIT_RE = re.compile(r",| / |/| and |;")


def _tokenize(value: str) -> List[str]:
    return [t.strip() for t in _SPLIT_RE.split(value) if t.strip() and t.strip().lower() != NOT_REPORTED.lower()]


def _distinct_tokens(paper_titles: set, knowledge_by_title: Dict[str, dict], column: str) -> str:
    seen: List[str] = []
    seen_lower = set()
    for title in paper_titles:
        row = knowledge_by_title.get(title)
        if not row:
            continue
        for token in _tokenize(row.get(column, NOT_REPORTED)):
            key = token.lower()
            if key not in seen_lower:
                seen_lower.add(key)
                seen.append(token)
    return ", ".join(seen) if seen else NOT_REPORTED


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _best_performance(paper_titles: set, knowledge_by_title: Dict[str, dict]) -> str:
    best_num, best_display = -1.0, None
    for title in paper_titles:
        row = knowledge_by_title.get(title)
        if not row:
            continue
        for metric_col in _METRIC_COLUMNS:
            value = row.get(metric_col, NOT_REPORTED)
            if value == NOT_REPORTED:
                continue
            match = _NUMBER_RE.search(value)
            if not match:
                continue
            num = float(match.group(0))
            if num > best_num:
                best_num = num
                best_display = f"{value} ({metric_col})"
    return best_display or NOT_REPORTED


def _build_problem_solution_table(
    papers: List[Paper], paper_problems: List[PaperProblems], knowledge_table: List[dict],
) -> List[dict]:
    knowledge_by_title = {row.get("Paper Title", ""): row for row in knowledge_table}

    items: List[Tuple[str, str]] = []
    for paper, pp in zip(papers, paper_problems):
        if pp.main_problem != NOT_REPORTED:
            items.append((paper.title, pp.main_problem))
        for secondary in pp.secondary_problems:
            items.append((paper.title, secondary))

    clusters = cluster_texts(items, config.PROBLEM_CLUSTER_SIMILARITY_THRESHOLD)
    if not clusters:
        return []

    labels = _label_problem_clusters(clusters)

    rows = []
    for cluster, label in zip(clusters, labels):
        paper_titles = sorted({title for title, _ in cluster})
        rows.append({
            "Research Problem": label,
            "Frequency": len(paper_titles),
            "AI Techniques Used": _distinct_tokens(set(paper_titles), knowledge_by_title, _COL_AI_TECHNIQUE),
            "AI Models Used": _distinct_tokens(set(paper_titles), knowledge_by_title, _COL_AI_MODEL),
            "Best Reported Performance": _best_performance(set(paper_titles), knowledge_by_title),
            "Supporting Papers": ", ".join(paper_titles),
            "_solution_diversity": len(set(
                t.lower() for title in paper_titles
                for t in _tokenize(knowledge_by_title.get(title, {}).get(_COL_AI_TECHNIQUE, NOT_REPORTED))
                + _tokenize(knowledge_by_title.get(title, {}).get(_COL_AI_MODEL, NOT_REPORTED))
            )),
        })

    rows.sort(key=lambda r: r["Frequency"], reverse=True)
    return rows


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def _build_summary(table: List[dict], overall_analysis: List[dict]) -> List[dict]:
    if not table:
        return [{"Category": c, "Value": "No data available"} for c in (
            "Most Common Research Problem", "Most Used AI Technique", "Most Used AI Model",
            "Problem With Most Proposed Solutions", "Problem That Remains Least Solved",
        )]

    overall_by_category = {row.get("Category"): row.get("Most Common") for row in overall_analysis}

    most_common_problem = max(table, key=lambda r: r["Frequency"])
    most_solutions = max(table, key=lambda r: r["_solution_diversity"])
    least_solved = min(table, key=lambda r: r["_solution_diversity"])

    return [
        {"Category": "Most Common Research Problem",
         "Value": f"{most_common_problem['Research Problem']} ({most_common_problem['Frequency']} papers)"},
        {"Category": "Most Used AI Technique",
         "Value": overall_by_category.get("Most Used AI Technique", NOT_REPORTED)},
        {"Category": "Most Used AI Model",
         "Value": overall_by_category.get("Most Used AI Model", NOT_REPORTED)},
        {"Category": "Problem With Most Proposed Solutions",
         "Value": f"{most_solutions['Research Problem']} ({most_solutions['_solution_diversity']} distinct techniques/models)"},
        {"Category": "Problem That Remains Least Solved",
         "Value": f"{least_solved['Research Problem']} ({least_solved['_solution_diversity']} distinct techniques/models)"},
    ]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

@dataclass
class ProblemSolutionAnalysis:
    problem_solution_table: List[dict] = field(default_factory=list)
    summary: List[dict] = field(default_factory=list)
    # See pipeline_status.py - lets app.py show WHY this is empty instead
    # of a single generic "No data extracted" message for every cause.
    status: str = pipeline_status.SUCCESS
    status_reason: str = ""


def analyze_problems_and_solutions(
    papers: List[Paper], knowledge_table: List[dict], overall_analysis: List[dict],
    rag_cache: Dict[str, PaperIndex],
) -> ProblemSolutionAnalysis:
    """Entry point used by app.py, right after knowledge_analysis.analyze_papers().
    `knowledge_table`/`overall_analysis` are that call's own outputs, reused
    here at zero extra fetch/LLM cost for the solution side. `rag_cache` is
    the shared {paper title -> PaperIndex} built once in app.py."""
    if not papers:
        return ProblemSolutionAnalysis(
            status=pipeline_status.INVALID_INPUT,
            status_reason=pipeline_status.MESSAGES[pipeline_status.INVALID_INPUT],
        )

    paper_problems, extract_status, extract_reason = _extract_problems(papers, rag_cache)
    table = _build_problem_solution_table(papers, paper_problems, knowledge_table)
    summary = _build_summary(table, overall_analysis)

    # Strip the internal bookkeeping field before handing rows to the UI.
    public_table = [{k: v for k, v in row.items() if not k.startswith("_")} for row in table]

    if public_table:
        final_status, final_reason = pipeline_status.SUCCESS, ""
    else:
        # Table came back empty - report exactly why (quota/retrieval/
        # extraction failure, or genuinely no evidence) rather than
        # silently returning an empty list the UI can't distinguish.
        final_status, final_reason = extract_status, extract_reason

    return ProblemSolutionAnalysis(
        problem_solution_table=public_table, summary=summary,
        status=final_status, status_reason=final_reason,
    )
