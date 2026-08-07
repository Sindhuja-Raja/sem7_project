"""
Streamlit entry point for the Research Paper Retrieval System.

Pipeline (fully real-time, no database):

  user query
      -> LLM Query Understanding Agent: intent extraction +
         per-source search query generation      (query_understanding.py)
      -> parallel retrieval from 5 academic APIs  (retrieve.py)
      -> duplicate removal (DOI / title / authors/year)(utils.py)
      -> dense embedding + cosine similarity ranking   (embedding.py)
      -> cross-encoder re-ranking of the shortlist      (rerank.py)
      -> LLM summary + relevance explanation per result (llm.py)
      -> Per candidate paper, in validation-gated order:
         extract complete PDF text (PyMuPDF, cached to
         disk)                                       (pdf_extraction.py)
         -> Paper Validation Agent: PDF readable, minimum
            word count, meaningful (non-degenerate) content
            - deliberately lenient, never rejects a paper
            for a missing section - checked on the raw
            extracted text, BEFORE any chunking/embedding/
            FAISS work is ever spent on a paper that's
            going to be rejected                        (paper_validation.py)
         -> only if VALID: chunk + embed + FAISS-index,
            shared by every analysis agent below         (rag_pipeline.py)
         An INVALID candidate is discarded and automatically
         replaced by the next-ranked candidate, repeating
         until display_n VALID papers are collected or the
         ranked candidate pool is exhausted                (app.py)
      -> LLM knowledge extraction + aggregation,
         via RAG retrieval, not full-PDF text       (knowledge_analysis.py)
      -> Problem-Solution Analysis Agent: literature
         synthesis across all papers together,
         via RAG retrieval                          (problem_solution_analysis.py)
      -> render in the UI

  On demand, after the user selects 2+ of the results above:
      -> Agentic AI Contradiction Detection, via
         RAG retrieval (reuses the same index)      (contradiction_detection.py)
      -> Root Cause Analysis, for any detected
         Contradiction / Partial Contradiction,
         via RAG retrieval                          (root_cause_analysis.py)

  On demand, after everything above (uses whatever of it has run so far):
      -> Inventor Agent: evidence-based improved
         solution recommendation, with supplementary
         RAG retrieval                              (inventor_agent.py)
"""

import os

# HuggingFace model cache check:
# Only force offline mode if the primary embedding model is already cached locally.
# When models are not cached, this allows them to be downloaded from HuggingFace.
# setdefault() so an explicit environment override (HF_HUB_OFFLINE=1 in .env) wins.
# Must be evaluated before sentence_transformers/transformers/huggingface_hub are
# imported anywhere - hence this sits above every other import in this entry-point.
def _hf_models_are_cached() -> bool:
    """Returns True if the primary HuggingFace models are already cached locally"""
    import subprocess
    try:
        result = subprocess.run(
            ["python", "-c",
             "from sentence_transformers import SentenceTransformer; "
             "SentenceTransformer('BAAI/bge-large-en-v1.5', local_files_only=True); print('ok')"],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except Exception:
        return False

if os.environ.get("HF_HUB_OFFLINE", "") not in ("1", "true", "True"):
    # Only force offline mode if models are actually cached; otherwise allow download
    if _hf_models_are_cached():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # else: allow HuggingFace to download models on first run

import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import llm
from contradiction_detection import (
    ContradictionReport, classify_pairs, extract_claims, match_claim_pairs, predict_contradiction_pairs,
)
from embedding import cluster_texts, embed_texts, rank_by_similarity
from inventor_agent import GENERAL_BEST_PRACTICE, SUPPORTED_BY_LITERATURE, generate_recommendation
from knowledge_analysis import analyze_papers
from paper_validation import INVALID, VALID, PaperValidationResult, validate_paper_text
from pdf_extraction import get_full_paper_text
from problem_solution_analysis import analyze_problems_and_solutions
from query_understanding import build_source_queries, understand_query
from rag_pipeline import (
    NOT_FOUND_IN_CONTEXT, PaperIndex, build_paper_index_from_sections, get_chunk_embedding_model,
)
from rerank import rerank_papers
from retrieve import fetch_all_sources, filter_papers_by_metadata
from root_cause_analysis import analyze_root_causes
from utils import Paper, deduplicate_papers
import ui_theme

try:
    _hero_metric_row = ui_theme.hero_metric_row
except AttributeError:
    import importlib

    ui_theme = importlib.reload(ui_theme)
    _hero_metric_row = getattr(ui_theme, "hero_metric_row", ui_theme.metric_row)

logger = logging.getLogger(__name__)

_COVERAGE_FIELDS = ["Dataset", "Optimizer", "Loss Function", "Hyperparameters", "Hardware", "Evaluation Metrics"]
_TOKEN_SPLIT_RE = re.compile(r",| / |/| and |;")


def _tokenize(value: str) -> List[str]:
    return [t.strip() for t in _TOKEN_SPLIT_RE.split(value or "") if t.strip() and t.strip().lower() != NOT_FOUND_IN_CONTEXT.lower()]

st.set_page_config(
    page_title="Research Paper Retrieval System",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass
class PaperCollectionResult:
    valid_papers: List[Paper]
    rag_cache: Dict[str, PaperIndex]
    validation_results: List[PaperValidationResult]
    retrieved_count: int      # candidates actually attempted (RAG-indexed + validated)
    rejected_count: int       # of those, how many failed validation
    replaced_count: int       # valid papers that had to come from beyond the first
                               # target_count-sized window, i.e. successful backfills


def _extract_validate_and_index(paper: Paper) -> Tuple[Paper, PaperValidationResult, Optional[PaperIndex]]:
    """One candidate paper, in validation-gated order:
        1. extract complete PDF text (pdf_extraction.get_full_paper_text -
           PyMuPDF, all pages, cleaned, cached to disk)
        2. validate the raw extracted text (paper_validation.
           validate_paper_text - independent of the RAG pipeline, knows
           nothing about chunking/embeddings/FAISS)
        3. ONLY if VALID: chunk + embed + FAISS-index
           (rag_pipeline.build_paper_index_from_sections)
    An INVALID paper is rejected at step 2 and never reaches step 3, so
    chunking/embedding/FAISS-building - real, non-trivial compute - is
    never spent on a paper that's going to be discarded. Never raises."""
    try:
        sections = get_full_paper_text(paper)
    except Exception as exc:  # noqa: BLE001 - one paper's failure must never break the batch
        logger.error("Full-text extraction FAILED for %s: %s", paper.title[:60], exc)
        sections = {}

    result = validate_paper_text(paper, sections)
    if result.status != VALID:
        return paper, result, None

    try:
        paper_index = build_paper_index_from_sections(paper, sections)
    except Exception as exc:  # noqa: BLE001 - one paper's failure must never break the batch
        logger.error("Chunking/embedding/FAISS indexing FAILED for %s: %s", paper.title[:60], exc)
        downgraded = PaperValidationResult(
            paper_title=paper.title, status=INVALID, reasons=[f"Indexing error: {exc}"],
        )
        return paper, downgraded, None

    return paper, result, paper_index


def _collect_valid_papers(candidates: List[Paper], target_count: int, status) -> PaperCollectionResult:
    """Runs _extract_validate_and_index over `candidates` (already
    similarity/cross-encoder ranked, most relevant first) in batches,
    discarding INVALID papers (PDF unreadable, too little extracted text,
    or degenerate/garbled content - never a missing section) BEFORE any
    chunking/embedding/FAISS work, and automatically continuing to the
    next-ranked candidate, until `target_count` VALID papers are
    collected or `candidates` is exhausted. A paper failing validation
    never shrinks the final result set below what was requested, as long
    as enough ranked candidates remain to try."""
    valid_papers: List[Paper] = []
    rag_cache: Dict[str, PaperIndex] = {}
    validation_results: List[PaperValidationResult] = []
    rejected_count = 0
    replaced_count = 0
    batch_size = max(target_count, 8)  # process at least 8 per batch to amortize overhead
    idx = 0
    batch_num = 0

    get_chunk_embedding_model()  # warm the cache on the main thread so the
                                  # concurrent workers below never race the first load

    while len(valid_papers) < target_count and idx < len(candidates):
        batch_num += 1
        batch = candidates[idx: idx + batch_size]
        idx += len(batch)

        status.write(
            f"{'Retrieving replacement candidates' if batch_num > 1 else 'Extracting & validating'} "
            f"{len(batch)} paper(s) (batch {batch_num}) - chunking/embedding/FAISS only runs for "
            "papers that pass validation..."
        )
        with ThreadPoolExecutor(max_workers=min(8, len(batch))) as executor:
            batch_outcomes = list(executor.map(_extract_validate_and_index, batch))

        batch_valid_count = 0
        for paper, result, paper_index in batch_outcomes:
            validation_results.append(result)
            if result.status == VALID and paper_index is not None:
                rag_cache[paper.title] = paper_index
                batch_valid_count += 1
                if len(valid_papers) < target_count:
                    valid_papers.append(paper)
                    if batch_num > 1:
                        replaced_count += 1

        rejected_count += len(batch) - batch_valid_count
        status.write(
            f"Batch {batch_num}: {batch_valid_count} VALID, {len(batch) - batch_valid_count} INVALID "
            f"(discarded before any chunking/embedding/FAISS work). {len(valid_papers)} of "
            f"{target_count} valid papers collected so far."
        )

    if len(valid_papers) < target_count:
        status.write(
            f"⚠️ No more relevant candidates available - stopped with {len(valid_papers)} of "
            f"{target_count} valid papers."
        )

    return PaperCollectionResult(
        valid_papers=valid_papers, rag_cache=rag_cache, validation_results=validation_results,
        retrieved_count=idx, rejected_count=rejected_count, replaced_count=replaced_count,
    )


def _generate_paper_ai_fields(paper: Paper, query: str) -> None:
    """Generates ai_summary and ai_reason for one paper in-place.
    Designed to run concurrently inside a ThreadPoolExecutor."""
    paper.ai_summary = llm.summarize_paper(paper.title, paper.abstract)
    paper.ai_reason = llm.explain_relevance(query, paper.title, paper.abstract)


def run_pipeline(
    query: str, display_n: int, results_per_source: int, top_k: int,
    year_from: Optional[int] = None, year_to: Optional[int] = None,
    publishers: Optional[List[str]] = None,
) -> None:
    """Execute the full retrieve -> dedupe -> rank -> rerank -> explain
    pipeline for `query` and stash the results in session_state.

    Args:
        query:              Raw user query string.
        display_n:          Number of valid papers to collect and show.
        results_per_source: Max papers fetched from each API source.
        top_k:              Candidates kept after embedding, before cross-encoder.
    """
    import config as _cfg
    # Temporarily override pipeline tuning with caller-supplied values
    _orig_rps = _cfg.RESULTS_PER_SOURCE
    _orig_topk = _cfg.TOP_K_AFTER_EMBEDDING
    _cfg.RESULTS_PER_SOURCE = results_per_source
    _cfg.TOP_K_AFTER_EMBEDDING = top_k

    try:
        with st.status("Running AI-powered retrieval pipeline...", expanded=True) as status:
            status.write("Understanding research intent (LLM Query Understanding Agent)...")
            qu = understand_query(query)
            if qu.llm_generated:
                status.write(
                    f"Domain: **{qu.research_domain}** | Problem: **{qu.research_problem}** | "
                    f"Primary technique: **{qu.primary_ai_technique}**"
                )
                status.write(f"Search intent: *{qu.search_intent}*")
                status.write(f"Generated boolean query: `{qu.search_query}`")
            else:
                status.write("GROQ_API_KEY not set - falling back to the raw title as the search query.")

            source_queries = build_source_queries(qu)
            status.write(
                f"Retrieving up to **{results_per_source}** papers per source from OpenAlex, Crossref, "
                f"arXiv, CORE and Semantic Scholar in parallel..."
            )
            raw_papers, source_status = fetch_all_sources(source_queries)
            status.write(f"Retrieved {len(raw_papers)} raw records. " +
                         " | ".join(f"{src}: {s}" for src, s in source_status.items()))

            status.write("Removing duplicates (DOI, title similarity, authors, year)...")
            unique_papers = deduplicate_papers(raw_papers)
            status.write(f"{len(unique_papers)} unique papers after deduplication.")

            if year_from is not None or year_to is not None or publishers:
                status.write("Applying metadata filters (year range / publisher)...")
                unique_papers = filter_papers_by_metadata(unique_papers, year_from, year_to, publishers)
                status.write(f"{len(unique_papers)} papers remain after metadata filtering.")

            status.write(f"Computing dense semantic similarity — keeping top {top_k} candidates...")
            similarity_ranked = rank_by_similarity(query, unique_papers)
            shortlist = similarity_ranked[:top_k]

            status.write(f"Re-ranking {len(shortlist)} candidates with cross-encoder...")
            reranked = rerank_papers(query, shortlist)

            status.write(
                f"Collecting {display_n} VALID papers from {len(reranked)} ranked candidates — "
                f"invalid papers are auto-replaced by the next-ranked candidate..."
            )
            collection = _collect_valid_papers(reranked, display_n, status)
            top_papers = collection.valid_papers
            rag_cache = collection.rag_cache
            validation_results = collection.validation_results
            valid_papers = top_papers
            total_chunks = sum(len(idx.chunks) for idx in rag_cache.values())
            status.write(
                f"Papers retrieved: {collection.retrieved_count} | rejected: {collection.rejected_count} | "
                f"replaced: {collection.replaced_count} | final valid papers: {len(top_papers)}. "
                f"Indexed {total_chunks} chunks across {len(rag_cache)} papers."
            )

            # ── Parallel AI summaries + relevance explanations ─────────────
            if llm.is_available():
                if top_papers:
                    status.write(
                        f"Generating AI summaries and relevance explanations for {len(top_papers)} papers "
                        f"(parallel Groq calls)..."
                    )
                    _llm_workers = min(4, len(top_papers))  # max 4 concurrent Groq calls
                    with ThreadPoolExecutor(max_workers=_llm_workers) as _ex:
                        list(_ex.map(lambda p: _generate_paper_ai_fields(p, query), top_papers))
                else:
                    status.write("No papers remain after filtering, so AI summaries and relevance explanations are skipped.")
            else:
                status.write("GROQ_API_KEY not set - showing results without AI summaries/explanations.")
                for paper in top_papers:
                    paper.ai_summary = llm.summarize_paper(paper.title, paper.abstract)
                    paper.ai_reason = llm.explain_relevance(query, paper.title, paper.abstract)

            status.write("Running knowledge extraction (AI model, dataset, gaps, future work) "
                          "across VALID results only, from RAG-retrieved chunks only...")
            knowledge = analyze_papers(valid_papers, rag_cache, max_papers=display_n)

            status.write("Running Problem-Solution Analysis across all VALID results...")
            problem_solution = analyze_problems_and_solutions(
                valid_papers, knowledge.knowledge_table, knowledge.overall_analysis, rag_cache,
            )

            status.update(label="Done.", state="complete", expanded=False)
    finally:
        # Always restore original config values
        _cfg.RESULTS_PER_SOURCE = _orig_rps
        _cfg.TOP_K_AFTER_EMBEDDING = _orig_topk

    st.session_state.results = top_papers
    st.session_state.valid_papers = valid_papers
    st.session_state.paper_validation_results = validation_results
    st.session_state.paper_collection_stats = {
        "retrieved": collection.retrieved_count,
        "rejected": collection.rejected_count,
        "replaced": collection.replaced_count,
        "final_valid": len(top_papers),
        "target": display_n,
    }
    st.session_state.last_query = query
    st.session_state.source_status = source_status
    st.session_state.query_understanding = qu
    st.session_state.knowledge_analysis = knowledge
    st.session_state.problem_solution_analysis = problem_solution
    st.session_state.rag_index_cache = rag_cache
    st.session_state.total_unique = len(unique_papers)
    # A new search invalidates any in-progress contradiction-detection
    # selection/report from the previous result set.
    st.session_state.contradiction_report = None
    st.session_state.contradiction_root_causes = None
    st.session_state.contradiction_comparison_table = None
    st.session_state.pop("contradiction_selection", None)
    st.session_state.inventor_recommendation = None


def render_paper(rank: int, paper: Paper, validation_status: Optional[str] = None) -> None:
    with st.container(border=True):
        # ── Title row ──────────────────────────────────────────────────────
        header_cols = st.columns([6, 1], vertical_alignment="center")
        header_cols[0].markdown(f"#### {rank}. {paper.title}")
        if validation_status == VALID:
            header_cols[1].markdown(
                ui_theme.chip("✓ Valid", "#22c55e"), unsafe_allow_html=True,
            )
        elif validation_status == INVALID:
            header_cols[1].markdown(
                ui_theme.chip("✗ Invalid", "#ef4444"), unsafe_allow_html=True,
            )

        # ── Metadata chips ─────────────────────────────────────────────────
        meta_parts = []
        if paper.authors_display():
            meta_parts.append(f":material/person: {paper.authors_display()}")
        if paper.year:
            meta_parts.append(f":material/calendar_today: {paper.year}")
        meta_parts.append(f":material/database: {paper.source}")
        if paper.doi:
            meta_parts.append(f":material/link: {paper.doi}")
        st.caption("  ·  ".join(meta_parts))

        # ── Score bar ──────────────────────────────────────────────────────
        score_cols = st.columns(2)
        score_cols[0].metric(
            ":material/analytics: Semantic similarity", f"{paper.similarity_score:.3f}",
        )
        score_cols[1].metric(
            ":material/psychology: Cross-encoder relevance", f"{paper.rerank_score:.3f}",
        )

        # ── Abstract ───────────────────────────────────────────────────────
        with st.expander(":material/article: Abstract", expanded=False):
            st.write(paper.abstract or "No abstract available.")

        # ── AI insights ────────────────────────────────────────────────────
        if paper.ai_summary and paper.ai_summary != "N/A":
            st.markdown(f"**:material/auto_awesome: AI summary** · {paper.ai_summary}")
        if paper.ai_reason and paper.ai_reason != "N/A":
            st.markdown(f"**:material/lightbulb: Why relevant** · {paper.ai_reason}")

        # ── PDF link ───────────────────────────────────────────────────────
        if paper.pdf_url:
            st.link_button(
                ":material/open_in_new: Open PDF / source", paper.pdf_url,
            )
        else:
            st.caption("No PDF link available from the source API.")


def render_validation_tab(validation_results: List) -> None:
    """Paper Validation Agent report - runs automatically as part of the
    search pipeline, before any chunking/embedding/FAISS work. Every
    downstream module (Knowledge Extraction, Problem-Solution,
    Contradictions, Root Cause, Ideas) only ever receives the papers
    marked VALID here."""
    ui_theme.section_header(
        "✅", "Paper Validation",
        "Every retrieved paper is checked before any chunking or knowledge extraction: PDF readable, "
        "extracted text above the minimum word count, and meaningful (non-degenerate) content - "
        "deliberately lenient, a paper is never rejected just for missing a specific section "
        "(e.g. Future Work or Conclusion). INVALID papers are excluded from every downstream module "
        "below - one paper failing validation never stops the rest.",
    )
    if not validation_results:
        st.info("Run a search first.")
        return

    valid_count = sum(1 for r in validation_results if r.status == VALID)
    total = len(validation_results)
    ui_theme.metric_row([
        ("Total Papers", total),
        ("Valid", valid_count),
        ("Invalid", total - valid_count),
    ])

    for r in validation_results:
        with st.container(border=True):
            header_cols = st.columns([5, 2])
            header_cols[0].markdown(f"**{r.paper_title}**")
            color = "#22c55e" if r.status == VALID else "#ef4444"
            header_cols[1].markdown(ui_theme.chip(r.status, color), unsafe_allow_html=True)
            st.caption(f"Word count: {r.word_count}  •  Sections found: {', '.join(r.sections_found) or 'None'}")
            if r.reasons:
                for reason in r.reasons:
                    st.markdown(f"- {reason}")

    with st.expander("Show as table"):
        render_table("Validation Report", [r.to_row() for r in validation_results])


def render_overview_strip(
    query: str, results: List[Paper], knowledge, problem_solution, rag_cache: dict,
) -> None:
    """Always-visible summary banner above the tabs."""
    total_chunks = sum(len(idx.chunks) for idx in (rag_cache or {}).values())
    st.markdown(
        f"""
        <div class="rs-hero">
            <p style="margin:0 0 14px 0;font-size:0.82rem;opacity:0.65;text-transform:uppercase;
                letter-spacing:0.07em;">
                :material/search: Search query
            </p>
            <p style="margin:0 0 18px 0;font-size:1.1rem;font-weight:600;">{query or '—'}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _hero_metric_row([
        ("Papers found", st.session_state.get("total_unique", len(results))),
        ("Papers displayed", len(results)),
        ("Chunks indexed", total_chunks),
        ("LLM", "✓ Online" if llm.is_available() else "✗ Off"),
    ])
    stats = st.session_state.get("paper_collection_stats")
    if stats:
        st.caption("Validation backfill — invalid papers auto-replaced by next-ranked candidate")
        ui_theme.metric_row([
            ("Retrieved", stats["retrieved"]),
            ("Rejected", stats["rejected"]),
            ("Replaced", stats["replaced"]),
            ("Final valid", f"{stats['final_valid']} / {stats['target']}"),
        ])
    st.caption("Agent pipeline status")
    ui_theme.agent_status_row([
        ("Retrieval & ranking", bool(results)),
        ("RAG index", bool(rag_cache)),
        ("Paper validation", st.session_state.get("paper_validation_results") is not None),
        ("Knowledge extraction", bool(knowledge and knowledge.knowledge_table)),
        ("Problem-solution", bool(problem_solution and problem_solution.problem_solution_table)),
        ("Contradiction detection", st.session_state.get("contradiction_report") is not None),
        ("Inventor agent", st.session_state.get("inventor_recommendation") is not None),
    ])


def main() -> None:
    ui_theme.inject_css()

    # ── Hero banner ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(14,165,233,0.12) 100%);
            border: 1px solid rgba(99,102,241,0.30);
            border-radius: 16px;
            padding: 28px 32px 22px 32px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        ">
            <div style="position:absolute;top:0;left:0;right:0;height:3px;
                background:linear-gradient(90deg,#6366f1,#0ea5e9,#14b8a6);
                border-radius:16px 16px 0 0;"></div>
            <h1 style="margin:0 0 6px 0;font-size:2rem;font-weight:800;
                background:linear-gradient(90deg,#a5b4fc,#7dd3fc);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                :material/menu_book: Research Paper Retrieval System
            </h1>
            <p style="margin:0;font-size:0.92rem;opacity:0.72;max-width:820px;">
                Real-time, database-free retrieval from OpenAlex, Crossref, arXiv, CORE"""
        + (" and Semantic Scholar" if config.ENABLE_SEMANTIC_SCHOLAR else "")
        + """ — ranked with dense embeddings, cross-encoder re-ranking and LLM explanations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### :material/tune: Settings")

        speed_mode = st.radio(
            "Speed mode",
            options=["⚡ Fast", "⚖️ Balanced", "🔬 Deep"],
            index=1,
            help=(
                "**⚡ Fast** — 50 papers/source, top-50 re-ranked. Best for quick exploration.\n\n"
                "**⚖️ Balanced** — 100 papers/source, top-150 re-ranked. Good quality + speed.\n\n"
                "**🔬 Deep** — 200 papers/source, top-500 re-ranked. Maximum coverage (slowest)."
            ),
        )
        _speed_config = {
            "⚡ Fast":     {"rps": 50,  "topk": 300},
            "⚖️ Balanced": {"rps": 100, "topk": 400},
            "🔬 Deep":     {"rps": 200, "topk": 600},
        }[speed_mode]
        results_per_source = _speed_config["rps"]
        top_k_rerank      = _speed_config["topk"]

        display_n = st.slider(
            "Papers to display",
            min_value=5,
            max_value=50,
            value=10,
            help=(
                "How many valid papers to collect and display. "
                "Lower = faster (fewer PDFs downloaded and fewer Groq calls). "
                f"Current mode fetches up to {results_per_source} papers/source."
            ),
        )

        st.caption(
            f"Mode: **{speed_mode}** · {results_per_source} papers/source · "
            f"top-{top_k_rerank} re-ranked · {display_n} displayed"
        )

        year_from = st.number_input("Published from year", min_value=1900, max_value=2100, value=None, format="%d")
        year_to = st.number_input("Published to year", min_value=1900, max_value=2100, value=None, format="%d")
        publisher_filter = st.text_input(
            "Publisher filter",
            placeholder="e.g. IEEE, Springer",
            help="Optional publisher filter to reduce retrieval noise and speed up downstream processing.",
        )
        publisher_values = [p.strip() for p in publisher_filter.split(",") if p.strip()]

        st.markdown("### :material/sensors: API status")
        # LLM
        if llm.is_available():
            st.success(f"Groq LLM · `{config.GROQ_MODEL}`", icon=":material/check_circle:")
        else:
            st.info(
                "Groq LLM · not configured — AI summaries, knowledge extraction, "
                "and all analysis features are disabled.  \n"
                "**[Get free key →](https://console.groq.com/keys)**  \n"
                "Add `GROQ_API_KEY=your_key` to `.env`",
                icon=":material/info:",
            )
        # CORE
        if config.CORE_API_KEY:
            st.success("CORE API · configured", icon=":material/check_circle:")
        else:
            st.info(
                "CORE API · not configured, this source is skipped.  \n"
                "**[Get free key →](https://core.ac.uk/services/api)**",
                icon=":material/info:",
            )
        # Semantic Scholar
        if config.ENABLE_SEMANTIC_SCHOLAR:
            if config.SEMANTIC_SCHOLAR_API_KEY:
                st.success("Semantic Scholar · key set", icon=":material/check_circle:")
            else:
                st.info(
                    "Semantic Scholar · public rate limit (works, but may be slow).  \n"
                    "**[Request API key →](https://www.semanticscholar.org/product/api)**",
                    icon=":material/info:",
                )
        else:
            st.caption("Semantic Scholar disabled (rate-limit issues)")
        st.success("OpenAlex · no key needed", icon=":material/check_circle:")
        st.success("Crossref · no key needed", icon=":material/check_circle:")
        st.success("arXiv · no key needed", icon=":material/check_circle:")


    # ── Search bar ─────────────────────────────────────────────────────────
    search_col, btn_col = st.columns([5, 1], vertical_alignment="bottom")
    with search_col:
        query = st.text_input(
            "Research topic or title",
            placeholder='e.g. "Agentic AI for Cloud Resource Allocation"',
            label_visibility="collapsed",
        )
    with btn_col:
        search_clicked = st.button(
            ":material/search: Search", type="primary", width="stretch",
        )

    if "results" not in st.session_state:
        st.session_state.results = None

    if search_clicked:
        if not query.strip():
            st.warning("Please enter a research paper title or topic.")
        else:
            run_pipeline(
                query.strip(), display_n, results_per_source, top_k_rerank,
                year_from=int(year_from) if year_from not in (None, "") else None,
                year_to=int(year_to) if year_to not in (None, "") else None,
                publishers=publisher_values,
            )

    qu = st.session_state.get("query_understanding")
    if qu is not None:
        with st.expander(":material/manage_search: Query understanding (extracted research intent)", expanded=False):
            st.json(qu.to_dict())

    results: List[Paper] = st.session_state.results
    if results is not None:
        if not results:
            st.info(
                "No papers found for this query. Try a broader topic or different phrasing.",
                icon=":material/search_off:",
            )
        else:
            knowledge = st.session_state.get("knowledge_analysis")
            problem_solution = st.session_state.get("problem_solution_analysis")
            rag_cache = st.session_state.get("rag_index_cache")
            valid_papers: List[Paper] = st.session_state.get("valid_papers") or []
            validation_results = st.session_state.get("paper_validation_results") or []

            render_overview_strip(
                st.session_state.get("last_query", ""), results, knowledge, problem_solution, rag_cache,
            )

            (
                tab_papers, tab_validation, tab_overview, tab_knowledge, tab_problem_solution,
                tab_contradictions, tab_root_cause, tab_ideas, tab_risk, tab_proposal,
            ) = st.tabs([
                ":material/article: Papers",
                ":material/verified: Validation",
                ":material/analytics: Overview",
                ":material/psychology: Knowledge",
                ":material/join_inner: Problems & Solutions",
                ":material/balance: Contradictions",
                ":material/troubleshoot: Root cause",
                ":material/lightbulb: Ideas",
                ":material/warning: Risk indicators",
                ":material/description: Proposal",
            ])

            with tab_papers:
                st.caption(
                    f"Showing **{len(results)}** of "
                    f"{st.session_state.get('total_unique', len(results))} unique papers found — "
                    f"ranked by semantic similarity then cross-encoder relevance."
                )
                status_by_title = {r.paper_title: r.status for r in validation_results}
                for i, paper in enumerate(results, start=1):
                    render_paper(i, paper, status_by_title.get(paper.title))

            with tab_validation:
                render_validation_tab(validation_results)

            with tab_overview:
                render_overview_analytics_tab(knowledge, valid_papers)

            with tab_knowledge:
                render_knowledge_extraction_tab(knowledge)

            with tab_problem_solution:
                render_problem_solution_analysis(problem_solution)

            with tab_contradictions:
                render_contradiction_detection(valid_papers)

            with tab_root_cause:
                render_root_cause_tab()

            with tab_ideas:
                render_inventor_agent(valid_papers)

            with tab_risk:
                render_risk_indicators_tab()

            with tab_proposal:
                render_proposal_preview_tab()


def render_table(title: str, rows: List[dict]) -> None:
    st.subheader(title)
    if not rows:
        st.info("No data extracted for this table.")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch")


def render_overall_analysis(rows: List[dict]) -> None:
    """Renders every category's full frequency breakdown (all extracted
    technologies/models/datasets/etc., most-used first) instead of
    collapsing each category down to a single "Most Common" table row -
    so a category with no repeated value still lists everything that was
    actually found, rather than a bare "No Common Pattern Found"."""
    st.subheader("Overall Analysis")
    if not rows:
        st.info("No data extracted for this table.")
        return
    for entry in rows:
        st.markdown(f"**{entry['Category']}:**")
        items = entry.get("Items") or []
        if not items:
            value = entry.get("Most Common", "Not Found in Retrieved Context")
            evidence = entry.get("Evidence")
            st.markdown(f"- {value}" + (f" ({evidence})" if evidence else ""))
        else:
            st.markdown("\n".join(f"- {item['Value']} ({item['Papers']})" for item in items))


def _items_for(overall_analysis: List[dict], category: str) -> List[dict]:
    for entry in overall_analysis:
        if entry["Category"] == category:
            return entry.get("Items") or []
    return []


def _technology_landscape_treemap(overall_analysis: List[dict]):
    """One treemap: root -> category -> value, sized by paper count. Skips
    "Highest Reported Accuracy" (not a frequency breakdown)."""
    ids, labels, parents, values = ["root"], ["Technology Landscape"], [""], [0]
    total = 0
    for entry in overall_analysis:
        category = entry["Category"]
        if category == "Highest Reported Accuracy":
            continue
        items = entry.get("Items") or []
        if not items:
            continue
        cat_id = f"cat::{category}"
        cat_total = sum(i["Count"] for i in items)
        ids.append(cat_id)
        labels.append(category)
        parents.append("root")
        values.append(cat_total)
        total += cat_total
        for item in items[:12]:
            ids.append(f"{cat_id}::{item['Value']}")
            labels.append(item["Value"])
            parents.append(cat_id)
            values.append(item["Count"])
    if total == 0:
        return None
    values[0] = total
    return ui_theme.plotly_treemap(labels, parents, values, ids=ids, title="Technology Landscape", height=440)


def _coverage_heatmap(knowledge_table: List[dict]):
    """Rows = papers, columns = the same technical fields inventor_agent's
    field_coverage counts in aggregate - this is that same signal's
    per-paper drill-down. Cell = 1 (reported) / 0 (Not Found in Retrieved
    Context)."""
    if not knowledge_table:
        return None
    y_labels = [row.get("Paper Title", "")[:60] for row in knowledge_table]
    z, hover = [], []
    for row in knowledge_table:
        z_row, hover_row = [], []
        for field_name in _COVERAGE_FIELDS:
            reported = str(row.get(field_name, NOT_FOUND_IN_CONTEXT)).strip() not in ("", NOT_FOUND_IN_CONTEXT)
            z_row.append(1 if reported else 0)
            hover_row.append("Reported" if reported else "Not Found in Retrieved Context")
        z.append(z_row)
        hover.append(hover_row)
    return ui_theme.plotly_heatmap(
        z, _COVERAGE_FIELDS, y_labels, colorscale=[[0, "#ef4444"], [1, "#22c55e"]],
        hover_text=hover, title="Field Coverage Heatmap",
    )


def _trend_timeline(knowledge_table: List[dict]):
    """Stacked bar of AI Technique frequency by publication Year - which
    techniques are gaining traction over time. Top 6 techniques by total
    frequency only, so the legend stays readable."""
    year_technique_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: Dict[str, int] = defaultdict(int)
    years = set()
    for row in knowledge_table:
        try:
            year = int(str(row.get("Year", "")).strip())
        except (TypeError, ValueError):
            continue
        years.add(year)
        for technique in _tokenize(row.get("AI Technique", NOT_FOUND_IN_CONTEXT)):
            year_technique_counts[year][technique] += 1
            totals[technique] += 1
    if not years or not totals:
        return None

    sorted_years = sorted(years)
    top_techniques = [t for t, _ in sorted(totals.items(), key=lambda x: x[1], reverse=True)[:6]]
    series = {t: [year_technique_counts[y].get(t, 0) for y in sorted_years] for t in top_techniques}
    return ui_theme.plotly_stacked_bar(
        sorted_years, series, title="Research Trend: AI Technique by Year",
        x_title="Year", y_title="Papers",
    )


def _paper_embeddings(papers: List[Paper]):
    if len(papers) < 2:
        return None
    return embed_texts([p.text_for_embedding() for p in papers], is_query=False)


def _paper_similarity_network_figure(papers: List[Paper], threshold: float = config.PROBLEM_CLUSTER_SIMILARITY_THRESHOLD):
    """Nodes = displayed papers, edges = pairwise cosine similarity above
    `threshold`. Reuses embedding.embed_texts (already L2-normalized, so a
    plain dot product is the cosine similarity) - no new model, no backend
    change, just a fresh similarity computation over the small displayed
    set."""
    vecs = _paper_embeddings(papers)
    if vecs is None:
        return None
    sims = vecs @ vecs.T
    nodes = [
        {"id": f"paper::{i}", "label": p.title[:40], "hover": p.title, "color": ui_theme.category_color(0), "size": 22}
        for i, p in enumerate(papers)
    ]
    edges = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            sim = float(sims[i, j])
            if sim >= threshold:
                edges.append({
                    "source": f"paper::{i}", "target": f"paper::{j}",
                    "width": 1 + 3 * sim, "color": "#94a3b8", "label": f"similarity {sim:.2f}",
                })
    if not edges:
        return None
    return ui_theme.plotly_network(nodes, edges, title="Paper Similarity Network", height=480)


def _topic_clusters_figure(papers: List[Paper]):
    """2D PCA projection of the same paper embeddings, colored by
    embedding.cluster_texts' existing greedy clustering (reusing the same
    threshold research-gap/future-work clustering already uses) - a visual
    map of how the displayed papers group by topic."""
    if len(papers) < 2:
        return None
    vecs = _paper_embeddings(papers)
    if vecs is None:
        return None
    clusters = cluster_texts(
        [(p.title, p.text_for_embedding()) for p in papers], config.KNOWLEDGE_GAP_CLUSTER_SIMILARITY_THRESHOLD,
    )
    cluster_of_title = {title: ci for ci, cluster in enumerate(clusters) for title, _ in cluster}

    from sklearn.decomposition import PCA
    coords = PCA(n_components=2, random_state=42).fit_transform(vecs)
    cluster_ids = [cluster_of_title.get(p.title, 0) for p in papers]
    return ui_theme.plotly_scatter_clusters(
        coords[:, 0].tolist(), coords[:, 1].tolist(), [p.title for p in papers], cluster_ids,
        title="Topic Clusters", height=460,
    )


def render_overview_analytics_tab(knowledge, papers: List[Paper]) -> None:
    """Cross-paper aggregation, visualized: Technology Landscape treemap,
    per-category ranking bars, Research Domain donut, Field Coverage
    Heatmap, Research Trend Timeline, a keyword tag-cloud, and (cross-
    cutting, not tied to one backend table) a Paper Similarity Network and
    Topic Clusters map - replacing the old flat "Overall Analysis" table.
    Research Gap / Future Work / Research Insights stay as tables for now
    (their own visual redesign is a later step)."""
    ui_theme.section_header(
        "📊", "Overview & Analytics",
        "Cross-paper aggregation - most-used technique/model/dataset/framework, research gaps, "
        "and future-work themes across every analyzed paper, computed deterministically wherever "
        "possible.",
    )
    if knowledge is None:
        st.info("Run a search first.")
        return
    if not llm.is_available():
        st.warning("GROQ_API_KEY not set - these tables are empty.")
        return

    rows = knowledge.overall_analysis
    if not rows:
        st.info("No data extracted for this section.")
        return

    landscape = _technology_landscape_treemap(rows)
    if landscape:
        st.plotly_chart(landscape, width="stretch")
    else:
        st.info("No technology data was extracted across the analyzed papers.")

    st.markdown("**Most Used, By Category**")
    ranking_cols = st.columns(3)
    for col, (label, category) in zip(ranking_cols, [
        ("AI Technique", "Most Used AI Technique"),
        ("Dataset", "Most Used Dataset"),
        ("Framework", "Most Used Framework"),
    ]):
        items = _items_for(rows, category)[:8]
        with col:
            if items:
                fig = ui_theme.plotly_bar_ranking(
                    [i["Value"] for i in items], [i["Count"] for i in items], title=label, height=280,
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption(f"No {label} data extracted.")

    domain_items = _items_for(rows, "Most Common Research Domain")
    accuracy_entry = next((r for r in rows if r["Category"] == "Highest Reported Accuracy"), None)
    donut_col, accuracy_col = st.columns([3, 2])
    with donut_col:
        if domain_items:
            fig = ui_theme.plotly_donut(
                [i["Value"] for i in domain_items], [i["Count"] for i in domain_items],
                title="Research Domain Distribution", height=320,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No Research Domain data extracted.")
    with accuracy_col:
        st.markdown("**Highest Reported Accuracy**")
        if accuracy_entry:
            st.metric("Best Result", accuracy_entry.get("Most Common", NOT_FOUND_IN_CONTEXT))
            st.caption(accuracy_entry.get("Evidence", ""))
        else:
            st.caption("Not available.")

    coverage_fig = _coverage_heatmap(knowledge.knowledge_table)
    if coverage_fig:
        st.plotly_chart(coverage_fig, width="stretch")

    trend_fig = _trend_timeline(knowledge.knowledge_table)
    if trend_fig:
        st.plotly_chart(trend_fig, width="stretch")

    with st.expander("🔤 Keyword Cloud"):
        cloud_cols = st.columns(3)
        cloud_specs = [
            ("AI Techniques", "Most Used AI Technique"),
            ("Datasets", "Most Used Dataset"),
            ("Frameworks", "Most Used Framework"),
        ]
        for idx, (col, (label, category)) in enumerate(zip(cloud_cols, cloud_specs)):
            with col:
                st.caption(label)
                items = _items_for(rows, category)
                ui_theme.tag_cloud(
                    [(item["Value"], item["Count"]) for item in items], color=ui_theme.category_color(idx),
                )

    with st.expander("Show detailed breakdown (all extracted values)"):
        render_overall_analysis(rows)

    st.markdown("**Paper Similarity Network**")
    similarity_fig = _paper_similarity_network_figure(papers)
    if similarity_fig:
        st.plotly_chart(similarity_fig, width="stretch")
    else:
        st.caption("No paper pairs are similar enough to draw an edge.")

    st.markdown("**Topic Clusters**")
    clusters_fig = _topic_clusters_figure(papers)
    if clusters_fig:
        st.plotly_chart(clusters_fig, width="stretch")

    render_table("Research Gap Analysis", knowledge.research_gap_analysis)
    render_table("Future Work Analysis", knowledge.future_work_analysis)
    render_table("Research Insights", knowledge.research_insights)


_TECHNIQUE_MODEL_FIELDS = [
    "AI Technique", "AI Model", "Model Architecture", "LLM", "Agent Framework", "Embedding Model",
]
_SETUP_FIELDS = [
    "Dataset", "Preprocessing", "Optimizer", "Loss Function", "Learning Rate", "Epochs",
    "Hyperparameters", "Training Strategy", "Programming Language", "Framework", "Hardware",
]
_METRIC_FIELDS = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "mAP"]
_NARRATIVE_FIELDS = ["Objectives", "Key Contributions", "Research Gap", "Limitations", "Future Work"]


def _evidence_source_color(source: str) -> str:
    """The knowledge_table "Evidence Source" value is a full label like
    "RAG (14 chunks from Full PDF)" or "RAG (1 chunk from Abstract
    fallback)" (rag_pipeline.PaperIndex.source_label), not a bare
    "Full PDF"/"Abstract fallback" - match by substring, not equality."""
    if "Full PDF" in source:
        return ui_theme.SOURCE_COLORS["Full PDF"]
    if "Abstract fallback" in source:
        return ui_theme.SOURCE_COLORS["Abstract fallback"]
    return ui_theme.SOURCE_COLORS["Not Available"]


def _row_chip_values(row: dict, fields: List[str]) -> List[str]:
    """Reported values for `fields`, split into individual chip tokens
    (reuses the same tokenization the backend uses for its own frequency
    counting, so a chip here always matches how it was counted there)."""
    values: List[str] = []
    for field_name in fields:
        raw = str(row.get(field_name, NOT_FOUND_IN_CONTEXT)).strip()
        if not raw or raw == NOT_FOUND_IN_CONTEXT:
            continue
        tokens = _tokenize(raw)
        values.extend(tokens if tokens else [raw])
    return values


def render_knowledge_extraction_tab(knowledge) -> None:
    """One expandable card per paper: chip rows grouped by theme
    (Technique & Model, Data & Setup), stat tiles for reported metrics, and
    narrative fields as text - replacing the old flat, 26-column table.
    Retrieved-chunk evidence nests inside each card instead of a separate
    always-visible global table."""
    ui_theme.section_header(
        "🧠", "Knowledge Extraction",
        "Extraction is RAG-based: only the chunks retrieved as most relevant to each task are "
        "sent to the LLM - never the whole PDF - falling back to a single abstract chunk only if "
        "the PDF can't be downloaded or parsed. See each card's evidence-source chip for which "
        "was used.",
    )
    if knowledge is None:
        st.info("Run a search first.")
        return
    if not llm.is_available():
        st.warning("GROQ_API_KEY not set - knowledge extraction is empty.")
        return
    if not knowledge.knowledge_table:
        st.info("No data extracted for this section.")
        return

    chunks_by_paper: Dict[str, List[dict]] = defaultdict(list)
    for chunk_row in knowledge.retrieved_chunks:
        chunks_by_paper[chunk_row["Paper Title"]].append(chunk_row)

    for row in knowledge.knowledge_table:
        paper_title = row.get("Paper Title", "Untitled")
        source = row.get("Evidence Source", "Not Available")
        source_color = _evidence_source_color(source)

        with st.container(border=True):
            header_cols = st.columns([5, 2])
            header_cols[0].markdown(f"### {paper_title}")
            header_cols[1].markdown(ui_theme.chip(source, source_color), unsafe_allow_html=True)

            domain = row.get("Research Domain", NOT_FOUND_IN_CONTEXT)
            caption = f"Year: {row.get('Year', NOT_FOUND_IN_CONTEXT)}"
            if domain and domain != NOT_FOUND_IN_CONTEXT:
                caption += f"  •  Domain: {domain}"
            st.caption(caption)

            technique_chips = _row_chip_values(row, _TECHNIQUE_MODEL_FIELDS)
            if technique_chips:
                st.markdown("**Technique & Model**")
                ui_theme.chip_row(technique_chips, color=ui_theme.category_color(0))

            setup_chips = _row_chip_values(row, _SETUP_FIELDS)
            if setup_chips:
                st.markdown("**Data & Setup**")
                ui_theme.chip_row(setup_chips, color=ui_theme.category_color(1))

            metric_items = [
                (m, row.get(m)) for m in _METRIC_FIELDS
                if row.get(m, NOT_FOUND_IN_CONTEXT) != NOT_FOUND_IN_CONTEXT
            ]
            if metric_items:
                st.markdown("**Reported Results**")
                metric_cols = st.columns(len(metric_items))
                for col, (label, value) in zip(metric_cols, metric_items):
                    col.metric(label, value)

            for label in _NARRATIVE_FIELDS:
                value = row.get(label, NOT_FOUND_IN_CONTEXT)
                if value and value != NOT_FOUND_IN_CONTEXT:
                    st.markdown(f"**{label}:** {value}")

            paper_chunks = chunks_by_paper.get(paper_title, [])
            with st.expander(f"Show retrieved evidence ({len(paper_chunks)} chunks)"):
                if paper_chunks:
                    st.dataframe(pd.DataFrame(paper_chunks), width="stretch")
                else:
                    st.caption("No chunks retrieved for this paper.")

    with st.expander("Show full Knowledge Retrieval Table (all papers, all fields)"):
        render_table("Knowledge Retrieval Table", knowledge.knowledge_table)


def render_problem_solution_analysis(analysis) -> None:
    """Problem-Solution Analysis Agent - literature synthesis across all
    displayed papers together (never a per-paper summary). Runs
    automatically as part of the search pipeline, positioned between
    Knowledge Retrieval and (on-demand) Contradiction Detection."""
    ui_theme.section_header(
        "🧩", "Problem ↔ Solution",
        "Common research problems synthesized across all retrieved papers together, mapped to "
        "the AI techniques/models used to address each one.",
    )
    if analysis is None:
        st.info("Run a search first.")
        return
    if not llm.is_available():
        st.warning("GROQ_API_KEY not set - problem-solution analysis is empty.")
        return
    if not analysis.problem_solution_table:
        st.info("No data extracted for this section.")
        return

    table = analysis.problem_solution_table
    fig = ui_theme.plotly_bar_ranking(
        [row["Research Problem"] for row in table], [row["Frequency"] for row in table],
        title="Research Problems by Frequency", height=max(320, 40 * len(table) + 60),
    )
    st.plotly_chart(fig, width="stretch")

    for row in table:
        freq = row["Frequency"]
        with st.expander(f"{row['Research Problem']}  ·  {freq} paper{'s' if freq != 1 else ''}"):
            technique_chips = _tokenize(row.get("AI Techniques Used", NOT_FOUND_IN_CONTEXT))
            if technique_chips:
                st.markdown("**AI Techniques Used**")
                ui_theme.chip_row(technique_chips, color=ui_theme.category_color(0))

            model_chips = _tokenize(row.get("AI Models Used", NOT_FOUND_IN_CONTEXT))
            if model_chips:
                st.markdown("**AI Models Used**")
                ui_theme.chip_row(model_chips, color=ui_theme.category_color(1))

            st.metric("Best Reported Performance", row.get("Best Reported Performance", NOT_FOUND_IN_CONTEXT))

            supporting = [p.strip() for p in row.get("Supporting Papers", "").split(", ") if p.strip()]
            if supporting:
                st.markdown("**Supporting Papers**")
                ui_theme.chip_row(supporting, color=ui_theme.category_color(2))

    if analysis.summary:
        st.markdown("**Summary**")
        summary_cols = st.columns(len(analysis.summary))
        for col, entry in zip(summary_cols, analysis.summary):
            col.metric(entry["Category"], entry["Value"])


_CLASSIFICATION_ORDER = [
    "Agreement", "Different Context", "Insufficient Evidence", "Partial Contradiction", "Contradiction",
]


def _short_paper_label(label: str) -> str:
    return label.split(":", 1)[-1].strip()[:40]


def _conflict_matrix_figure(table: List[dict]):
    """Paper x Paper heatmap - cell = the most severe classification found
    between that pair (Contradiction outranks Partial outranks Different
    Context/Insufficient Evidence outranks Agreement), so a paper pair
    compared on several topics still surfaces its worst disagreement."""
    if not table:
        return None
    code = {c: i for i, c in enumerate(_CLASSIFICATION_ORDER)}
    papers = sorted({row["Paper A"] for row in table} | {row["Paper B"] for row in table})
    if len(papers) < 2:
        return None
    idx = {p: i for i, p in enumerate(papers)}
    n = len(papers)
    z = [[None] * n for _ in range(n)]
    hover = [[""] * n for _ in range(n)]
    for row in table:
        i, j = idx[row["Paper A"]], idx[row["Paper B"]]
        c = code.get(row["Classification"], code["Insufficient Evidence"])
        text = f"{row['Compared Topic']}: {row['Classification']}"
        for a, b in ((i, j), (j, i)):
            if z[a][b] is None or c > z[a][b]:
                z[a][b] = c
                hover[a][b] = text

    n_cats = len(_CLASSIFICATION_ORDER)
    colorscale = []
    for i, cat in enumerate(_CLASSIFICATION_ORDER):
        color = ui_theme.CLASSIFICATION_COLORS[cat]
        colorscale.append([i / n_cats, color])
        colorscale.append([(i + 1) / n_cats, color])

    labels = [_short_paper_label(p) for p in papers]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, text=hover,
        hovertemplate="%{y} vs %{x}<br>%{text}<extra></extra>",
        colorscale=colorscale, zmin=0, zmax=n_cats, showscale=False, xgap=2, ygap=2,
    ))
    fig.update_layout(
        xaxis=dict(side="top"), margin=dict(l=10, r=10, t=60, b=10),
        height=max(320, 40 * n + 100), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _contradiction_sankey_figure(table: List[dict]):
    """Paper A -> Topic -> Paper B flow, link color = classification -
    topics that recur across several paper pairs converge into one node,
    surfacing which topics generate the most cross-paper disagreement."""
    if not table:
        return None
    papers = sorted({row["Paper A"] for row in table} | {row["Paper B"] for row in table})
    topics = sorted({row["Compared Topic"] for row in table})
    if not papers or not topics:
        return None

    labels = [_short_paper_label(p) for p in papers] + list(topics)
    paper_idx = {p: i for i, p in enumerate(papers)}
    topic_idx = {t: len(papers) + i for i, t in enumerate(topics)}

    sources, targets, values, colors = [], [], [], []
    for row in table:
        color = ui_theme.CLASSIFICATION_COLORS.get(row["Classification"], "#94a3b8")
        sources.append(paper_idx[row["Paper A"]])
        targets.append(topic_idx[row["Compared Topic"]])
        values.append(1)
        colors.append(color)
        sources.append(topic_idx[row["Compared Topic"]])
        targets.append(paper_idx[row["Paper B"]])
        values.append(1)
        colors.append(color)

    node_colors = [ui_theme.category_color(0)] * len(papers) + [ui_theme.category_color(3)] * len(topics)
    return ui_theme.plotly_sankey(
        labels, sources, targets, values, link_colors=colors, node_colors=node_colors,
        title="Paper → Topic → Paper Contradiction Flow",
        height=max(400, 30 * (len(papers) + len(topics))),
    )


def _render_evidence_tree(comparisons) -> None:
    """Classification -> (Paper A vs Paper B on Topic) -> Claim A / Claim B
    / Supporting Evidence, nested expanders - replaces the old flat
    "Contradiction Details" table, and (unlike that table, which only
    covered Contradiction/Partial Contradiction) covers every classification
    so Agreements are visible too, not just conflicts."""
    by_classification: Dict[str, list] = defaultdict(list)
    for c in comparisons:
        by_classification[c.classification].append(c)

    for classification in reversed(_CLASSIFICATION_ORDER):
        items = by_classification.get(classification, [])
        if not items:
            continue
        color = ui_theme.CLASSIFICATION_COLORS.get(classification, "#94a3b8")
        st.markdown(ui_theme.chip(f"{classification} ({len(items)})", color), unsafe_allow_html=True)
        for c in items:
            with st.expander(f"{c.topic}  —  {_short_paper_label(c.paper_a)} vs {_short_paper_label(c.paper_b)}"):
                st.markdown(f"**{c.paper_a}**\n\n{c.claim_a}")
                st.markdown(f"**{c.paper_b}**\n\n{c.claim_b}")
                st.caption(f"Confidence: {c.confidence}%  •  Reason: {c.reason}")
                if c.supporting_evidence:
                    st.markdown("**Supporting Evidence**")
                    for evidence in c.supporting_evidence:
                        st.markdown(f"- {evidence}")


def render_contradiction_detection(papers: List[Paper]) -> None:
    """Agentic AI Contradiction Detection - on-demand comparison of 2+
    user-selected papers, run independently of the automatic search
    pipeline above. Does not touch anything in the Knowledge Retrieval
    section."""
    ui_theme.section_header(
        "⚖️", "Contradictions",
        "Select two or more of the retrieved papers to extract their scientific claims "
        "(methods, findings, performance, conclusions, advantages, limitations) and check "
        "for agreements, contradictions, or insufficient evidence between them.",
    )

    if len(papers) < 2:
        st.info("Need at least 2 retrieved papers to compare.")
        return

    labels = [f"Paper {i}: {p.title}" for i, p in enumerate(papers, start=1)]
    label_to_paper = dict(zip(labels, papers))

    selected_labels = st.multiselect(
        "Select papers to compare (2 or more)", labels, key="contradiction_selection",
    )
    detect_clicked = st.button("Detect Contradictions", key="detect_contradictions_btn")

    if detect_clicked:
        if len(selected_labels) < 2:
            st.warning("Select at least 2 papers before detecting contradictions.")
        elif not llm.is_available():
            st.warning("GROQ_API_KEY not set - contradiction detection requires the LLM.")
        else:
            selected_papers = [label_to_paper[label] for label in selected_labels]
            rag_cache = st.session_state.get("rag_index_cache") or {}
            with st.status("Running contradiction detection...", expanded=True) as status:
                status.write(
                    f"Retrieving claim-relevant chunks for {len(selected_papers)} selected papers "
                    "from their already-built RAG index..."
                )
                claims_by_paper = extract_claims(selected_papers, selected_labels, rag_cache)
                total_claims = sum(len(c) for c in claims_by_paper.values())
                status.write(f"Extracted {total_claims} scientific claims across the selected papers.")

                empty_labels = [selected_labels[i] for i, c in claims_by_paper.items() if not c]
                if empty_labels:
                    status.write(
                        "⚠️ No retrievable chunks (no abstract, and no indexable PDF) for: "
                        + ", ".join(empty_labels) + " - these contribute no claims."
                    )

                status.write("Matching claims across papers by topic (dense embedding similarity, "
                              "so unrelated claims are never compared)...")
                matched_pairs = match_claim_pairs(claims_by_paper)
                status.write(f"{len(matched_pairs)} claim pairs discuss the same topic and will be compared.")

                status.write("Generating a lightweight contradiction-risk prediction for each matched pair...")
                predictions = predict_contradiction_pairs(matched_pairs)
                status.write("Classifying each matched pair (Agreement / Contradiction / "
                              "Partial Contradiction / Different Context / Insufficient Evidence)...")
                comparisons = classify_pairs(matched_pairs)

                flagged = [c for c in comparisons if c.classification in ("Contradiction", "Partial Contradiction")]
                if flagged:
                    status.write(f"Running Root Cause Analysis on {len(flagged)} detected "
                                  "contradiction(s) (comparing dataset, setup, model, hyperparameters, "
                                  "hardware, and more between the two papers)...")
                else:
                    status.write("No contradictions or partial contradictions detected - "
                                  "skipping Root Cause Analysis.")
                root_causes = analyze_root_causes(comparisons, selected_papers, selected_labels, rag_cache)

                status.update(label="Done.", state="complete", expanded=False)

            st.session_state.contradiction_report = ContradictionReport(
                comparisons=comparisons,
                claims_extracted=total_claims,
                predictions=predictions,
            )
            st.session_state.contradiction_root_causes = root_causes

    report: ContradictionReport = st.session_state.get("contradiction_report")
    if report is None:
        return

    table = report.comparison_table()
    root_causes = st.session_state.get("contradiction_root_causes") or []
    # Append Root Cause Analysis as three extra columns on the existing
    # comparison table, per-row - contradiction_detection.py's own
    # comparison_table() output is unchanged, this only extends the
    # copy of it that gets displayed (and later reused by the Inventor Agent).
    for row, rc in zip(table, root_causes):
        row["Root Cause"] = rc.root_cause_category
        row["Root Cause Explanation"] = rc.root_cause_explanation
        row["Root Cause Confidence"] = rc.confidence
    st.session_state.contradiction_comparison_table = table

    if not table:
        st.info("No comparable claim pairs were found across the selected papers.")
    else:
        summary = report.summary()
        st.markdown("**Summary**")
        st.metric("Total Comparisons", summary["Total Comparisons"])
        ui_theme.mixed_chip_row([
            (f"{cls} ({summary[cls]})", ui_theme.CLASSIFICATION_COLORS[cls])
            for cls in _CLASSIFICATION_ORDER if summary.get(cls)
        ])

        st.markdown("**Conflict Matrix**")
        matrix_fig = _conflict_matrix_figure(table)
        if matrix_fig:
            st.plotly_chart(matrix_fig, width="stretch")
        else:
            st.caption("Need at least 2 distinct papers in the comparisons to draw a matrix.")

        st.markdown("**Contradiction Flow**")
        sankey_fig = _contradiction_sankey_figure(table)
        if sankey_fig:
            st.plotly_chart(sankey_fig, width="stretch")

        st.markdown("**Evidence Tree**")
        _render_evidence_tree(report.comparisons)

        with st.expander("Show as tables (Comparison Table / Contradiction Details)"):
            st.dataframe(pd.DataFrame(table), width="stretch")
            details = report.contradiction_details()
            if details:
                st.dataframe(pd.DataFrame(details), width="stretch")


def _evidence_badge_html(supporting_papers: str) -> str:
    """The validated recommendation/improvement rows don't carry a separate
    "Evidence Type" column - it's encoded in "Supporting Papers" itself:
    the literal GENERAL_BEST_PRACTICE string, or a real comma-joined list
    of paper titles otherwise (inventor_agent._resolve_evidence)."""
    if supporting_papers == GENERAL_BEST_PRACTICE:
        return ui_theme.chip("Best-Practice Suggestion", ui_theme.EVIDENCE_TYPE_COLORS[GENERAL_BEST_PRACTICE])
    return ui_theme.chip("Literature-Backed", ui_theme.EVIDENCE_TYPE_COLORS[SUPPORTED_BY_LITERATURE])


def _render_innovation_cards(rows: List[dict]) -> None:
    for row in rows:
        with st.container(border=True):
            header_cols = st.columns([5, 2])
            header_cols[0].markdown(f"#### {row['Suggested Improvement']}")
            header_cols[1].markdown(_evidence_badge_html(row["Supporting Papers"]), unsafe_allow_html=True)
            st.markdown(f"**Current Approach:** {row['Current Approach']}")
            st.markdown(f"**Why This Is Better:** {row['Why this is Better']}")
            st.markdown(f"**Expected Benefit:** {row['Expected Benefit']}")
            if row["Supporting Papers"] != GENERAL_BEST_PRACTICE:
                st.markdown("**Supporting Papers**")
                ui_theme.chip_row(
                    [p.strip() for p in row["Supporting Papers"].split(", ")], color=ui_theme.category_color(2),
                )


def _render_improvement_cards(rows: List[dict]) -> None:
    for row in rows:
        with st.container(border=True):
            header_cols = st.columns([5, 2])
            header_cols[0].markdown(f"#### {row['Problem']}")
            header_cols[1].markdown(_evidence_badge_html(row["Supporting Papers"]), unsafe_allow_html=True)
            st.markdown(f"**Current Observation:** {row['Current Observation']}")
            st.markdown(f"**Suggested Improvement:** {row['Suggested Improvement']}")
            st.markdown(f"**Expected Benefit:** {row['Expected Benefit']}")
            st.caption(row["Reason"])
            if row["Supporting Papers"] != GENERAL_BEST_PRACTICE:
                ui_theme.chip_row(
                    [p.strip() for p in row["Supporting Papers"].split(", ")], color=ui_theme.category_color(2),
                )


def _render_final_recommendation_hero(rows: List[dict]) -> None:
    by_category = {row["Category"]: row["Value"] for row in rows}
    title = by_category.pop("Suggested Research Title", None)
    with st.container(border=True):
        if title:
            st.markdown(f"### {title}")
        cols = st.columns(2)
        for i, (category, value) in enumerate(by_category.items()):
            cols[i % 2].markdown(f"**{category}**\n\n{value}")


def render_inventor_agent(papers: List[Paper]) -> None:
    """Inventor Agent - on-demand, positioned last since it synthesizes
    across every other analysis section's already-computed output (never
    reads papers or fetches PDFs itself)."""
    ui_theme.section_header(
        "💡", "Ideas & Recommendations",
        "Synthesizes Knowledge Retrieval, Problem-Solution Analysis and (if run) Contradiction "
        "Detection / Root Cause Analysis into an evidence-based recommendation for what a new "
        "researcher should build next. Not a per-paper summary, not a random idea. Missing data is "
        "never treated as a dead end - every gap is labeled 'Supported by Retrieved Literature' or "
        "'General AI Best Practice' and paired with a concrete, actionable improvement.",
    )

    knowledge = st.session_state.get("knowledge_analysis")
    problem_solution = st.session_state.get("problem_solution_analysis")
    if knowledge is None or problem_solution is None:
        st.info("Run a search first so Knowledge Retrieval and Problem-Solution Analysis have data to build on.")
        return

    generate_clicked = st.button("Generate Improved Solution Recommendation", key="inventor_agent_btn")

    if generate_clicked:
        if not llm.is_available():
            st.warning("GROQ_API_KEY not set - the Inventor Agent requires the LLM.")
        else:
            contradiction_table = st.session_state.get("contradiction_comparison_table") or []
            rag_cache = st.session_state.get("rag_index_cache") or {}
            with st.status("Running Inventor Agent...", expanded=True) as status:
                status.write("Synthesizing findings across Knowledge Retrieval, Problem-Solution "
                              "Analysis" + (" and Contradiction Detection" if contradiction_table else "") +
                              " (plus a little supplementary RAG-retrieved evidence per paper) "
                              "into an evidence-based recommendation...")
                recommendation = generate_recommendation(
                    papers,
                    knowledge.knowledge_table,
                    knowledge.overall_analysis,
                    knowledge.research_gap_analysis,
                    knowledge.future_work_analysis,
                    problem_solution.problem_solution_table,
                    problem_solution.summary,
                    contradiction_table,
                    rag_cache,
                )
                status.update(label="Done.", state="complete", expanded=False)

            st.session_state.inventor_recommendation = recommendation

    recommendation = st.session_state.get("inventor_recommendation")
    if recommendation is None:
        return

    if recommendation.insufficient_evidence:
        st.info(recommendation.insufficient_evidence_reason)
        return

    st.markdown("**Recommendations**")
    if recommendation.recommendations:
        _render_innovation_cards(recommendation.recommendations)
    else:
        st.caption("No recommendations generated.")

    st.markdown("**Improvement Table (What's Missing)**")
    if recommendation.improvement_table:
        _render_improvement_cards(recommendation.improvement_table)
    else:
        st.caption("No improvement items generated.")

    if recommendation.final_recommendation:
        st.markdown("**Final Recommendation**")
        _render_final_recommendation_hero(recommendation.final_recommendation)

    with st.expander("Show as tables"):
        render_table("Recommendation Table", recommendation.recommendations)
        render_table("Improvement Table", recommendation.improvement_table)
        render_table("Final Research Recommendation", recommendation.final_recommendation)


def _confidence_color(confidence: int) -> str:
    if confidence >= 70:
        return "#22c55e"
    if confidence >= 40:
        return "#f59e0b"
    return "#94a3b8"


def _fishbone_figure(paper_a: str, paper_b: str, topic: str, root_cause: str, evidence_a: str, evidence_b: str):
    """A single-bone cause diagram: spine = the contradiction (effect),
    one bone = the actual determined root cause, two leaves = the
    supporting evidence quoted for each paper. Root Cause Analysis only
    ever determines ONE category per pair (never a full multi-aspect
    breakdown) - this draws exactly that, not a fabricated multi-branch
    fishbone the underlying data doesn't support."""
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=8, y1=0, line=dict(color="#94a3b8", width=2))
    fig.add_annotation(
        x=8.2, y=0, text=f"<b>Effect</b><br>{paper_a} vs {paper_b}<br>({topic})",
        showarrow=False, xanchor="left", align="left", font=dict(size=11),
    )
    fig.add_shape(type="line", x0=4, y0=0, x1=4, y1=1.8, line=dict(color=ui_theme.CATEGORY_PALETTE[4], width=2))
    fig.add_annotation(
        x=4, y=2.2, text=f"<b>Root Cause</b><br>{root_cause}",
        showarrow=False, font=dict(size=12, color=ui_theme.CATEGORY_PALETTE[4]),
    )
    fig.add_shape(type="line", x0=2, y0=0, x1=2, y1=-1.5, line=dict(color="#64748b", width=1, dash="dot"))
    fig.add_annotation(
        x=2, y=-1.9, text=f"Evidence ({paper_a}):<br>{(evidence_a or '')[:100]}",
        showarrow=False, font=dict(size=9), align="center",
    )
    fig.add_shape(type="line", x0=6, y0=0, x1=6, y1=-1.5, line=dict(color="#64748b", width=1, dash="dot"))
    fig.add_annotation(
        x=6, y=-1.9, text=f"Evidence ({paper_b}):<br>{(evidence_b or '')[:100]}",
        showarrow=False, font=dict(size=9), align="center",
    )
    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.5, 13]), yaxis=dict(visible=False, range=[-3.2, 3.2]),
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _root_cause_network_figure(flagged_pairs):
    """flagged_pairs = [(ClaimComparison, RootCauseResult), ...]. Nodes =
    papers + root-cause categories, edges = "this pair's contradiction was
    caused by this category" - surfaces recurring root causes (e.g.
    "Different Dataset" explaining 3 separate contradictions) that no
    single row shows on its own."""
    if not flagged_pairs:
        return None
    papers = sorted({_short_paper_label(c.paper_a) for c, _ in flagged_pairs}
                     | {_short_paper_label(c.paper_b) for c, _ in flagged_pairs})
    categories = sorted({rc.root_cause_category for _, rc in flagged_pairs})
    nodes = (
        [{"id": f"p::{p}", "label": p, "color": ui_theme.category_color(0), "size": 24} for p in papers]
        + [{"id": f"c::{cat}", "label": cat, "color": ui_theme.category_color(4), "size": 32} for cat in categories]
    )
    edges = []
    for c, rc in flagged_pairs:
        cat_id = f"c::{rc.root_cause_category}"
        edges.append({"source": f"p::{_short_paper_label(c.paper_a)}", "target": cat_id, "color": "#94a3b8"})
        edges.append({"source": f"p::{_short_paper_label(c.paper_b)}", "target": cat_id, "color": "#94a3b8"})
    return ui_theme.plotly_network(nodes, edges, title="Root Cause Network", height=440)


def _root_cause_bubble_figure(flagged_pairs):
    """One bubble per distinct root_cause_category - x = average
    confidence across the contradictions it was chosen for, y/size = how
    many contradictions it explains."""
    if not flagged_pairs:
        return None
    by_category: Dict[str, list] = defaultdict(list)
    for _, rc in flagged_pairs:
        by_category[rc.root_cause_category].append(rc.confidence or 0)
    categories = sorted(by_category.keys())
    x = [sum(by_category[cat]) / len(by_category[cat]) for cat in categories]
    y = [len(by_category[cat]) for cat in categories]
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers+text", text=categories, textposition="top center",
        marker=dict(size=[16 + 10 * n for n in y], color=[ui_theme.category_color(i) for i in range(len(categories))]),
    ))
    fig.update_layout(
        xaxis=dict(title="Average Confidence", range=[0, 100]),
        yaxis=dict(title="Contradictions Explained", dtick=1),
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_root_cause_tab() -> None:
    """One fishbone-style cause card per flagged contradiction (with a real
    confidence gauge), plus an aggregate Root Cause Network and a
    frequency-vs-confidence bubble chart across all of them - reads
    directly from the ClaimComparison/RootCauseResult pairs
    Contradictions-tab detection already produced."""
    ui_theme.section_header(
        "🔎", "Root Cause Analysis",
        "For every Contradiction / Partial Contradiction found in the Contradictions tab, "
        "compares methodology, dataset, model, hyperparameters, hardware and more between the "
        "two papers to identify the single most probable reason they disagree.",
    )
    report: ContradictionReport = st.session_state.get("contradiction_report")
    root_causes = st.session_state.get("contradiction_root_causes")
    if report is None or root_causes is None:
        st.info(
            "Run Contradiction Detection in the Contradictions tab first - Root Cause Analysis "
            "runs automatically on any detected Contradiction / Partial Contradiction."
        )
        return

    flagged_pairs = [
        (c, rc) for c, rc in zip(report.comparisons, root_causes)
        if c.classification in ("Contradiction", "Partial Contradiction")
    ]
    if not flagged_pairs:
        st.info("No contradictions or partial contradictions were detected - nothing to analyze.")
        return

    for c, rc in flagged_pairs:
        with st.container(border=True):
            header_cols = st.columns([4, 1])
            header_cols[0].markdown(f"**{c.paper_a}** vs **{c.paper_b}**  —  _{c.topic}_")
            with header_cols[1]:
                confidence = rc.confidence or 0
                st.plotly_chart(
                    ui_theme.plotly_gauge(confidence, title="Confidence", color=_confidence_color(confidence), height=140),
                    width="stretch",
                )
            st.plotly_chart(
                _fishbone_figure(
                    _short_paper_label(c.paper_a), _short_paper_label(c.paper_b), c.topic,
                    rc.root_cause_category, rc.evidence_a, rc.evidence_b,
                ),
                width="stretch",
            )
            st.caption(rc.root_cause_explanation)

    st.markdown("**Root Cause Network**")
    network_fig = _root_cause_network_figure(flagged_pairs)
    if network_fig:
        st.plotly_chart(network_fig, width="stretch")

    st.markdown("**Root Cause Frequency vs Confidence**")
    bubble_fig = _root_cause_bubble_figure(flagged_pairs)
    if bubble_fig:
        st.plotly_chart(bubble_fig, width="stretch")


def _field_coverage_gap_pct(knowledge_table: List[dict]) -> Optional[float]:
    """Average, across the same fields inventor_agent's field_coverage
    counts, of what fraction of papers are missing that field."""
    if not knowledge_table:
        return None
    total = len(knowledge_table)
    ratios = []
    for field_name in _COVERAGE_FIELDS:
        missing = sum(
            1 for row in knowledge_table
            if str(row.get(field_name, NOT_FOUND_IN_CONTEXT)).strip() in ("", NOT_FOUND_IN_CONTEXT)
        )
        ratios.append(missing / total)
    return 100 * sum(ratios) / len(ratios)


def _evidence_reliability_risk_pct(knowledge_table: List[dict]) -> Optional[float]:
    """% of papers whose Evidence Source is anything other than "Full
    PDF" (i.e. abstract-only or unavailable) - RAG evidence grounded only
    in an abstract is inherently less reliable than full-text retrieval."""
    if not knowledge_table:
        return None
    total = len(knowledge_table)
    unreliable = sum(1 for row in knowledge_table if "Full PDF" not in str(row.get("Evidence Source", "")))
    return 100 * unreliable / total


def render_risk_indicators_tab() -> None:
    """Real-signal risk indicators - every axis is a number already
    computed elsewhere in this app (field coverage, evidence source,
    contradiction classification, root-cause confidence). Deliberately no
    single fabricated "risk score" - a radar of independently-meaningful,
    fully-traceable signals instead, explicitly labeled as not an ML
    prediction."""
    ui_theme.section_header(
        "⚠️", "Risk Indicators",
        "Real-signal risk indicators - not a machine-learned failure prediction. Each axis is "
        "computed directly from retrieval/analysis coverage already shown elsewhere in this app.",
    )
    knowledge = st.session_state.get("knowledge_analysis")
    if knowledge is None or not knowledge.knowledge_table:
        st.info("Run a search first.")
        return

    axes: List[str] = []
    values: List[float] = []

    coverage_risk = _field_coverage_gap_pct(knowledge.knowledge_table)
    if coverage_risk is not None:
        axes.append("Field Coverage Gaps")
        values.append(coverage_risk)

    evidence_risk = _evidence_reliability_risk_pct(knowledge.knowledge_table)
    if evidence_risk is not None:
        axes.append("Evidence Reliability Risk")
        values.append(evidence_risk)

    report: ContradictionReport = st.session_state.get("contradiction_report")
    if report is not None and report.comparisons:
        summary = report.summary()
        total = summary["Total Comparisons"]
        if total:
            conflict = summary["Contradiction"] + summary["Partial Contradiction"]
            axes.append("Contradiction Rate")
            values.append(100 * conflict / total)

    root_causes = st.session_state.get("contradiction_root_causes")
    if report is not None and root_causes:
        flagged_confidences = [
            rc.confidence for c, rc in zip(report.comparisons, root_causes)
            if c.classification in ("Contradiction", "Partial Contradiction") and rc.confidence is not None
        ]
        if flagged_confidences:
            axes.append("Root Cause Uncertainty")
            values.append(100 - sum(flagged_confidences) / len(flagged_confidences))

    if not axes:
        st.info("No risk signal data available yet.")
        return

    fig = ui_theme.plotly_radar(
        axes, values, title="Risk Indicators (higher = more risk)", color=ui_theme.CATEGORY_PALETTE[4],
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Computed directly from retrieval/analysis coverage (field coverage gaps, evidence-source "
        "reliability, contradiction rate, root-cause confidence) - not a machine-learned failure "
        "prediction."
    )

    metric_cols = st.columns(len(axes))
    for col, (axis, value) in zip(metric_cols, zip(axes, values)):
        col.metric(axis, f"{value:.0f}")


def _proposal_markdown(
    final_recommendation: List[dict], problem_solution_summary: List[dict],
    research_gap_analysis: List[dict], knowledge_table: List[dict], overall_analysis: List[dict],
    improvement_table: List[dict],
) -> str:
    """Pure formatting over data every other tab already computed - no new
    extraction, no new LLM call."""
    by_category = {r["Category"]: r["Value"] for r in final_recommendation}
    title = by_category.get("Suggested Research Title") or "Untitled Proposal"
    technique = by_category.get("Recommended AI Technique", NOT_FOUND_IN_CONTEXT)
    model = by_category.get("Recommended AI Model", NOT_FOUND_IN_CONTEXT)
    dataset = by_category.get("Recommended Dataset", NOT_FOUND_IN_CONTEXT)
    framework = by_category.get("Recommended Framework", NOT_FOUND_IN_CONTEXT)
    metrics = by_category.get("Recommended Evaluation Metrics", NOT_FOUND_IN_CONTEXT)
    setup = by_category.get("Recommended Experimental Setup", NOT_FOUND_IN_CONTEXT)
    advantages = by_category.get("Expected Advantages", NOT_FOUND_IN_CONTEXT)
    risks = by_category.get("Possible Risks", NOT_FOUND_IN_CONTEXT)

    summary_by_category = {r["Category"]: r["Value"] for r in problem_solution_summary}
    most_common_problem = summary_by_category.get("Most Common Research Problem", NOT_FOUND_IN_CONTEXT)

    gap_labels = [g["Research Gap"] for g in research_gap_analysis[:3]]
    accuracy_entry = next((r for r in overall_analysis if r["Category"] == "Highest Reported Accuracy"), None)
    best_accuracy = accuracy_entry.get("Most Common", NOT_FOUND_IN_CONTEXT) if accuracy_entry else NOT_FOUND_IN_CONTEXT

    related_work_lines = [
        f"| {row.get('Paper Title', '')} | {row.get('Year', '')} | {row.get('AI Technique', '')} | "
        f"{row.get('Key Contributions', '')} |"
        for row in knowledge_table
    ]
    related_work_table = (
        "| Paper | Year | AI Technique | Key Contribution |\n|---|---|---|---|\n" + "\n".join(related_work_lines)
        if related_work_lines else "_No papers analyzed._"
    )

    limitation_lines = [f"- **{row['Problem']}:** {row['Suggested Improvement']}" for row in improvement_table[:5]]
    limitations_section = "\n".join(limitation_lines) if limitation_lines else "_None identified._"

    gap_sentence = (
        " Key research gaps identified across the literature: " + "; ".join(gap_labels) + "."
        if gap_labels else ""
    )
    abstract = (
        f"This work proposes {technique} using {model}, evaluated on {dataset}"
        + (f" with {framework}" if framework != NOT_FOUND_IN_CONTEXT else "")
        + f", addressing {most_common_problem}. {advantages}"
    )
    conclusion_problem = most_common_problem.lower() if most_common_problem != NOT_FOUND_IN_CONTEXT else "the identified research problem"

    return f"""# {title}

## Abstract
{abstract}

## I. Introduction
The reviewed literature most commonly addresses **{most_common_problem}**.{gap_sentence}

## II. Related Work
{related_work_table}

## III. Proposed Methodology
- **AI Technique:** {technique}
- **AI Model:** {model}
- **Dataset:** {dataset}
- **Framework:** {framework}
- **Evaluation Metrics:** {metrics}
- **Experimental Setup:** {setup}

## IV. Expected Results
{advantages}

Best reported result in the analyzed literature: **{best_accuracy}**.

## V. Risks & Limitations
{risks}

{limitations_section}

## VI. Conclusion
This proposal builds on {conclusion_problem} by combining {technique} with {model}, targeting the \
gaps and limitations identified across the reviewed literature.
"""


def render_proposal_preview_tab() -> None:
    ui_theme.section_header(
        "📝", "Proposal Preview",
        "An IEEE-style formatted document assembled from the Inventor Agent's Final Recommendation "
        "and the analysis above - pure formatting, no new extraction.",
    )
    recommendation = st.session_state.get("inventor_recommendation")
    if recommendation is None or recommendation.insufficient_evidence or not recommendation.final_recommendation:
        st.info(
            "Run Ideas & Recommendations first - the Proposal Preview is assembled from its "
            "Final Recommendation."
        )
        return

    knowledge = st.session_state.get("knowledge_analysis")
    problem_solution = st.session_state.get("problem_solution_analysis")

    markdown = _proposal_markdown(
        recommendation.final_recommendation,
        problem_solution.summary if problem_solution else [],
        knowledge.research_gap_analysis if knowledge else [],
        knowledge.knowledge_table if knowledge else [],
        knowledge.overall_analysis if knowledge else [],
        recommendation.improvement_table,
    )

    with st.container(border=True):
        st.markdown(markdown)

    st.download_button(
        "Download as Markdown", data=markdown, file_name="research_proposal.md", mime="text/markdown",
    )


if __name__ == "__main__":
    main()
