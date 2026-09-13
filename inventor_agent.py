"""
Inventor Agent.

Runs on demand (its own button in app.py), positioned after every other
analysis section - Knowledge Retrieval, Overall Analysis, Research Gap
Analysis, Future Work Analysis, Problem-Solution Analysis, and (if the
user chose to run it) Contradiction Detection / Root Cause Analysis -
since it synthesizes across all of their outputs rather than reading
papers itself.

It performs NO new PDF fetching: everything it reasons over is either the
already-extracted, already-aggregated structured data the other modules
produced (condensed into JSON), a small amount of supplementary evidence
retrieved from each paper's ALREADY-BUILT FAISS index (rag_pipeline.py) via
config.RAG_QUERY_INVENTOR, or deterministic Python counts of how many
papers are missing each technical field (field_coverage - computed here,
never guessed by the LLM).

Behaves like an experienced AI Research Mentor, not a report generator:
missing/"Not Found in Retrieved Context" data is never a stopping
condition. When a field is missing across most papers, that absence IS the
finding - the agent must name the gap and recommend a concrete, actionable
fix (e.g. "no papers report a training optimizer -> recommend AdamW for
stable convergence"), never respond with a bare "not enough information".
Every recommendation is labeled either "Supported by Retrieved Literature"
(grounded in specific papers, cited by real title) or "General AI Best
Practice" (a well-known field-standard fix for an identified gap, not
itself present in the retrieved papers) - this labeling is enforced at the
code level (_validate_recommendations / _validate_improvement_table), not
just requested in the prompt, so a hallucinated paper citation can never
reach the UI.

insufficient_evidence is now a true floor case only: no papers at all, no
LLM configured, or (extremely unlikely) the LLM returns nothing usable AND
every field is already fully reported across every paper, so there is
neither a literature-grounded recommendation nor a coverage gap to build a
best-practice one from.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import config
import llm
from rag_pipeline import NOT_FOUND_IN_CONTEXT, PaperIndex, retrieve_chunks
from utils import Paper

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE = "Insufficient evidence to recommend an improved solution."
GENERAL_BEST_PRACTICE = "General AI Best Practice"
SUPPORTED_BY_LITERATURE = "Supported by Retrieved Literature"

_METRIC_COLUMNS = ["Accuracy", "F1 Score", "Precision", "Recall", "AUC", "mAP"]
_FLAGGED_CLASSIFICATIONS = ("Contradiction", "Partial Contradiction")

_FINAL_RECOMMENDATION_FIELDS = [
    ("recommended_ai_technique", "Recommended AI Technique"),
    ("recommended_ai_model", "Recommended AI Model"),
    ("recommended_dataset", "Recommended Dataset"),
    ("recommended_framework", "Recommended Framework"),
    ("recommended_evaluation_metrics", "Recommended Evaluation Metrics"),
    ("recommended_experimental_setup", "Recommended Experimental Setup"),
    ("suggested_research_title", "Suggested Research Title"),
    ("expected_advantages", "Expected Advantages"),
    ("possible_risks", "Possible Risks"),
]
# Plain example JSON fragment (e.g. "recommended_ai_technique": "...", ...) spliced
# into the prompt below, so the schema shown to the LLM always matches
# _FINAL_RECOMMENDATION_FIELDS without hand-duplicating the key list.
_FINAL_RECOMMENDATION_JSON_SKELETON = ", ".join(f'"{key}": "..."' for key, _ in _FINAL_RECOMMENDATION_FIELDS)

# Fields checked for coverage across the paper set - deliberately the ones
# a research mentor would flag as a methodological weakness when absent
# (as opposed to e.g. "LLM"/"Agent Framework", which are legitimately
# "Not Reported" for most non-LLM papers and aren't a weakness per se).
_COVERAGE_FIELDS = [
    "Dataset", "Optimizer", "Loss Function", "Hyperparameters", "Hardware", "Evaluation Metrics",
]
_METRIC_FAMILY_LABEL = "Any Evaluation Metric (Accuracy/Precision/Recall/F1/AUC/mAP)"

# Deterministic, code-level fallback so the Improvement Table is NEVER
# empty when there is real missing-data signal, even if the LLM call fails
# or returns something unusable - "identify what is missing and recommend
# best practices" is guaranteed structurally, not just by prompt instruction.
# field -> (Problem, Suggested Improvement, Expected Benefit, Reason)
_BEST_PRACTICE_FALLBACKS: Dict[str, Tuple[str, str, str, str]] = {
    "Dataset": (
        "No Public Benchmark Dataset Reported",
        "Evaluate the proposed method on a standard, publicly available benchmark dataset for this domain.",
        "Improved reproducibility and fair comparison with existing methods.",
        "Public datasets let other researchers reproduce results and compare methods on equal footing.",
    ),
    "Optimizer": (
        "Training Optimizer Not Reported",
        "Use AdamW as the training optimizer.",
        "More stable convergence and better generalization.",
        "AdamW is widely adopted in modern deep learning for stable convergence.",
    ),
    "Loss Function": (
        "Loss Function Not Reported",
        "Report and justify the loss function used for training.",
        "Improved transparency and reproducibility of the training procedure.",
        "The loss function materially affects convergence and results, and should be disclosed.",
    ),
    "Hyperparameters": (
        "Hyperparameter Selection Not Reported",
        "Use Bayesian Optimization or Optuna for hyperparameter search instead of manual tuning.",
        "Improved model performance and reproducibility.",
        "Manual hyperparameter selection is difficult to reproduce and often suboptimal.",
    ),
    "Hardware": (
        "Hardware/Compute Environment Not Reported",
        "Report the hardware and compute budget used (GPU/TPU type, memory, training time).",
        "Enables fair comparison of computational cost across methods.",
        "Compute cost is often as important as accuracy when judging practical applicability.",
    ),
    "Evaluation Metrics": (
        "Evaluation Metrics Not Reported",
        "Report standard evaluation metrics explicitly (Accuracy, Precision, Recall, F1-score).",
        "A more complete, comparable picture of model performance.",
        "Unreported evaluation criteria make it impossible to compare against other work.",
    ),
    _METRIC_FAMILY_LABEL: (
        "Weak or Missing Performance Metrics",
        "Evaluate the model using Accuracy, Precision, Recall, F1-score and ROC-AUC rather than a single metric.",
        "A more comprehensive, comparable evaluation that reveals failure modes a single metric hides.",
        "Recent research consistently reports multiple complementary metrics rather than accuracy alone.",
    ),
}


@dataclass
class InventorRecommendation:
    recommendations: List[dict] = field(default_factory=list)          # existing improvement-vs-strength table
    improvement_table: List[dict] = field(default_factory=list)        # Problem/Observation/Fix/Benefit/Reason/Papers
    final_recommendation: List[dict] = field(default_factory=list)     # Category/Value rows
    insufficient_evidence: bool = False
    insufficient_evidence_reason: str = ""


# --------------------------------------------------------------------------
# Context condensation - pure Python, zero LLM/fetch cost. Keeps the
# single synthesis call's input bounded regardless of how many papers/
# gaps/contradictions exist.
# --------------------------------------------------------------------------

def _best_metric(row: dict) -> str:
    for metric_col in _METRIC_COLUMNS:
        value = row.get(metric_col, NOT_FOUND_IN_CONTEXT)
        if value and value != NOT_FOUND_IN_CONTEXT:
            return f"{value} ({metric_col})"
    return NOT_FOUND_IN_CONTEXT


def _condense_papers(knowledge_table: List[dict]) -> List[dict]:
    return [
        {
            "title": row.get("Paper Title", NOT_FOUND_IN_CONTEXT),
            "year": row.get("Year", NOT_FOUND_IN_CONTEXT),
            "ai_technique": row.get("AI Technique", NOT_FOUND_IN_CONTEXT),
            "ai_model": row.get("AI Model", NOT_FOUND_IN_CONTEXT),
            "dataset": row.get("Dataset", NOT_FOUND_IN_CONTEXT),
            "framework": row.get("Framework", NOT_FOUND_IN_CONTEXT),
            "best_reported_metric": _best_metric(row),
            "key_contribution": row.get("Key Contributions", NOT_FOUND_IN_CONTEXT),
            "limitations": row.get("Limitations", NOT_FOUND_IN_CONTEXT),
        }
        for row in knowledge_table
    ]


def _condense_contradictions(contradiction_table: List[dict]) -> List[dict]:
    flagged = [row for row in contradiction_table if row.get("Classification") in _FLAGGED_CLASSIFICATIONS]
    condensed = []
    for row in flagged[: config.INVENTOR_MAX_CONTRADICTIONS]:
        condensed.append({
            "paper_a": row.get("Paper A"), "paper_b": row.get("Paper B"),
            "topic": row.get("Compared Topic"), "classification": row.get("Classification"),
            "root_cause": row.get("Root Cause"), "root_cause_explanation": row.get("Root Cause Explanation"),
        })
    return condensed


def _compute_field_coverage(knowledge_table: List[dict]) -> List[dict]:
    """Deterministic count of how many papers are missing each technical
    field - the grounding for "Most papers do not report X" style
    observations, so the LLM never has to guess (or hallucinate) a count.
    """
    total = len(knowledge_table)
    if not total:
        return []

    coverage = []
    for field_name in _COVERAGE_FIELDS:
        missing = sum(
            1 for row in knowledge_table
            if not str(row.get(field_name, NOT_FOUND_IN_CONTEXT)).strip()
            or str(row.get(field_name, NOT_FOUND_IN_CONTEXT)).strip() == NOT_FOUND_IN_CONTEXT
        )
        coverage.append({
            "field": field_name, "reported_count": total - missing,
            "missing_count": missing, "total_papers": total,
        })

    any_metric_missing = sum(
        1 for row in knowledge_table
        if all(str(row.get(f, NOT_FOUND_IN_CONTEXT)).strip() == NOT_FOUND_IN_CONTEXT for f in _METRIC_COLUMNS)
    )
    coverage.append({
        "field": _METRIC_FAMILY_LABEL, "reported_count": total - any_metric_missing,
        "missing_count": any_metric_missing, "total_papers": total,
    })
    return coverage


_SUPPLEMENTARY_EVIDENCE_CHARS = 250  # truncate each chunk - this is explicitly "supplementary
                                      # only" context (see docstring below), not primary
                                      # evidence, so it doesn't need the full ~480-token chunk.
                                      # CONFIRMED via live run: 5 papers x full untruncated
                                      # chunks was a real contributor to a 413 "Request too
                                      # large" from Groq (see config.INVENTOR_MAX_TOKENS).


def _retrieve_supplementary_evidence(papers: List[Paper], rag_cache: Dict[str, PaperIndex]) -> List[dict]:
    """Top RAG_TOP_K_INVENTOR_PER_PAPER chunk(s) per paper for
    config.RAG_QUERY_INVENTOR, from each paper's already-built FAISS
    index. Supplementary only - the structured summaries above carry the
    primary evidentiary weight."""
    evidence = []
    for paper in papers:
        paper_index = rag_cache.get(paper.title, PaperIndex(paper_id=paper.title))
        for rc in retrieve_chunks(paper_index, config.RAG_QUERY_INVENTOR, config.RAG_TOP_K_INVENTOR_PER_PAPER):
            text = rc.chunk.chunk_text
            if len(text) > _SUPPLEMENTARY_EVIDENCE_CHARS:
                text = text[:_SUPPLEMENTARY_EVIDENCE_CHARS] + "..."
            evidence.append({
                "paper": paper.title,
                "section": rc.chunk.section_name,
                "text": text,
            })
    return evidence


def _build_context(
    knowledge_table: List[dict], overall_analysis: List[dict],
    research_gap_analysis: List[dict], future_work_analysis: List[dict],
    problem_solution_table: List[dict], problem_solution_summary: List[dict],
    contradiction_table: List[dict], supplementary_evidence: List[dict],
    field_coverage: List[dict],
) -> dict:
    return {
        "papers": _condense_papers(knowledge_table),
        "overall_analysis": overall_analysis,
        "field_coverage": field_coverage,
        "research_gap_analysis": research_gap_analysis[: config.INVENTOR_MAX_GAPS_FUTURE_WORK],
        "future_work_analysis": future_work_analysis[: config.INVENTOR_MAX_GAPS_FUTURE_WORK],
        # Capped like research_gap_analysis/future_work_analysis above (both
        # already sorted most-frequent-first by problem_solution_analysis.py,
        # so this keeps the most-supported problems, not an arbitrary slice).
        # CONFIRMED via live run against 5 real papers: passing the table
        # uncapped pushed one real Inventor Agent call to 11,932 tokens on a
        # request, over this tier's 8,000 hard per-request cap - Groq
        # rejected it outright (413 "Request too large"), which is NOT
        # fixable by retrying/waiting the way a 429 is, since the request
        # itself is oversized regardless of remaining budget.
        "problem_solution_table": problem_solution_table[: config.INVENTOR_MAX_PROBLEM_SOLUTION_ROWS],
        "problem_solution_summary": problem_solution_summary,
        "contradictions": _condense_contradictions(contradiction_table),
        "supplementary_retrieved_evidence": supplementary_evidence,
    }


# --------------------------------------------------------------------------
# LLM synthesis (one call)
# --------------------------------------------------------------------------

INVENTOR_SYSTEM_PROMPT = f"""You are an experienced AI Research Mentor advising a new researcher on what to \
build next. You are given structured findings ALREADY EXTRACTED from a set of retrieved research papers (as \
JSON) and must recommend IMPROVED research directions grounded in the strengths and weaknesses actually \
identified in that literature.

CORE RULE - never stop the analysis because information is missing:
When a field is missing (reported as "{NOT_FOUND_IN_CONTEXT}") for most or all papers, that absence IS a \
finding, not a dead end. Name the gap as a Problem, then give a concrete, actionable Suggested Improvement -
a specific technique/tool/metric/dataset a researcher could actually go implement - never a vague restatement \
like "information not available" or "not enough information". For example:
- Instead of "Dataset not mentioned" -> "Most papers do not use a publicly available benchmark dataset. \
Consider evaluating the proposed method on standard benchmark datasets to improve reproducibility and fair \
comparison."
- Instead of "Accuracy not reported" -> "Most papers do not report accuracy. Future work should evaluate the \
model using Accuracy, Precision, Recall, and F1-score."
- Instead of "Optimizer not mentioned" -> "Training optimizer is not reported. AdamW is widely adopted in \
modern deep learning and can be considered for stable convergence."

You are given:
- papers: condensed per-paper facts (technique, model, dataset, framework, best reported metric,
  key contribution, limitations).
- overall_analysis: most-used AI technique/model/dataset/framework/metric and highest reported
  accuracy across the papers.
- field_coverage: EXACT counts (computed deterministically, not by you) of how many of the given papers are
  missing each of Dataset/Optimizer/Loss Function/Hyperparameters/Hardware/Evaluation Metrics and any
  evaluation metric at all. Use these exact numbers whenever you state "N of M papers do not report X" -
  never invent or estimate a count yourself.
- research_gap_analysis: semantically clustered research gaps with frequency and a recommendation
  for each.
- future_work_analysis: semantically clustered future-work themes with frequency (already merged -
  do not re-merge, just use it).
- problem_solution_table / problem_solution_summary: common research problems mapped to the AI
  techniques/models used to address them, and which problems are over- vs under-solved.
- contradictions: pairs of papers whose claims directly conflicted, with the analyzed root cause of
  the conflict when available. This may be an empty list if no contradiction analysis was run -
  in that case simply skip step 4 below, do not fabricate a contradiction.
- supplementary_retrieved_evidence: a small number of raw retrieved text chunks (paper, section,
  text) semantically retrieved for "best results, highest accuracy, datasets, models, future work,
  research gaps" - use these as extra grounding/color where relevant, in addition to (not instead
  of) the structured fields above.

Work through these steps:
1. Identify the most common research problems, and the most successful AI techniques, AI models,
   datasets, frameworks and evaluation metrics - "successful" means both frequently used AND
   associated with strong reported results in the given data, not just popular.
2. Identify weaknesses across the literature using field_coverage, limitations, and research gaps (e.g. low
   accuracy, no real-world deployment, small dataset, poor scalability, high computational cost, lack of
   explainability, no hyperparameter optimization, poor generalization, no baseline comparison, weak
   evaluation - single metric only, static decision making) - ground each in specific papers' limitations,
   the research gaps given, or a field_coverage count.
3. Use future_work_analysis (already merged) to see what the field itself says should happen next.
4. If contradictions is non-empty: for each one, decide which side is better supported (e.g.
   broader/more rigorous reported evaluation, stronger evidence source, consistency with the wider
   literature's overall_analysis) and state that explicitly. Skip this step entirely if
   contradictions is empty.
5. Build "recommendations": 3-6 rows that each combine a strength identified in step 1 with a fix for
   a weakness identified in step 2 - answering "what should a new researcher build?" Never copy a
   single existing paper's approach unchanged; each recommendation must be an improvement, not a
   restatement.
6. Build "improvement_table": one row for EVERY weakness/gap you identified in step 2 (typically 4-10 rows) -
   this is the primary deliverable. Each row must have a specific, actionable Suggested Improvement (a named
   technique/tool/metric, e.g. "AdamW", "SHAP/LIME/Attention Visualization", "Bayesian Optimization or
   Optuna", "Vision Transformer", "model pruning/quantization/knowledge distillation" - not a vague
   instruction to "improve" something).
7. Produce ONE final consolidated recommendation: technique, model, dataset, framework, evaluation
   metrics, experimental setup, a suggested research title, expected advantages, possible risks.

Evidence labeling - EVERY row in "recommendations" and "improvement_table" must set "evidence_type" to
exactly one of:
- "{SUPPORTED_BY_LITERATURE}" - the observation AND the specific fix both come from the given papers
  (e.g. a future-work theme the papers themselves state, or a technique another paper in the set already
  used successfully). supporting_papers must then list the real paper titles that support it.
- "{GENERAL_BEST_PRACTICE}" - the gap is observed in the given data, but the specific fix you're
  recommending (e.g. AdamW, SHAP, Optuna, ViT) is a well-known field-standard technique NOT itself present
  in the retrieved papers. supporting_papers must then be an empty list - do not invent a citation for a
  suggestion the papers never made.

Rules - do not hallucinate:
- Every "supporting_papers" entry must be a real paper title from the "papers" list given - never invent
  one or cite one that isn't present in "papers". If you cannot name a real supporting paper, use
  evidence_type "{GENERAL_BEST_PRACTICE}" and leave supporting_papers empty instead.
- current_observation / "N of M papers..." claims must be grounded in field_coverage, limitations, or
  research_gap_analysis given to you - never a fabricated statistic.
- Recommending a technique/model absent from every field in the given data is allowed (that's exactly what
  "{GENERAL_BEST_PRACTICE}" is for) as long as it's a genuine, well-established fix for a gap the data
  actually shows - not a random idea unconnected to any observed weakness.
- Both "recommendations" and "improvement_table" must be non-empty as long as "papers" is non-empty - there
  is always at least one gap (even "no explicit failure analysis" or "single-dataset evaluation") worth
  naming. Do not return an empty list and do not respond with anything resembling "insufficient
  information" - that is exactly the failure mode you must avoid.

Respond with ONLY a JSON object of this exact shape:
{{"recommendations": [{{"current_approach": "...", "suggested_improvement": "...",
"why_better": "...", "expected_benefit": "...", "evidence_type": "{GENERAL_BEST_PRACTICE}" | \
"{SUPPORTED_BY_LITERATURE}", "supporting_papers": ["...", "..."]}}],
"improvement_table": [{{"problem": "...", "current_observation": "...", "suggested_improvement": "...",
"expected_benefit": "...", "reason": "...", "evidence_type": "{GENERAL_BEST_PRACTICE}" | \
"{SUPPORTED_BY_LITERATURE}", "supporting_papers": ["...", "..."]}}],
"final_recommendation": {{{_FINAL_RECOMMENDATION_JSON_SKELETON}}}}}
No markdown fences, no commentary."""


def _log_context_size_breakdown(context: dict) -> None:
    """Diagnostic only - logs each context section's approximate token
    share so a future 413 ("Request too large") can be traced to the
    specific section that grew too big, instead of needing another live
    probe to find out (see config.INVENTOR_MAX_TOKENS for the incident
    this was added after)."""
    if not logger.isEnabledFor(logging.INFO):
        return
    sizes = {key: len(json.dumps(value)) // 4 for key, value in context.items()}
    logger.info("Inventor Agent context size (approx tokens by section): %s | total=%d", sizes, sum(sizes.values()))


def _call_inventor_llm(context: dict) -> dict:
    _log_context_size_breakdown(context)
    content = llm._call_groq(
        messages=[
            {"role": "system", "content": INVENTOR_SYSTEM_PROMPT},
            # Compact (no indent) - this JSON is read by the LLM, not a human;
            # indent=2's extra whitespace was pure wasted tokens on a
            # per-request budget already tight enough to 413 (see
            # config.INVENTOR_MAX_TOKENS for the incident this was found in).
            {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
        ],
        max_tokens=config.INVENTOR_MAX_TOKENS,
        temperature=0.2,
        json_mode=True,
        agent_name="InventorAgent",
        paper_id=f"papers[{len(context.get('papers', []))}]",
        cache_version=config.LLM_CACHE_VERSION,
    )
    data = llm.parse_json(content) if content else None
    if not isinstance(data, dict):
        if content:
            logger.warning("Inventor Agent returned unusable JSON: %s", content[:200])
        return {}
    return data


# --------------------------------------------------------------------------
# Response validation - defends against hallucinated paper citations even
# though the prompt already forbids them. A row's evidence_type is only
# ever trusted as "Supported by Retrieved Literature" if at least one
# cited title actually matches a retrieved paper; otherwise it is force-
# downgraded to "General AI Best Practice" rather than shown as
# unverifiable.
# --------------------------------------------------------------------------

def _resolve_evidence(entry: dict, known_lower: Dict[str, str]) -> str:
    """Returns the string to show in the Supporting Papers cell: verified
    real paper titles, or the literal "General AI Best Practice" label."""
    supporting = entry.get("supporting_papers")
    supporting = supporting if isinstance(supporting, list) else []
    verified_titles = [known_lower[str(t).strip().lower()] for t in supporting if str(t).strip().lower() in known_lower]
    if verified_titles:
        return ", ".join(verified_titles)
    return GENERAL_BEST_PRACTICE


def _validate_recommendations(raw_recommendations, known_titles: set) -> List[dict]:
    if not isinstance(raw_recommendations, list):
        return []

    known_lower = {t.lower(): t for t in known_titles}
    rows = []
    for entry in raw_recommendations:
        if not isinstance(entry, dict):
            continue
        rows.append({
            "Current Approach": str(entry.get("current_approach") or "Not Reported").strip(),
            "Suggested Improvement": str(entry.get("suggested_improvement") or "Not Reported").strip(),
            "Why this is Better": str(entry.get("why_better") or "Not Reported").strip(),
            "Expected Benefit": str(entry.get("expected_benefit") or "Not Reported").strip(),
            "Supporting Papers": _resolve_evidence(entry, known_lower),
        })
    return rows


def _validate_improvement_table(raw_rows, known_titles: set) -> List[dict]:
    if not isinstance(raw_rows, list):
        return []

    known_lower = {t.lower(): t for t in known_titles}
    rows = []
    for entry in raw_rows:
        if not isinstance(entry, dict):
            continue
        rows.append({
            "Problem": str(entry.get("problem") or "Not Reported").strip(),
            "Current Observation": str(entry.get("current_observation") or "Not Reported").strip(),
            "Suggested Improvement": str(entry.get("suggested_improvement") or "Not Reported").strip(),
            "Expected Benefit": str(entry.get("expected_benefit") or "Not Reported").strip(),
            "Reason": str(entry.get("reason") or "Not Reported").strip(),
            "Supporting Papers": _resolve_evidence(entry, known_lower),
        })
    return rows


def _fallback_improvement_table(field_coverage: List[dict]) -> List[dict]:
    """Deterministic, code-level floor: whenever field_coverage shows a
    field missing across at least half the papers, emit a best-practice
    row for it even if the LLM call failed or returned nothing usable for
    improvement_table. Guarantees "identify what is missing and recommend
    best practices" holds structurally, not just via prompt compliance."""
    rows = []
    for entry in field_coverage:
        total = entry.get("total_papers", 0)
        missing = entry.get("missing_count", 0)
        if not total or missing / total < 0.5:
            continue
        fallback = _BEST_PRACTICE_FALLBACKS.get(entry.get("field"))
        if not fallback:
            continue
        problem, suggestion, benefit, reason = fallback
        rows.append({
            "Problem": problem,
            "Current Observation": f"{missing} of {total} papers do not report this ({entry.get('field')}).",
            "Suggested Improvement": suggestion,
            "Expected Benefit": benefit,
            "Reason": reason,
            "Supporting Papers": GENERAL_BEST_PRACTICE,
        })
    return rows


def _build_final_recommendation(raw_final: dict) -> List[dict]:
    if not isinstance(raw_final, dict):
        raw_final = {}
    return [
        {"Category": label, "Value": str(raw_final.get(key) or "Not Reported").strip()}
        for key, label in _FINAL_RECOMMENDATION_FIELDS
    ]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def generate_recommendation(
    papers: List[Paper],
    knowledge_table: List[dict],
    overall_analysis: List[dict],
    research_gap_analysis: List[dict],
    future_work_analysis: List[dict],
    problem_solution_table: List[dict],
    problem_solution_summary: List[dict],
    contradiction_table: List[dict],
    rag_cache: Dict[str, PaperIndex],
) -> InventorRecommendation:
    if not knowledge_table:
        return InventorRecommendation(
            insufficient_evidence=True,
            insufficient_evidence_reason="No Knowledge Retrieval data is available yet - run a search first.",
        )

    if not llm.is_available():
        return InventorRecommendation(
            insufficient_evidence=True,
            insufficient_evidence_reason="GROQ_API_KEY not configured - the Inventor Agent requires the LLM.",
        )
    if llm.is_daily_quota_exhausted():
        return InventorRecommendation(
            insufficient_evidence=True,
            insufficient_evidence_reason="Daily Groq token quota exhausted - the Inventor Agent requires the LLM.",
        )

    supplementary_evidence = _retrieve_supplementary_evidence(papers, rag_cache)
    field_coverage = _compute_field_coverage(knowledge_table)
    context = _build_context(
        knowledge_table, overall_analysis, research_gap_analysis, future_work_analysis,
        problem_solution_table, problem_solution_summary, contradiction_table,
        supplementary_evidence, field_coverage,
    )
    try:
        data = _call_inventor_llm(context)
    except llm.GroqDailyQuotaExceeded as exc:
        logger.warning("Inventor Agent: daily Groq quota exhausted - %s", exc)
        # Degrade exactly like an unparsable/empty LLM response: the
        # deterministic field_coverage fallback below still produces a
        # real (non-empty) improvement_table from already-computed data,
        # preserving a useful result instead of an all-or-nothing failure.
        data = {}

    known_titles = {p.title for p in papers}
    recommendations = _validate_recommendations(data.get("recommendations"), known_titles)
    improvement_table = _validate_improvement_table(data.get("improvement_table"), known_titles)
    final_recommendation = _build_final_recommendation(data.get("final_recommendation"))

    if not improvement_table:
        improvement_table = _fallback_improvement_table(field_coverage)

    if not recommendations and not improvement_table:
        return InventorRecommendation(
            insufficient_evidence=True,
            insufficient_evidence_reason=INSUFFICIENT_EVIDENCE,
        )

    return InventorRecommendation(
        recommendations=recommendations,
        improvement_table=improvement_table,
        final_recommendation=final_recommendation,
    )
