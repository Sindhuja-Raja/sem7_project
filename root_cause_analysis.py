"""
Root Cause Analysis - an additive layer on top of Contradiction Detection.

Runs automatically, immediately after classify_pairs() identifies at least
one Contradiction / Partial Contradiction pair, as part of the same
"Detect Contradictions" click in app.py. It does NOT modify
contradiction_detection.py in any way - it only reads its output
(ClaimComparison rows) and the same Paper objects the user selected.

For every Contradiction / Partial Contradiction pair, this module:
    1. Retrieves evidence chunks (config.RAG_QUERY_ROOT_CAUSE: "methodology,
       experimental setup, results, discussion") for the two papers
       involved, from each paper's already-built FAISS index
       (rag_pipeline.py) - deduplicated so a paper referenced by multiple
       contradictions is only queried once. No PDF fetching happens here;
       that already happened once, in app.py's validation-gated collection loop.
    2. Sends the contradicting claims plus both papers' retrieved chunks to
       the LLM in a single batched call, asking it to compare Research
       Objective, Problem Statement, Application Domain, Dataset, Data
       Size, Experimental Setup, AI Technique, AI Model, Framework,
       Hyperparameters, Training Strategy, Optimizer, Loss Function,
       Evaluation Metrics, Hardware, Cloud/Edge Environment, Baseline
       Methods and Publication Year, and pick exactly one root cause
       category (or an honest "could not determine" / "insufficient
       evidence" sentinel) - never inventing a reason.

Output is a list parallel to the ClaimComparison list passed in, so app.py
can zip it onto the existing comparison_table() rows as three new columns
without touching contradiction_detection.py's data model.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config
import llm
from contradiction_detection import ClaimComparison
from rag_pipeline import PaperIndex, retrieve_context
from utils import Paper

logger = logging.getLogger(__name__)

ROOT_CAUSE_CATEGORIES = [
    "Different Dataset", "Different Experimental Setup", "Different AI Model",
    "Different Hyperparameters", "Different Evaluation Metrics", "Different Hardware",
    "Different Workload", "Different Application Domain", "Different Baseline Comparison",
    "Different Research Objective", "Different Training Strategy", "Different Data Distribution",
    "Different Publication Context", "Insufficient Experimental Details",
]
NOT_DETERMINED = "Root Cause Could Not Be Determined"
INSUFFICIENT_EVIDENCE = "Insufficient Evidence to Determine Root Cause"
NOT_APPLICABLE = "Not Applicable"

_FLAGGED_CLASSIFICATIONS = ("Contradiction", "Partial Contradiction")


@dataclass
class RootCauseResult:
    root_cause_category: str = NOT_APPLICABLE
    root_cause_explanation: str = ""
    evidence_a: str = ""
    evidence_b: str = ""
    confidence: Optional[int] = None  # None = not applicable (no numeric confidence to show)


# --------------------------------------------------------------------------
# Evidence gathering (dedupe by paper label - a paper referenced by
# several contradictions is only retrieved once)
# --------------------------------------------------------------------------

def _retrieve_evidence_for_labels(
    labels: List[str], label_to_paper: Dict[str, Paper], rag_cache: Dict[str, PaperIndex],
) -> Dict[str, Tuple[str, str]]:
    """label -> (retrieved context, evidence source label). The source
    label ("RAG (N chunks from Full PDF)" vs "...Abstract fallback") is
    kept so the prompt can still tell the LLM to lower its confidence
    when a paper's evidence only ever came from its abstract."""
    evidence: Dict[str, Tuple[str, str]] = {}
    for label in labels:
        paper = label_to_paper[label]
        paper_index = rag_cache.get(paper.title, PaperIndex(paper_id=paper.title))
        context, _ = retrieve_context(
            paper_index,
            queries=[(config.RAG_QUERY_ROOT_CAUSE, config.RAG_TOP_K_ROOT_CAUSE)],
        )
        evidence[label] = (context, paper_index.source_label)
    return evidence


# --------------------------------------------------------------------------
# LLM comparison
# --------------------------------------------------------------------------

ROOT_CAUSE_SYSTEM_PROMPT = f"""You are a research analyst investigating WHY two academic papers \
reached contradictory conclusions on the same topic. You are given each paper's RETRIEVED CHUNKS \
(selected by semantic search as the passages most relevant to methodology/experimental setup/ \
results/discussion, or a single abstract chunk if the full PDF wasn't available - never the whole \
paper) and the specific claims that were found to contradict each other.

For EACH contradiction, compare these aspects between the two papers wherever the given retrieved \
chunks allow: Research Objective, Problem Statement, Application Domain, Dataset, Data Size, \
Experimental Setup, AI Technique, AI Model, Framework, Hyperparameters, Training Strategy, \
Optimizer, Loss Function, Evaluation Metrics, Hardware, Cloud/Edge Environment, Baseline \
Methods, Publication Year.

Determine the SINGLE most probable root cause category, chosen from EXACTLY this list:
{ROOT_CAUSE_CATEGORIES}

Rules - do not guess, do not invent a reason:
- Only name a category if the given retrieved chunks actually support it (e.g. only answer
  "Different Dataset" if the two papers' retrieved chunks name different datasets).
- If the retrieved chunks cover several relevant aspects but none clearly explains the
  contradiction, respond with "{NOT_DETERMINED}" instead of forcing a category.
- If the retrieved chunks for either paper are too sparse/vague to compare meaningfully, respond
  with "{INSUFFICIENT_EVIDENCE}" instead (use this for both root_cause_category AND
  evidence_a/evidence_b when the relevant one is missing).
- confidence (0-100): your confidence in this determination. If either paper's evidence source
  says "Abstract fallback" rather than "Full PDF", your confidence must be substantially lower
  (usually under 40) since abstracts rarely contain enough methodological detail to be certain.
- evidence_a / evidence_b: quote or closely paraphrase the SPECIFIC retrieved chunk text you were
  given (never invent a section/page number or a detail not present in the given chunks) that
  supports your determination, for each paper respectively.

Respond with ONLY a JSON object: {{"root_causes": [{{"index": 1, "root_cause_category": "...", \
"explanation": "...", "evidence_a": "...", "evidence_b": "...", "confidence": <0-100 integer>}}]}}
One entry per contradiction given, "index" matching the 1-based position below. No markdown \
fences, no commentary."""


def _build_root_cause_message(
    comparisons: List[ClaimComparison], evidence: Dict[str, Tuple[str, str]],
) -> str:
    labels = sorted({c.paper_a for c in comparisons} | {c.paper_b for c in comparisons})
    evidence_blocks = [
        f"[{label}]\nEvidence source: {evidence.get(label, ('', 'Not Available'))[1]}\n"
        f"{evidence.get(label, ('(no text available)', ''))[0] or '(no text available)'}"
        for label in labels
    ]

    pair_blocks = [
        f"Contradiction {i}:\nPaper A: {c.paper_a}\nPaper B: {c.paper_b}\nTopic: {c.topic}\n"
        f"Claim A: {c.claim_a}\nClaim B: {c.claim_b}\nClassification: {c.classification}"
        for i, c in enumerate(comparisons, start=1)
    ]

    return (
        "=== Paper Evidence ===\n\n" + "\n\n".join(evidence_blocks)
        + "\n\n=== Contradictions to Analyze ===\n\n" + "\n\n".join(pair_blocks)
    )


def _call_root_cause_llm(
    comparisons: List[ClaimComparison], evidence: Dict[str, Tuple[str, str]],
) -> List[RootCauseResult]:
    content = llm._call_groq(
        messages=[
            {"role": "system", "content": ROOT_CAUSE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_root_cause_message(comparisons, evidence)},
        ],
        max_tokens=min(4000, config.ROOT_CAUSE_MAX_TOKENS_PER_PAIR * len(comparisons) + 200),
        temperature=0.1,
        json_mode=True,
        agent_name="RootCauseAnalysis",
        paper_id=f"pairs[{len(comparisons)}]",
        cache_version=config.LLM_CACHE_VERSION,
    )
    data = llm.parse_json(content) if content else None
    by_index: Dict[int, dict] = {}
    if isinstance(data, dict) and isinstance(data.get("root_causes"), list):
        for entry in data["root_causes"]:
            if not isinstance(entry, dict):
                continue
            try:
                by_index[int(entry.get("index"))] = entry
            except (TypeError, ValueError):
                continue
    elif content:
        logger.warning("Root cause analysis returned unusable JSON: %s", content[:200])

    results = []
    for i in range(1, len(comparisons) + 1):
        entry = by_index.get(i)
        if not entry:
            results.append(RootCauseResult(
                root_cause_category=INSUFFICIENT_EVIDENCE,
                root_cause_explanation="Root cause analysis unavailable for this pair.",
                evidence_a=INSUFFICIENT_EVIDENCE, evidence_b=INSUFFICIENT_EVIDENCE,
                confidence=0,
            ))
            continue

        category = str(entry.get("root_cause_category") or "").strip()
        if category not in ROOT_CAUSE_CATEGORIES and category not in (NOT_DETERMINED, INSUFFICIENT_EVIDENCE):
            category = NOT_DETERMINED

        try:
            confidence = int(entry.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0
        confidence = max(0, min(100, confidence))

        results.append(RootCauseResult(
            root_cause_category=category,
            root_cause_explanation=str(entry.get("explanation") or "").strip() or NOT_DETERMINED,
            evidence_a=str(entry.get("evidence_a") or "").strip() or INSUFFICIENT_EVIDENCE,
            evidence_b=str(entry.get("evidence_b") or "").strip() or INSUFFICIENT_EVIDENCE,
            confidence=confidence,
        ))
    return results


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def analyze_root_causes(
    comparisons: List[ClaimComparison], papers: List[Paper], paper_labels: List[str],
    rag_cache: Dict[str, PaperIndex],
) -> List[RootCauseResult]:
    """Parallel to `comparisons` (same length/order). Only entries whose
    classification is Contradiction / Partial Contradiction get a real
    analysis; every other entry gets a "Not Applicable" placeholder.
    Never modifies `comparisons` itself - contradiction_detection.py's
    output is read-only input here. `rag_cache` is the shared
    {paper title -> PaperIndex} built once in app.py."""
    results = [RootCauseResult() for _ in comparisons]

    flagged_idxs = [i for i, c in enumerate(comparisons) if c.classification in _FLAGGED_CLASSIFICATIONS]
    if not flagged_idxs:
        return results

    def _unavailable(reason: str) -> List[RootCauseResult]:
        for i in flagged_idxs:
            results[i] = RootCauseResult(
                root_cause_category=INSUFFICIENT_EVIDENCE,
                root_cause_explanation=reason,
                evidence_a=INSUFFICIENT_EVIDENCE, evidence_b=INSUFFICIENT_EVIDENCE,
                confidence=0,
            )
        return results

    if not llm.is_available():
        return _unavailable("GROQ_API_KEY not configured - root cause analysis unavailable.")
    if llm.is_daily_quota_exhausted():
        return _unavailable("Daily Groq token quota exhausted - root cause analysis unavailable.")

    label_to_paper = dict(zip(paper_labels, papers))
    flagged_comparisons = [comparisons[i] for i in flagged_idxs]
    needed_labels = sorted({c.paper_a for c in flagged_comparisons} | {c.paper_b for c in flagged_comparisons})
    evidence = _retrieve_evidence_for_labels(needed_labels, label_to_paper, rag_cache)

    try:
        raw_results = _call_root_cause_llm(flagged_comparisons, evidence)
    except llm.GroqDailyQuotaExceeded as exc:
        logger.warning("Root cause analysis: daily Groq quota exhausted - %s", exc)
        return _unavailable("Daily Groq token quota exhausted - root cause analysis unavailable.")
    for idx, result in zip(flagged_idxs, raw_results):
        results[idx] = result

    return results
