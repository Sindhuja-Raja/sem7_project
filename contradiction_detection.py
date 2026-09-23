"""
Agentic AI - Contradiction Detection module.

Runs on demand, after the user selects 2+ papers from the already-ranked
results and clicks "Detect Contradictions" - it is not part of the
automatic search pipeline.

Workflow:
    selected papers
        -> retrieve_claim_evidence() semantic retrieval (config.RAG_QUERY_
                                  CONTRADICTION: "claims, experimental
                                  results, evaluation, discussion,
                                  findings") against each paper's already-
                                  built FAISS index (rag_pipeline.py) -
                                  no PDF fetching happens here anymore,
                                  the shared per-paper index built once in
                                  app.py is reused
        -> extract_claims()      one batched LLM call extracts scientific
                                  claims per paper from ONLY the retrieved
                                  chunks (method, findings, performance,
                                  conclusions, advantages, limitations) -
                                  never authors/refs/etc.
        -> match_claim_pairs()   dense embedding similarity (reusing
                                  embedding.py's cached BGE model) keeps
                                  only claim pairs *from different papers*
                                  above a similarity threshold - a real
                                  filter, not a prompt instruction, so
                                  unrelated claims (e.g. latency vs energy)
                                  never reach the LLM
        -> classify_pairs()      one batched LLM call classifies each
                                  matched pair into exactly one of
                                  Agreement / Contradiction / Partial
                                  Contradiction / Different Context /
                                  Insufficient Evidence, with a reason and
                                  grounded supporting evidence
        -> ContradictionReport   comparison rows + full detail rows +
                                  summary counts, ready for st.dataframe
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

import config
import llm
import pipeline_status
from embedding import embed_texts
from rag_pipeline import PaperIndex, retrieve_context
from utils import Paper

logger = logging.getLogger(__name__)

CLASSIFICATIONS = [
    "Agreement", "Contradiction", "Partial Contradiction",
    "Different Context", "Insufficient Evidence",
]


@dataclass
class Claim:
    paper_label: str   # e.g. "Paper 2: Agentic AI for ..."
    topic: str
    claim_text: str
    claim_type: str


@dataclass
class ClaimComparison:
    paper_a: str
    paper_b: str
    topic: str
    claim_a: str
    claim_b: str
    classification: str
    confidence: int
    reason: str
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class ContradictionReport:
    comparisons: List[ClaimComparison] = field(default_factory=list)
    claims_extracted: int = 0
    predictions: List[dict] = field(default_factory=list)
    # See pipeline_status.py - lets app.py show WHY there are no
    # comparisons instead of a single generic "No comparable claim pairs
    # were found" message for every possible cause.
    status: str = pipeline_status.SUCCESS
    status_reason: str = ""

    def comparison_table(self) -> List[dict]:
        rows = [
            {
                "Paper A": c.paper_a,
                "Paper B": c.paper_b,
                "Compared Topic": c.topic,
                "Classification": c.classification,
                "Confidence": c.confidence,
                "Reason": c.reason,
            }
            for c in self.comparisons
        ]
        if self.predictions and len(self.predictions) == len(rows):
            for row, prediction in zip(rows, self.predictions):
                row["Predicted Contradiction"] = prediction.get("prediction", "Low Risk")
                row["Prediction Score"] = prediction.get("score", 0.0)
                row["Prediction Reason"] = prediction.get("reason", "")
        return rows

    def contradiction_details(self) -> List[dict]:
        """Full record (claim text + supporting evidence) for every
        Contradiction / Partial Contradiction pair, per the spec."""
        return [
            {
                "Paper A": c.paper_a,
                "Paper B": c.paper_b,
                "Compared Topic": c.topic,
                "Claim A": c.claim_a,
                "Claim B": c.claim_b,
                "Classification": c.classification,
                "Confidence": c.confidence,
                "Reason": c.reason,
                "Supporting Evidence": "; ".join(c.supporting_evidence) or "Not Mentioned",
            }
            for c in self.comparisons
            if c.classification in ("Contradiction", "Partial Contradiction")
        ]

    def summary(self) -> dict:
        counts = {label: 0 for label in CLASSIFICATIONS}
        for c in self.comparisons:
            if c.classification in counts:
                counts[c.classification] += 1
        counts["Total Comparisons"] = len(self.comparisons)
        return counts


# --------------------------------------------------------------------------
# Step 0: RAG retrieval of claim-relevant chunks per selected paper, from
# each paper's already-built FAISS index (no PDF fetching happens here -
# that already happened once, in app.py's validation-gated collection loop).
# --------------------------------------------------------------------------

def retrieve_claim_evidence(
    papers: List[Paper], rag_cache: Dict[str, PaperIndex],
) -> Dict[int, str]:
    """Returns {paper index (0-based) -> assembled retrieved-chunk context}
    via semantic retrieval against config.RAG_QUERY_CONTRADICTION."""
    contexts: Dict[int, str] = {}
    for i, paper in enumerate(papers):
        paper_index = rag_cache.get(paper.title, PaperIndex(paper_id=paper.title))
        context, _ = retrieve_context(
            paper_index,
            queries=[(config.RAG_QUERY_CONTRADICTION, config.RAG_TOP_K_CONTRADICTION)],
        )
        contexts[i] = context
    return contexts


# --------------------------------------------------------------------------
# Step 1: claim extraction (one batched Groq call for all selected papers)
# --------------------------------------------------------------------------

CLAIM_EXTRACTION_SYSTEM_PROMPT = f"""You extract scientific claims from RETRIEVED CHUNKS of \
academic papers for a contradiction-detection system. For EACH paper you are given its retrieved \
chunks (selected by semantic search as the passages most relevant to claims/results/evaluation/ \
discussion) - not the whole paper. Extract up to {config.CONTRADICTION_MAX_CLAIMS_PER_PAPER} \
concise, checkable scientific claims per paper - never paper metadata.

Extract ONLY claims that are one of:
- Proposed method: the technique/approach the paper uses.
- Experimental finding: something the experiments observed.
- Performance claim: a quantitative/qualitative result, e.g. "reduces latency by 35%".
- Research conclusion: what the authors conclude.
- Advantage: a benefit the authors claim for their approach.
- Limitation: a limitation the authors acknowledge.

Ignore authors, affiliations, funding, references, acknowledgements, and generic
background/motivation sentences that assert nothing checkable.

Each claim needs a short "topic" label (2-4 words, e.g. "Cloud Latency", "Model Accuracy",
"Training Cost") naming WHAT the claim is about, using consistent/specific labels so claims
about the same underlying subject get the same topic wording across papers.

Base every claim ONLY on the retrieved chunks given for that paper - never invent a claim, a
number, or a fact that is not present in them, and never use outside/general knowledge.

Respond with ONLY a JSON object of the form:
{{"papers": [{{"index": 1, "claims": [{{"topic": "...", "claim_text": "...", "claim_type": "..."}}]}}]}}
The "papers" array must have one entry per paper given, "index" matching the 1-based position
below. No markdown fences, no commentary."""


def _build_claim_extraction_message(papers: List[Paper], contexts: Dict[int, str]) -> str:
    blocks = []
    for i, p in enumerate(papers):
        context = contexts.get(i, "")
        blocks.append(
            f"Paper {i + 1}: {p.title}\n"
            f"Retrieved chunks:\n{context or '(no relevant chunks retrieved)'}"
        )
    return "\n\n".join(blocks)


def extract_claims(
    papers: List[Paper], paper_labels: List[str], rag_cache: Dict[str, PaperIndex],
) -> Tuple[Dict[int, List[Claim]], str, str]:
    """Returns ({paper index (0-based) -> claims}, pipeline_status, reason).
    Empty per-paper lists if the LLM is unavailable or its output can't be
    parsed/matched - the status/reason tell the caller (and ultimately the
    UI) WHICH of those happened, instead of an empty dict looking
    identical to "genuinely no claims in these papers". `rag_cache` is the
    shared {paper title -> PaperIndex} built once in app.py."""
    empty = {i: [] for i in range(len(papers))}
    if not papers:
        return empty, pipeline_status.INVALID_INPUT, pipeline_status.MESSAGES[pipeline_status.INVALID_INPUT]
    if not llm.is_available():
        return empty, pipeline_status.LLM_UNAVAILABLE, pipeline_status.MESSAGES[pipeline_status.LLM_UNAVAILABLE]
    if llm.is_daily_quota_exhausted():
        return empty, pipeline_status.DAILY_QUOTA_EXCEEDED, pipeline_status.MESSAGES[pipeline_status.DAILY_QUOTA_EXCEEDED]

    contexts = retrieve_claim_evidence(papers, rag_cache)
    if not any(c.strip() for c in contexts.values()):
        logger.warning("Claim extraction: RAG retrieved zero usable chunks for all %d paper(s).", len(papers))
        return empty, pipeline_status.RETRIEVAL_FAILURE, pipeline_status.MESSAGES[pipeline_status.RETRIEVAL_FAILURE]

    try:
        content = llm._call_groq(
            messages=[
                {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": _build_claim_extraction_message(papers, contexts)},
            ],
            max_tokens=min(5000, 700 * len(papers)),
            temperature=0.1,
            json_mode=True,
            agent_name="ContradictionClaimExtraction",
            paper_id=f"batch[{len(papers)}]",
            cache_version=config.LLM_CACHE_VERSION,
        )
    except llm.GroqDailyQuotaExceeded as exc:
        logger.warning("Claim extraction: daily Groq quota exhausted - %s", exc)
        return empty, pipeline_status.DAILY_QUOTA_EXCEEDED, pipeline_status.MESSAGES[pipeline_status.DAILY_QUOTA_EXCEEDED]
    data = llm.parse_json(content) if content else None
    if not isinstance(data, dict) or not isinstance(data.get("papers"), list):
        if content:
            logger.warning("Claim extraction returned unusable JSON: %s", content[:200])
        return empty, pipeline_status.EXTRACTION_FAILURE, pipeline_status.MESSAGES[pipeline_status.EXTRACTION_FAILURE]

    by_index: Dict[int, dict] = {}
    for entry in data["papers"]:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("index")) - 1
        except (TypeError, ValueError):
            continue
        by_index[idx] = entry

    result = dict(empty)
    for i in range(len(papers)):
        entry = by_index.get(i)
        if not entry or not isinstance(entry.get("claims"), list):
            continue
        claims = []
        for c in entry["claims"]:
            if not isinstance(c, dict):
                continue
            claim_text = str(c.get("claim_text") or "").strip()
            if not claim_text:
                continue
            claims.append(Claim(
                paper_label=paper_labels[i],
                topic=str(c.get("topic") or "General").strip(),
                claim_text=claim_text,
                claim_type=str(c.get("claim_type") or "Unspecified").strip(),
            ))
        result[i] = claims

    for i, paper in enumerate(papers):
        claims = result.get(i, [])
        logger.debug("[CLAIM DEBUG] Paper=%s | Number of claims=%d", paper.title[:70], len(claims))
        for j, c in enumerate(claims):
            logger.debug(
                "  Claim ID=%d | type=%s | text=%s...",
                j, c.claim_type, c.claim_text[:80],
            )

    total_claims = sum(len(v) for v in result.values())
    if total_claims == 0:
        return result, pipeline_status.INSUFFICIENT_EVIDENCE, (
            "The LLM call succeeded, but found no checkable scientific claims in the retrieved "
            "evidence for any selected paper - often because only an abstract was available."
        )
    return result, pipeline_status.SUCCESS, ""


# --------------------------------------------------------------------------
# Step 2: match claim pairs by topic similarity (dense embeddings, not LLM)
# --------------------------------------------------------------------------

def match_claim_pairs(claims_by_paper: Dict[int, List[Claim]]) -> List[Tuple[Claim, Claim, float]]:
    """Cross-paper claim pairs above the topic-similarity threshold, most
    similar first, capped at CONTRADICTION_MAX_CLAIM_PAIRS. Claims from the
    same paper are never paired - contradiction detection is inherently
    cross-document."""
    flat: List[Tuple[int, Claim]] = [
        (paper_idx, claim)
        for paper_idx, claims in claims_by_paper.items()
        for claim in claims
    ]
    claim_counts = {idx: len(claims) for idx, claims in claims_by_paper.items()}
    possible_cross_paper_pairs = sum(
        a * b for i, a in claim_counts.items() for j, b in claim_counts.items() if i < j
    )
    if len(flat) < 2:
        logger.debug(
            "[COMPARISON DEBUG] Claims per paper=%s | Total possible pairs=%d | Candidate pairs after filtering=0 "
            "(fewer than 2 total claims exist to pair at all)",
            claim_counts, possible_cross_paper_pairs,
        )
        return []

    vecs = embed_texts([claim.claim_text for _, claim in flat], is_query=False)

    pairs: List[Tuple[Claim, Claim, float]] = []
    for i in range(len(flat)):
        idx_i, claim_i = flat[i]
        for j in range(i + 1, len(flat)):
            idx_j, claim_j = flat[j]
            if idx_i == idx_j:
                continue
            similarity = float(np.dot(vecs[i], vecs[j]))
            if similarity >= config.CONTRADICTION_TOPIC_SIMILARITY_THRESHOLD:
                pairs.append((claim_i, claim_j, similarity))

    pairs.sort(key=lambda x: x[2], reverse=True)
    capped = pairs[: config.CONTRADICTION_MAX_CLAIM_PAIRS]
    logger.debug(
        "[COMPARISON DEBUG] Claims per paper=%s | Total possible pairs=%d | Candidate pairs after filtering=%d "
        "(similarity >= %.2f) | Pairs sent for semantic comparison=%d (after MAX_CLAIM_PAIRS cap)",
        claim_counts, possible_cross_paper_pairs, len(pairs),
        config.CONTRADICTION_TOPIC_SIMILARITY_THRESHOLD, len(capped),
    )
    return capped


# --------------------------------------------------------------------------
# Step 3: lightweight contradiction prediction (rule-based, no LLM)
# --------------------------------------------------------------------------

_CONTRADICTION_DIRECTION_TERMS = {
    "increase": "positive",
    "increases": "positive",
    "increased": "positive",
    "improve": "positive",
    "improves": "positive",
    "improved": "positive",
    "boost": "positive",
    "raise": "positive",
    "reduce": "negative",
    "reduces": "negative",
    "reduced": "negative",
    "decrease": "negative",
    "decreases": "negative",
    "decreased": "negative",
    "drop": "negative",
    "drops": "negative",
    "decline": "negative",
    "declines": "negative",
    "worsen": "negative",
    "worse": "negative",
    "lower": "negative",
    "higher": "positive",
    "better": "positive",
    "outperform": "positive",
    "underperform": "negative",
    "surpass": "positive",
    "lag": "negative",
    "fail": "negative",
    "succeed": "positive",
}


def _tokenize_for_prediction(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-zA-Z]+", (text or "").lower()) if tok]


def _extract_numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _estimate_contradiction_score(claim_a: Claim, claim_b: Claim, similarity: float) -> Tuple[float, str]:
    text_a = claim_a.claim_text.lower()
    text_b = claim_b.claim_text.lower()
    tokens_a = _tokenize_for_prediction(text_a)
    tokens_b = _tokenize_for_prediction(text_b)

    score = 0.0
    reason_parts = []

    if claim_a.topic and claim_b.topic and claim_a.topic.lower() == claim_b.topic.lower():
        score += 0.1
        reason_parts.append("same topic")

    if similarity >= config.CONTRADICTION_TOPIC_SIMILARITY_THRESHOLD:
        score += 0.15
        reason_parts.append("high similarity")

    a_terms = [tok for tok in tokens_a if tok in _CONTRADICTION_DIRECTION_TERMS]
    b_terms = [tok for tok in tokens_b if tok in _CONTRADICTION_DIRECTION_TERMS]
    if a_terms and b_terms:
        a_polarity = _CONTRADICTION_DIRECTION_TERMS.get(a_terms[0], "neutral")
        b_polarity = _CONTRADICTION_DIRECTION_TERMS.get(b_terms[0], "neutral")
        if a_polarity != b_polarity and a_polarity != "neutral" and b_polarity != "neutral":
            score += 0.55
            reason_parts.append("opposite direction terms")

    nums_a = _extract_numbers(text_a)
    nums_b = _extract_numbers(text_b)
    if nums_a and nums_b and len(nums_a) == len(nums_b):
        if any(x < 0 for x in nums_a) or any(x < 0 for x in nums_b):
            pass
    if nums_a and nums_b:
        score += 0.05
        reason_parts.append("numeric evidence")

    if any(marker in text_a + text_b for marker in ["vs", "versus", "compared to", "relative to"]):
        score += 0.05
        reason_parts.append("comparison framing")

    if any(marker in text_a + text_b for marker in ["not", "no", "never"]):
        score += 0.03
        reason_parts.append("negation")

    score = min(1.0, score)
    reason = "; ".join(reason_parts) if reason_parts else "weak overlap"
    return score, reason


def predict_contradiction_pairs(pairs: List[Tuple[Claim, Claim, float]]) -> List[dict]:
    """Rule-based prediction of which claim pairs are likely contradictions.

    This is intentionally lightweight and explainable: it uses the same claim
    text and similarity signal already available in the contradiction pipeline,
    without requiring an extra LLM call.
    """
    predictions = []
    for claim_a, claim_b, similarity in pairs:
        score, reason = _estimate_contradiction_score(claim_a, claim_b, similarity)
        if score >= 0.8:
            prediction = "Likely Contradiction"
        elif score >= 0.6:
            prediction = "Possible Contradiction"
        else:
            prediction = "Low Risk"
        predictions.append({
            "prediction": prediction,
            "score": round(score, 3),
            "reason": reason,
        })
    return predictions


# --------------------------------------------------------------------------
# Step 4: classify each matched pair (one batched Groq call)
# --------------------------------------------------------------------------

CLASSIFICATION_SYSTEM_PROMPT = """You compare pairs of scientific claims from different academic \
papers and classify the relationship between each pair into EXACTLY ONE category:
- "Agreement": both claims support the same conclusion.
- "Contradiction": the claims directly conflict / assert opposite things about the same topic
  under comparable conditions.
- "Partial Contradiction": the claims overlap but diverge in degree, scope, or conditions rather
  than flatly opposing each other.
- "Different Context": the claims use similar wording but concern different
  conditions/settings/scope, so they are not truly comparable.
- "Insufficient Evidence": there isn't enough detail in either claim to judge agreement vs
  contradiction.

Be conservative: only use "Contradiction" or "Partial Contradiction" when the claims genuinely
assert opposing things about a comparable topic and setting. If in doubt, prefer "Different
Context" or "Insufficient Evidence" over inventing a contradiction - never hallucinate a
contradiction that the claim text does not actually support.

For "supporting_evidence", quote or closely paraphrase the exact claim text you were given for
each side - never invent a section number, page number, or citation that was not given to you.

Respond with ONLY a JSON object of the form:
{"comparisons": [{"index": 1, "classification": "...", "confidence": <integer 0-100>,
"reason": "...", "supporting_evidence": ["...", "..."]}]}
One entry per pair given, "index" matching the 1-based position below. No markdown fences,
no commentary."""


def _build_classification_message(pairs: List[Tuple[Claim, Claim, float]]) -> str:
    blocks = []
    for i, (a, b, sim) in enumerate(pairs, start=1):
        blocks.append(
            f"Pair {i} (topic similarity {sim:.2f}):\n"
            f"{a.paper_label}\nTopic: {a.topic}\nClaim ({a.claim_type}): {a.claim_text}\n\n"
            f"{b.paper_label}\nTopic: {b.topic}\nClaim ({b.claim_type}): {b.claim_text}"
        )
    return "\n\n---\n\n".join(blocks)


def _unclassified_fallback(pairs: List[Tuple[Claim, Claim, float]], reason: str) -> List[ClaimComparison]:
    return [
        ClaimComparison(
            paper_a=a.paper_label, paper_b=b.paper_label,
            topic=a.topic, claim_a=a.claim_text, claim_b=b.claim_text,
            classification="Insufficient Evidence", confidence=0,
            reason=reason, supporting_evidence=[],
        )
        for a, b, _ in pairs
    ]


def classify_pairs(pairs: List[Tuple[Claim, Claim, float]]) -> List[ClaimComparison]:
    if not pairs:
        return []

    if not llm.is_available():
        return _unclassified_fallback(pairs, "GROQ_API_KEY not configured - classification unavailable.")
    if llm.is_daily_quota_exhausted():
        return _unclassified_fallback(pairs, "Daily Groq token quota exhausted - classification unavailable.")

    try:
        content = llm._call_groq(
            messages=[
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": _build_classification_message(pairs)},
            ],
            max_tokens=min(5000, 450 * len(pairs)),
            temperature=0.1,
            json_mode=True,
            agent_name="ContradictionClassification",
            paper_id=f"pairs[{len(pairs)}]",
            cache_version=config.LLM_CACHE_VERSION,
        )
    except llm.GroqDailyQuotaExceeded as exc:
        logger.warning("Contradiction classification: daily Groq quota exhausted - %s", exc)
        return _unclassified_fallback(pairs, "Daily Groq token quota exhausted - classification unavailable.")
    data = llm.parse_json(content) if content else None
    by_index: Dict[int, dict] = {}
    if isinstance(data, dict) and isinstance(data.get("comparisons"), list):
        for entry in data["comparisons"]:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            by_index[idx] = entry
    elif content:
        logger.warning("Contradiction classification returned unusable JSON: %s", content[:200])

    comparisons = []
    for i, (a, b, _) in enumerate(pairs, start=1):
        entry = by_index.get(i)
        classification = str(entry.get("classification") or "").strip() if entry else ""
        if classification not in CLASSIFICATIONS:
            classification = "Insufficient Evidence"
        try:
            confidence = int(entry.get("confidence")) if entry else 0
        except (TypeError, ValueError):
            confidence = 0
        confidence = max(0, min(100, confidence))
        reason = str(entry.get("reason") or "").strip() if entry else "Classification unavailable."
        evidence = entry.get("supporting_evidence") if entry else []
        evidence = [str(e).strip() for e in evidence] if isinstance(evidence, list) else []

        comparisons.append(ClaimComparison(
            paper_a=a.paper_label, paper_b=b.paper_label,
            topic=a.topic, claim_a=a.claim_text, claim_b=b.claim_text,
            classification=classification, confidence=confidence,
            reason=reason or "No reason returned.", supporting_evidence=evidence,
        ))

    counts = Counter(c.classification for c in comparisons)
    logger.debug(
        "[COMPARISON DEBUG] Compared=%d | Agreement=%d | Contradiction=%d | Partial Contradiction=%d | "
        "Different Context=%d | Insufficient Evidence=%d",
        len(comparisons), counts.get("Agreement", 0), counts.get("Contradiction", 0),
        counts.get("Partial Contradiction", 0), counts.get("Different Context", 0),
        counts.get("Insufficient Evidence", 0),
    )
    return comparisons
