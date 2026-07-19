"""
Full-PDF RAG Knowledge Extraction Agent (OOP, SOLID).

Replaces knowledge_analysis.py's single-combined-query retrieval with
PER-FIELD semantic retrieval: each knowledge-field GROUP (Problem &
Objectives, Technique & Model, Dataset & Preprocessing, Training
Strategy, Implementation, Evaluation & Results, Limitations &
Contributions, plus the existing Research Gap / Future Work queries) gets
its own dedicated RAG query with its own top-K, instead of competing with
every other field for a shared top-5 pool. This is the one thing the
prior single-combined-query design genuinely under-served: a field like
"Dataset" could get crowded out of the merged context by "Evaluation
Metrics" chunks scoring marginally higher on the shared query.

Everything upstream of retrieval is intentionally UNCHANGED and reused,
not rebuilt: PDF acquisition/cleaning is pdf_extraction.py (PyMuPDF, full
document, on-disk cache), chunking/embedding/FAISS indexing is
rag_pipeline.py (BAAI/bge-small-en-v1.5, in-memory per-search - no
persistent vector DB by design), and the LLM call (Groq, with its own
429-retry) is llm.py. This module owns exactly one concern: turning a
PaperIndex into a structured knowledge record via smarter retrieval and
an expanded extraction schema.

Class responsibilities (single-responsibility, dependency-injected so
each is independently testable):
    KnowledgeField          - declarative spec for one extracted field
                               (key, column label, prompt description).
    RetrievalQueryGroup      - declarative spec for one semantic query
                               (label, query text, top_k, which fields it
                               is meant to surface evidence for).
    FieldGroupRetriever      - runs every RetrievalQueryGroup against one
                               paper's PaperIndex and merges/dedupes the
                               results (thin orchestration over
                               rag_pipeline.retrieve_context - no new
                               retrieval math).
    ExtractionPromptBuilder   - builds the Groq system/user prompt from
                               the retrieved, section-tagged chunks and
                               the field schema.
    PaperKnowledgeExtractor   - orchestrates ONE paper end-to-end:
                               retrieve -> prompt -> call Groq -> parse ->
                               validate -> ExtractionResult. Never raises;
                               a failure at any step degrades that paper's
                               fields to NOT_REPORTED rather than stopping
                               the batch.
    KnowledgeExtractionPipeline - orchestrates ALL papers concurrently
                               (mirrors the prior _extract_rows), with the
                               Step 8-style per-paper log line.

knowledge_analysis.py's analyze_papers() calls KnowledgeExtractionPipeline
and reshapes ExtractionResult objects into the exact row shape it already
built internally - every downstream consumer (Overall Analysis, Problem-
Solution Analysis, Contradiction Detection, Root Cause Analysis, Inventor
Agent, Proposal Preview) keeps working completely unchanged.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
import llm
from rag_pipeline import NOT_FOUND_IN_CONTEXT, PaperIndex, RetrievedChunk, retrieve_context
from utils import Paper

logger = logging.getLogger(__name__)

NOT_REPORTED = NOT_FOUND_IN_CONTEXT


# --------------------------------------------------------------------------
# Declarative schema - the extraction field list and the retrieval query
# groups that are meant to surface evidence for them. Adding a field is a
# one-line change here, not a scattered edit across the module.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class KnowledgeField:
    key: str            # dict/JSON key, e.g. "model_architecture"
    column: str          # display column label, e.g. "Model Architecture"
    description: str     # one line telling the LLM what belongs in this field


KNOWLEDGE_FIELDS: List[KnowledgeField] = [
    KnowledgeField("research_domain", "Research Domain", "the broad academic field."),
    KnowledgeField("objectives", "Objectives", "the specific research objective(s) this paper sets out to achieve."),
    KnowledgeField("ai_technique", "AI Technique", "the main AI method/approach."),
    KnowledgeField("ai_model", "AI Model", 'the specific model/algorithm name(s), e.g. "PPO", "ResNet-50".'),
    KnowledgeField("model_architecture", "Model Architecture", "structural details of the model (layers, components, how they connect), distinct from just the model's name."),
    KnowledgeField("llm_used", "LLM", 'any named large language model used, e.g. "GPT-4", "Llama-3".'),
    KnowledgeField("agent_framework", "Agent Framework", 'any named agent/orchestration framework, e.g. "LangChain", "AutoGen".'),
    KnowledgeField("embedding_model", "Embedding Model", 'any named embedding model, e.g. "BGE", "text-embedding-ada-002".'),
    KnowledgeField("dataset", "Dataset", "named dataset(s) or data source(s)."),
    KnowledgeField("preprocessing", "Preprocessing", "data preprocessing/preparation steps applied before training or evaluation."),
    KnowledgeField("optimizer", "Optimizer", 'the training optimizer, e.g. "Adam", "SGD".'),
    KnowledgeField("loss_function", "Loss Function", "the loss function used."),
    KnowledgeField("learning_rate", "Learning Rate", "the learning rate value(s) used."),
    KnowledgeField("epochs", "Epochs", "number of training epochs."),
    KnowledgeField("hyperparameters", "Hyperparameters", "other notable hyperparameters (batch size, hidden dims, etc.) as one string."),
    KnowledgeField("training_strategy", "Training Strategy", 'the broader training methodology, e.g. "transfer learning", "curriculum learning", "end-to-end fine-tuning" - distinct from raw hyperparameter values.'),
    KnowledgeField("programming_language", "Programming Language", "named programming language(s)."),
    KnowledgeField("framework", "Framework", 'named software/ML framework, e.g. "PyTorch", "TensorFlow".'),
    KnowledgeField("hardware", "Hardware", 'named compute hardware, e.g. "NVIDIA A100", "TPU v4".'),
    KnowledgeField("evaluation_metrics", "Evaluation Metrics", "metric(s) used to evaluate results."),
    KnowledgeField("accuracy", "Accuracy", 'best-reported accuracy figure, with context, e.g. "95.2%".'),
    KnowledgeField("precision", "Precision", "best-reported precision figure."),
    KnowledgeField("recall", "Recall", "best-reported recall figure."),
    KnowledgeField("f1_score", "F1 Score", "best-reported F1 score."),
    KnowledgeField("auc", "AUC", "best-reported AUC value."),
    KnowledgeField("map_score", "mAP", "best-reported mAP value."),
    KnowledgeField("research_gap", "Research Gap", "the broader unsolved problem in the field that motivates or remains after this work."),
    KnowledgeField("limitations", "Limitations", "specific weaknesses/constraints of this paper's own approach, acknowledged by the authors."),
    KnowledgeField("future_work", "Future Work", "next steps the authors propose (or reasonably-inferred ones from Discussion/Limitations chunks if not explicitly stated)."),
    KnowledgeField("key_contributions", "Key Contributions", "one or two concise sentences on what this paper contributes."),
]

FIELD_KEYS: List[str] = [f.key for f in KNOWLEDGE_FIELDS]
FIELD_TO_COLUMN: Dict[str, str] = {f.key: f.column for f in KNOWLEDGE_FIELDS}


@dataclass(frozen=True)
class RetrievalQueryGroup:
    label: str
    query_text: str
    top_k: int


RETRIEVAL_QUERY_GROUPS: List[RetrievalQueryGroup] = [
    RetrievalQueryGroup(
        "Problem, Domain & Objectives",
        "research problem, objective, motivation, research domain, problem statement",
        config.RAG_TOP_K_PER_FIELD_GROUP,
    ),
    RetrievalQueryGroup(
        "Technique, Model & Architecture",
        "AI technique, model architecture, algorithm, large language model, agent framework, embedding model",
        config.RAG_TOP_K_PER_FIELD_GROUP,
    ),
    RetrievalQueryGroup(
        "Dataset & Preprocessing",
        "dataset, data collection, data preprocessing, data preparation",
        config.RAG_TOP_K_PER_FIELD_GROUP,
    ),
    RetrievalQueryGroup(
        "Training Strategy & Optimization",
        "training strategy, optimizer, loss function, learning rate, epochs, hyperparameters",
        config.RAG_TOP_K_PER_FIELD_GROUP,
    ),
    RetrievalQueryGroup(
        "Implementation",
        "programming language, software framework, hardware, compute environment",
        config.RAG_TOP_K_PER_FIELD_GROUP,
    ),
    RetrievalQueryGroup(
        "Evaluation & Results",
        "evaluation metrics, accuracy, precision, recall, F1 score, AUC, mAP, experimental results",
        config.RAG_TOP_K_PER_FIELD_GROUP,
    ),
    RetrievalQueryGroup(
        "Research Gap",
        config.RAG_QUERY_RESEARCH_GAP,
        config.RAG_TOP_K_RESEARCH_GAP,
    ),
    RetrievalQueryGroup(
        "Future Work",
        config.RAG_QUERY_FUTURE_WORK,
        config.RAG_TOP_K_FUTURE_WORK,
    ),
    RetrievalQueryGroup(
        "Limitations & Key Contributions",
        "limitations, weaknesses, key contributions, novelty, main contribution",
        config.RAG_TOP_K_PER_FIELD_GROUP,
    ),
]


def _build_extraction_system_prompt() -> str:
    field_lines = "\n".join(f"- {f.key}: {f.description}" for f in KNOWLEDGE_FIELDS)
    return f"""You are a research analyst extracting structured technical knowledge from a set of \
RETRIEVED CHUNKS from ONE academic paper for a systematic literature review. These chunks were \
selected by semantic search (one dedicated query per knowledge-field group, so every field below \
had its own chance to surface relevant evidence) as the passages most relevant to this extraction \
task - they are NOT the whole paper. Each chunk is tagged with its source section, e.g. \
"[Methodology] ...text...".

Any References or Acknowledgements text was already excluded before retrieval - never extract \
from citations.

Extract ONLY information explicitly stated in the given retrieved chunks. Never guess, never use \
outside/general knowledge about the paper or topic, and never use anything not present in the \
given chunks - even if you believe you know the answer from training data, if it is not in the \
chunks below you must not state it. If a field is not explicitly present anywhere in the given \
chunks, respond with the exact string "{NOT_REPORTED}" for it - never leave it blank, never write \
"N/A", "Not Mentioned", or "Unknown".

Exception - future_work only: if no chunk explicitly discusses future work, you MAY infer likely \
future directions ONLY from Discussion/Limitations chunks that ARE present below, phrased to make \
clear it's inferred (e.g. "Not explicitly stated; limitations suggest exploring..."). If nothing \
in the given chunks supports even that, use "{NOT_REPORTED}".

Respond with concise bullet-point-style phrases per field, never full paragraphs - a field's value \
should read like "Adam, lr=1e-4, batch size=32", not a sentence explaining it.

Fields (respond with exactly these {len(KNOWLEDGE_FIELDS)} keys):
{field_lines}

Respond with ONLY a JSON object of these {len(KNOWLEDGE_FIELDS)} keys, each a string value. No \
markdown fences, no commentary."""


EXTRACTION_SYSTEM_PROMPT = _build_extraction_system_prompt()


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

class FieldGroupRetriever:
    """Runs every RetrievalQueryGroup against one paper's already-built
    FAISS index and merges/dedupes the results - thin orchestration over
    rag_pipeline.retrieve_context, which already does the per-query
    embedding, search and score-based dedup; this class only supplies the
    more granular, per-field-group query list instead of one combined
    query, and reports how many distinct queries were actually issued."""

    def __init__(self, query_groups: Optional[List[RetrievalQueryGroup]] = None):
        self._query_groups = query_groups or RETRIEVAL_QUERY_GROUPS

    def retrieve(self, paper_index: PaperIndex) -> "RetrievalOutcome":
        queries = [(g.query_text, g.top_k) for g in self._query_groups]
        context, retrieved = retrieve_context(
            paper_index, queries=queries, max_total_chunks=config.RAG_MAX_CHUNKS_KNOWLEDGE_COMBINED,
        )
        return RetrievalOutcome(
            context=context, retrieved_chunks=retrieved, queries_issued=len(self._query_groups),
        )


@dataclass
class RetrievalOutcome:
    context: str
    retrieved_chunks: List[RetrievedChunk]
    queries_issued: int


# --------------------------------------------------------------------------
# Prompt building
# --------------------------------------------------------------------------

class ExtractionPromptBuilder:
    """Builds the Groq user message from a paper's title and its
    retrieved-chunk context. Kept separate from PaperKnowledgeExtractor
    so the prompt format can be tested/tuned in isolation."""

    def build_user_message(self, paper_title: str, context: str) -> str:
        return f"Paper title: {paper_title}\n\nRetrieved chunks:\n\n{context}"


# --------------------------------------------------------------------------
# Per-paper extraction
# --------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    paper_title: str
    fields: Dict[str, str] = field(default_factory=dict)
    evidence_source: str = "Not Available"
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    queries_issued: int = 0
    success: bool = False

    def to_row(self) -> dict:
        """Same shape knowledge_analysis.py's internal rows have always
        had - `_evidence_source`/`_retrieved_chunks` keys included, so the
        adapter in knowledge_analysis.py needs zero reshaping logic."""
        row = dict(self.fields)
        row["_evidence_source"] = self.evidence_source
        row["_retrieved_chunks"] = self.retrieved_chunks
        return row


class PaperKnowledgeExtractor:
    """Orchestrates ONE paper end-to-end: per-field-group retrieval ->
    prompt -> Groq call (retry-on-429 already handled inside llm.py) ->
    parse -> validate -> ExtractionResult. Never raises - any failure
    degrades this paper's fields to NOT_REPORTED rather than stopping the
    batch (Step 9: one paper's failure must never stop the rest)."""

    def __init__(
        self,
        retriever: Optional[FieldGroupRetriever] = None,
        prompt_builder: Optional[ExtractionPromptBuilder] = None,
    ):
        self._retriever = retriever or FieldGroupRetriever()
        self._prompt_builder = prompt_builder or ExtractionPromptBuilder()

    def extract(self, paper: Paper, paper_index: PaperIndex) -> ExtractionResult:
        result = ExtractionResult(
            paper_title=paper.title, fields={f.key: NOT_REPORTED for f in KNOWLEDGE_FIELDS},
        )

        try:
            outcome = self._retriever.retrieve(paper_index)
        except Exception as exc:  # noqa: BLE001 - retrieval issues must never break the batch
            logger.error("Knowledge extraction retrieval FAILED for %s: %s", paper.title[:60], exc)
            return result

        result.evidence_source = paper_index.source_label if outcome.retrieved_chunks else "Not Available"
        result.retrieved_chunks = outcome.retrieved_chunks
        result.queries_issued = outcome.queries_issued

        if not outcome.context.strip() or not llm.is_available():
            logger.info(
                "Knowledge extraction for %s: %s",
                paper.title[:60],
                "no LLM configured" if not llm.is_available() else "no retrievable context",
            )
            return result

        content = llm._call_groq(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": self._prompt_builder.build_user_message(paper.title, outcome.context)},
            ],
            max_tokens=1500,  # Groq counts this reservation against the account's TPM budget
                              # regardless of actual usage - kept modest (measured: 30 concise
                              # bullet-style fields need well under this) so it doesn't eat into
                              # the shared per-request token ceiling. See config.
                              # RAG_MAX_CHUNKS_KNOWLEDGE_COMBINED for the full budget breakdown.
            temperature=0.1,
            json_mode=True,
        )
        data = llm.parse_json(content) if content else None
        if isinstance(data, dict):
            for f in KNOWLEDGE_FIELDS:
                value = str(data.get(f.key) or "").strip()
                result.fields[f.key] = value or NOT_REPORTED
            result.success = True
        elif content:
            logger.error(
                "Knowledge extraction returned unusable JSON for %s: %s", paper.title[:60], content[:200],
            )
        else:
            logger.error("Knowledge extraction Groq call FAILED for %s (no content returned).", paper.title[:60])

        return result


# --------------------------------------------------------------------------
# Batch orchestration
# --------------------------------------------------------------------------

class KnowledgeExtractionPipeline:
    """Runs PaperKnowledgeExtractor across every paper concurrently
    (mirrors the prior module's per-paper-Groq-call, thread-pooled
    design), logging one line per paper (Step 8: title, chunks available,
    retrieval queries issued, extraction success)."""

    def __init__(self, extractor: Optional[PaperKnowledgeExtractor] = None, max_workers: Optional[int] = None):
        self._extractor = extractor or PaperKnowledgeExtractor()
        self._max_workers = max_workers or config.KNOWLEDGE_EXTRACTION_MAX_WORKERS

    def run(self, papers: List[Paper], rag_cache: Dict[str, PaperIndex]) -> List[ExtractionResult]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not papers:
            return []

        results: List[Optional[ExtractionResult]] = [None] * len(papers)
        workers = min(self._max_workers, len(papers))

        def _run_one(paper: Paper) -> ExtractionResult:
            paper_index = rag_cache.get(paper.title, PaperIndex(paper_id=paper.title))
            result = self._extractor.extract(paper, paper_index)
            logger.info(
                "[Knowledge Extraction] %s | chunks=%d | retrieval_queries=%d | success=%s",
                paper.title[:70], len(paper_index.chunks), result.queries_issued, result.success,
            )
            return result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(_run_one, p): i for i, p in enumerate(papers)}
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    results[i] = future.result()
                except Exception as exc:  # noqa: BLE001 - one paper's failure must never break the rest
                    logger.error("Knowledge extraction FAILED for %s: %s", papers[i].title[:60], exc)
                    results[i] = ExtractionResult(
                        paper_title=papers[i].title, fields={f.key: NOT_REPORTED for f in KNOWLEDGE_FIELDS},
                    )
        return results  # type: ignore[return-value]
