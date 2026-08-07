"""
Central configuration for the Research Paper Retrieval System.

All environment-dependent values (API keys, model names, tunable
pipeline constants) live here so the rest of the codebase never talks
to os.environ directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

# --------------------------------------------------------------------------
# API keys (all optional except none are required to run the app -
# sources without a key are simply skipped at retrieval time)
# --------------------------------------------------------------------------
CORE_API_KEY = os.getenv("CORE_API_KEY", "").strip()
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Model served by Groq for query expansion / summarization / relevance
# explanations. Any Groq-hosted chat model works, e.g.:
#   "llama-3.3-70b-versatile"  (Meta Llama 3.3, default)
#   "qwen/qwen3-32b"           (Qwen 3)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Optional contact email sent to OpenAlex/Crossref "polite pool" endpoints.
# Leave blank to call the public (rate-limited) endpoints anonymously.
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "").strip()

# --------------------------------------------------------------------------
# AI model identifiers
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"

# BGE models are trained to prepend an instruction to the *query* only
# (not to the documents/passages being searched over).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --------------------------------------------------------------------------
# Retrieval / ranking pipeline tuning
# --------------------------------------------------------------------------
RESULTS_PER_SOURCE = 200       # how many results to request from each API
                                # (raised from 20 → 200 so 5 sources can return
                                # up to 1,000 raw papers before deduplication)
REQUEST_TIMEOUT_SECONDS = 30   # per-request network timeout (raised for larger result sets)
TOP_K_AFTER_EMBEDDING = 500    # candidates kept after cosine-similarity ranking
                                # before the (more expensive) cross-encoder runs
                                # (raised from 30 → 500 to cover 1,000+ papers)
TITLE_SIMILARITY_DEDUP_THRESHOLD = 0.88  # difflib ratio above which two
                                          # titles are considered duplicates

# Rate limiting and retry configuration (handles 429 errors from APIs)
ENABLE_SEMANTIC_SCHOLAR = os.getenv("ENABLE_SEMANTIC_SCHOLAR", "true").lower() == "true"
                                         # disable if hitting rate limits too often
API_RETRY_MAX_ATTEMPTS = int(os.getenv("API_RETRY_MAX_ATTEMPTS", "3"))
                                         # max retries for rate-limited (429) API calls
                                         # (lowered from 4 → 3 to fail faster and move on)
API_RETRY_INITIAL_DELAY = float(os.getenv("API_RETRY_INITIAL_DELAY", "1.0"))
                                         # initial delay in seconds, doubles on each retry
                                         # (lowered from 2.0 → 1.0 for faster recovery)

# Academic source display names (used across retrieve.py / app.py)
SOURCE_OPENALEX = "OpenAlex"
SOURCE_CROSSREF = "Crossref"
SOURCE_ARXIV = "arXiv"
SOURCE_CORE = "CORE"
SOURCE_SEMANTIC_SCHOLAR = "Semantic Scholar"

# --------------------------------------------------------------------------
# Contradiction Detection module tuning
# --------------------------------------------------------------------------
CONTRADICTION_TOPIC_SIMILARITY_THRESHOLD = 0.50  # min cosine similarity between two
                                                  # claims (from different papers) before
                                                  # they're even considered "same topic"
                                                  # and sent to the LLM for classification
CONTRADICTION_MAX_CLAIM_PAIRS = 20     # cap on how many matched pairs get LLM-classified
CONTRADICTION_MAX_CLAIMS_PER_PAPER = 6  # requested upper bound in the extraction prompt

# --------------------------------------------------------------------------
# Knowledge Retrieval module tuning
# --------------------------------------------------------------------------
PDF_FETCH_TIMEOUT_SECONDS = 12     # seconds, PDF download — 12s fails fast on unresponsive
                                   # servers instead of waiting the full 20s; most academic
                                   # PDF servers respond in <5s when they're working at all
KNOWLEDGE_EXTRACTION_MAX_WORKERS = 3  # concurrent per-paper Groq calls. Groq free-tier
                                       # enforces ~12k TPM shared budget; 3 workers × ~4k
                                       # tokens each = ~12k, just at the limit without
                                       # colliding. Raised from 2 → 3 for 33% faster extraction.
ROOT_CAUSE_MAX_TOKENS_PER_PAIR = 260  # output budget per contradiction pair being explained

KNOWLEDGE_GAP_CLUSTER_SIMILARITY_THRESHOLD = 0.70  # cosine similarity (vs a cluster's running
                                                     # centroid) for grouping semantically similar
                                                     # Research Gap / Future Work statements before
                                                     # LLM labeling. Empirically, genuinely-same-
                                                     # theme statements score ~0.78-0.86, while
                                                     # same-domain-but-different-theme statements
                                                     # (e.g. "generalization" vs "energy efficiency"
                                                     # gaps, both cloud/edge computing) score
                                                     # ~0.60-0.67 - the threshold sits above that
                                                     # gap so it doesn't merge them.

# --------------------------------------------------------------------------
# Problem-Solution Analysis module tuning
# --------------------------------------------------------------------------
PROBLEM_CLUSTER_SIMILARITY_THRESHOLD = 0.75  # cosine similarity (vs a cluster's running
                                              # centroid) for merging semantically similar
                                              # research-problem labels. Empirically calibrated
                                              # on short 2-4 word phrases (e.g. "High Latency" /
                                              # "High Response Time" scored ~0.73-0.89 within a
                                              # true match, ~0.43-0.71 across distinct problems) -
                                              # set a bit higher than the (longer-sentence) gap
                                              # threshold because centroid-averaging over 3+ short
                                              # phrases can otherwise drift enough to falsely merge
                                              # adjacent-but-distinct problems (e.g. "resource
                                              # utilization" absorbing "energy consumption").

# --------------------------------------------------------------------------
# Inventor Agent tuning - this module does no PDF fetching or per-paper
# extraction of its own, so the only budget that matters is the single
# synthesis call's context size.
# --------------------------------------------------------------------------
INVENTOR_MAX_TOKENS = 2500          # output budget for the single synthesis call
INVENTOR_MAX_GAPS_FUTURE_WORK = 8   # top-N (by frequency) gap/future-work rows sent as context
INVENTOR_MAX_CONTRADICTIONS = 10    # top-N contradiction/partial-contradiction rows sent as context

# --------------------------------------------------------------------------
# RAG pipeline (rag_pipeline.py) - replaces "send the whole extracted PDF
# to the LLM" everywhere with "chunk once per paper, embed once, index in
# FAISS once, then every downstream agent retrieves only its own top-K
# most relevant chunks via a task-specific semantic query."
# --------------------------------------------------------------------------
RAG_CHUNK_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RAG_CHUNK_EMBEDDING_MODEL_FALLBACK = "intfloat/e5-base-v2"  # used only if the primary
                                                              # model can't be loaded
                                                              # (e.g. HF Hub unreachable)

# Chunk size deliberately deviates from the requested 800-1000 tokens: live
# testing surfaced a real conflict between that range and the requested
# embedding model. BAAI/bge-small-en-v1.5 is a BERT-based model with a hard
# max_seq_length of 512 tokens - confirmed at runtime via the transformers
# warning "Token indices sequence length is longer than the specified
# maximum sequence length for this model (891 > 512)" when 900-token
# chunks were actually embedded. Anything past token 512 is silently
# truncated before embedding, so a 900-token chunk's embedding vector
# would only ever represent its first ~57% - the similarity search itself
# would be scoring against a truncated chunk while the full (longer) text
# still gets sent to the LLM if retrieved. That defeats the actual purpose
# of chunking (accurate, complete-chunk semantic retrieval), so retrieval
# correctness took priority over literal 800-1000 compliance: chunk size
# is set safely under 512 (with headroom for the 2 special tokens
# SentenceTransformer's tokenizer adds automatically), and overlap is
# scaled down to keep roughly the same overlap-to-chunk-size ratio as the
# originally requested 150-200/800-1000 (~19%).
RAG_CHUNK_SIZE_TOKENS = 480
RAG_CHUNK_OVERLAP_TOKENS = 96

# Per-agent top-K: kept modest on purpose. A single chunk is up to 480
# tokens, so even top_k=3-4 is a fraction of a typical paper's full text -
# the goal is retrieval PRECISION (only genuinely relevant chunks) rather
# than just a smaller token count for its own sake.
RAG_TOP_K_PER_FIELD_GROUP = 4   # knowledge_extraction_agent.py: one query per
                                 # field group (Problem/Objectives, Technique/
                                 # Model, Dataset/Preprocessing, Training,
                                 # Implementation, Evaluation/Results,
                                 # Limitations/Contributions), so no single
                                 # field's evidence gets crowded out of a
                                 # shared pool by a higher-scoring different field
RAG_TOP_K_RESEARCH_GAP = 2
RAG_TOP_K_FUTURE_WORK = 2
RAG_MAX_CHUNKS_KNOWLEDGE_COMBINED = 12  # cap after merging+deduping every field-group
                                         # query above, so the one Knowledge Extraction LLM
                                         # call still gets a bounded, single combined context.
                                         # CONFIRMED via a live Groq probe: this account's
                                         # `on_demand` tier enforces a hard 12,000-token-per-
                                         # request/per-minute budget - 24 chunks (~11,500
                                         # tokens of context alone, before the system prompt
                                         # and output reservation) blew straight through it,
                                         # causing real 400/413 failures during Knowledge
                                         # Extraction. 12 chunks + this module's system prompt
                                         # + max_tokens=1500 measured at ~7,300 tokens total -
                                         # comfortably under the limit with headroom for
                                         # concurrent papers.
RAG_TOP_K_PROBLEM_SOLUTION = 3
RAG_TOP_K_CONTRADICTION = 3
RAG_TOP_K_ROOT_CAUSE = 3
RAG_TOP_K_INVENTOR_PER_PAPER = 1  # Inventor Agent already has rich structured summaries
                                   # from every other agent; RAG chunks here are only
                                   # supplementary grounding, not primary context

# Task-specific semantic retrieval queries (Step 5 of the spec) - the text
# embedded and searched against each paper's chunk index. Deliberately
# phrased as topic/keyword lists (not full questions) since that is what a
# bi-encoder retrieval query is actually being embedded against - the chunk
# content itself.
# Knowledge Extraction's field-group retrieval queries live in
# knowledge_extraction_agent.RETRIEVAL_QUERY_GROUPS (one query per field
# group, not a single combined query) - it reuses these two directly as
# two of those groups.
RAG_QUERY_RESEARCH_GAP = "limitations, discussion, conclusion, research gap, open problems"
RAG_QUERY_FUTURE_WORK = "future work, conclusion, discussion, next steps, future research directions"
RAG_QUERY_PROBLEM_SOLUTION = "problem statement, research problem, methodology, results"
RAG_QUERY_CONTRADICTION = "claims, experimental results, evaluation, discussion, findings"
RAG_QUERY_ROOT_CAUSE = "methodology, experimental setup, results, discussion"
RAG_QUERY_INVENTOR = "best results, highest accuracy, datasets, models, future work, research gaps"

# --------------------------------------------------------------------------
# Paper Validation Agent (paper_validation.py) - runs on the raw extracted
# text immediately after PDF extraction/cleaning and BEFORE chunking/
# embedding/FAISS indexing. Every downstream agent (Knowledge Extraction,
# Problem-Solution Analysis, Contradiction Detection, Root Cause Analysis,
# Inventor Agent) only ever sees papers that passed this validation.
#
# Deliberately lenient: a paper is rejected ONLY for genuinely unusable
# input (PDF unreadable, extracted text too short, or degenerate/garbage
# text) - never for missing a specific section. A paper missing its
# Conclusion or Future Work section is still processed; Knowledge
# Extraction reports "Not Found in Retrieved Context" for whichever
# specific fields have no evidence, rather than the whole paper being
# discarded over one missing section.
# --------------------------------------------------------------------------
PAPER_VALIDATION_MIN_WORDS = 1000  # minimum extracted-text word count (summed across every
                                    # extracted section) for a paper to be marked VALID
PAPER_VALIDATION_MIN_UNIQUE_WORDS = 100  # minimum DISTINCT word count - catches genuinely
                                          # degenerate extractions (e.g. a PDF-parsing failure
                                          # that yields the same word/symbol repeated thousands
                                          # of times, which would otherwise clear the word-count
                                          # floor above despite containing no real content). A
                                          # real paper's abstract alone clears this by a wide
                                          # margin, so this never rejects genuine papers.
PAPER_VALIDATION_MIN_ABSTRACT_WORDS = 80  # minimum word count for abstract-only fallbacks
PAPER_VALIDATION_MIN_ABSTRACT_UNIQUE_WORDS = 30
