"""
Retrieval-Augmented Generation pipeline.

Replaces "send the whole extracted PDF text to the LLM" everywhere in this
app with a real RAG flow:

    chunk_sections()   split each already-extracted, already-cleaned
                        section (pdf_extraction.get_full_paper_text's
                        output - PyMuPDF, complete PDF, cached to disk)
                        into ~900-token chunks with ~175-token overlap,
                        never splitting mid-sentence (so tables/equations/
                        headings stay intact - headings in particular are
                        never chunked at all, since they're the section
                        boundary, not content)
        -> embed_chunks()   BAAI/bge-small-en-v1.5 (falls back to
                            intfloat/e5-base-v2 if unavailable)
        -> FAISS IndexFlatIP, built fresh in memory per paper, per session -
           no persistent vector database
        -> retrieve_chunks()/retrieve_context()  task-specific semantic
           query -> top-K most relevant chunks only

This module deliberately has NO knowledge of how the text it chunks was
acquired (no pdf_extraction import) and NO knowledge of paper validation
(no paper_validation import) - build_paper_index_from_sections() takes an
already-extracted `sections` dict as a plain argument. The orchestration
layer (app.py) is responsible for calling, in order: pdf_extraction.
get_full_paper_text() -> paper_validation.validate_paper_text() -> (only
if VALID) build_paper_index_from_sections(). A paper that fails validation
is rejected before it ever reaches this module, so chunking/embedding/
FAISS-building - real, non-trivial compute - is never spent on a paper
that's going to be discarded.

One PaperIndex is built ONCE per paper (by app.py's collection loop) and
cached by paper title in a plain dict that every downstream agent
(Knowledge Extraction, Research Gap, Future Work, Problem-Solution,
Contradiction Detection, Root Cause Analysis, Inventor Agent) reuses for
its own retrieval query - embeddings are never recomputed per agent.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

import config
from utils import Paper

logger = logging.getLogger(__name__)

NOT_FOUND_IN_CONTEXT = "Not Found in Retrieved Context"


@dataclass
class Chunk:
    paper_id: str
    section_name: str
    chunk_id: str
    chunk_text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity_score: float


@dataclass
class PaperIndex:
    paper_id: str
    chunks: List[Chunk] = field(default_factory=list)
    vectors: Optional[np.ndarray] = None
    faiss_index: "faiss.Index" = None
    source_label: str = "Not Available"  # e.g. "RAG (14 chunks from Full PDF)",
                                          # "RAG (1 chunk from Abstract fallback)",
                                          # "Not Available"


# --------------------------------------------------------------------------
# Step 1: chunking - sentence-boundary packing, never splits mid-sentence
# (so equations/tables embedded in running text and section headings,
# which are never part of the chunked body text in the first place, stay
# intact), with token-based sizing via the embedding model's own tokenizer.
# --------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_oversized_sentence(tokenizer, sentence: str) -> List[str]:
    """Safety net for garbled PDF text with long unpunctuated runs (so no
    sentence boundary exists): if a single "sentence" alone exceeds the
    chunk budget, force-split it on word boundaries instead of letting it
    become one oversized chunk that the embedding model would silently
    truncate."""
    words = sentence.split(" ")
    pieces: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = _token_len(tokenizer, word)
        if current and current_tokens + word_tokens > config.RAG_CHUNK_SIZE_TOKENS:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _chunk_section(
    tokenizer, paper_id: str, section_name: str, text: str, start_index: int,
) -> Tuple[List[Chunk], int]:
    sentences: List[str] = []
    for sentence in _split_sentences(text):
        if _token_len(tokenizer, sentence) > config.RAG_CHUNK_SIZE_TOKENS:
            sentences.extend(_split_oversized_sentence(tokenizer, sentence))
        else:
            sentences.append(sentence)
    if not sentences:
        return [], start_index

    chunks: List[Chunk] = []
    current: List[str] = []
    current_tokens = 0
    idx = start_index

    def flush():
        nonlocal idx
        if not current:
            return
        chunk_text = " ".join(current)
        chunks.append(Chunk(
            paper_id=paper_id, section_name=section_name,
            chunk_id=f"{paper_id}::{section_name}::{idx}", chunk_text=chunk_text,
        ))
        idx += 1

    for sentence in sentences:
        sentence_tokens = _token_len(tokenizer, sentence)
        if current and current_tokens + sentence_tokens > config.RAG_CHUNK_SIZE_TOKENS:
            flush()
            # carry back trailing sentences worth ~overlap tokens into the next chunk
            overlap_sentences: List[str] = []
            overlap_tokens = 0
            for s in reversed(current):
                t = _token_len(tokenizer, s)
                if overlap_tokens + t > config.RAG_CHUNK_OVERLAP_TOKENS:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += t
            current = overlap_sentences
            current_tokens = overlap_tokens
        current.append(sentence)
        current_tokens += sentence_tokens

    flush()
    return chunks, idx


def chunk_sections(paper_id: str, sections: Dict[str, str]) -> List[Chunk]:
    """sections = {section_name: cleaned_text}. Chunks each section
    independently (chunks never span two sections) and returns the full
    flat list, ready to embed."""
    chunks: List[Chunk] = []
    idx = 0
    tokenizer = get_chunk_embedding_model()[0]
    for section_name, text in sections.items():
        section_chunks, idx = _chunk_section(tokenizer, paper_id, section_name, text, idx)
        chunks.extend(section_chunks)
    return chunks


# --------------------------------------------------------------------------
# Step 2: embedding generation - BAAI/bge-small-en-v1.5, falling back to
# intfloat/e5-base-v2 if the primary model can't be loaded.
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading chunk embedding model...")
def get_chunk_embedding_model():
    """Returns (tokenizer, model, model_name). Cached once per process -
    this is the "generate embeddings only once per session" requirement
    applied to the model itself, not just the per-paper chunk vectors."""
    try:
        model = SentenceTransformer(config.RAG_CHUNK_EMBEDDING_MODEL)
        name = config.RAG_CHUNK_EMBEDDING_MODEL
    except Exception as exc:  # noqa: BLE001 - must never crash the app over a model download
        logger.warning(
            "Failed to load %s (%s) - falling back to %s",
            config.RAG_CHUNK_EMBEDDING_MODEL, exc, config.RAG_CHUNK_EMBEDDING_MODEL_FALLBACK,
        )
        model = SentenceTransformer(config.RAG_CHUNK_EMBEDDING_MODEL_FALLBACK)
        name = config.RAG_CHUNK_EMBEDDING_MODEL_FALLBACK
    return model.tokenizer, model, name


def _embed(texts: List[str], is_query: bool) -> np.ndarray:
    """Applies the correct instruction/prefix convention for whichever
    model actually loaded (BGE vs E5 use different conventions), then
    encodes to L2-normalized float32 vectors (so inner product == cosine
    similarity for FAISS)."""
    _, model, name = get_chunk_embedding_model()
    if "e5" in name.lower():
        prefixed = [("query: " if is_query else "passage: ") + t for t in texts]
    else:  # BGE convention: instruction on the query side only
        prefixed = [config.BGE_QUERY_INSTRUCTION + t for t in texts] if is_query else texts
    vectors = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


# --------------------------------------------------------------------------
# Step 3: vector database (FAISS, in-memory, built fresh per paper/session)
# --------------------------------------------------------------------------

def _build_faiss_index(vectors: np.ndarray) -> "faiss.Index":
    index = faiss.IndexFlatIP(vectors.shape[1])  # exact inner-product search;
    index.add(vectors)                            # dataset is tens of chunks, no ANN needed
    return index


def build_paper_index_from_sections(paper: Paper, sections: Dict[str, str]) -> PaperIndex:
    """Chunk -> embed -> FAISS from an ALREADY-EXTRACTED, already-cleaned
    sections dict (pdf_extraction.get_full_paper_text's return value).
    Callers are expected to have already run paper_validation.
    validate_paper_text() on `sections` and only call this for a VALID
    result - this function does no validation itself and never falls back
    to the paper's abstract, since an invalid/textless paper should never
    reach here at all. "Not Available" only as a defensive fallback if
    chunking somehow still produces nothing for an already-validated
    paper."""
    paper_id = paper.title
    chunks = chunk_sections(paper_id, sections) if sections else []
    if not chunks:
        return PaperIndex(paper_id=paper_id, source_label="Not Available")

    vectors = _embed([c.chunk_text for c in chunks], is_query=False)
    faiss_index = _build_faiss_index(vectors)

    label = f"RAG ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''} from Full PDF)"
    return PaperIndex(paper_id=paper_id, chunks=chunks, vectors=vectors, faiss_index=faiss_index, source_label=label)


# --------------------------------------------------------------------------
# Step 4: semantic retrieval
# --------------------------------------------------------------------------

def retrieve_chunks(paper_index: PaperIndex, query: str, top_k: int) -> List[RetrievedChunk]:
    """Embeds `query` and searches this one paper's FAISS index - never
    recomputes chunk embeddings, only the (single, cheap) query vector."""
    if not paper_index.chunks or paper_index.faiss_index is None:
        return []

    query_vec = _embed([query], is_query=True)
    k = min(top_k, len(paper_index.chunks))
    scores, indices = paper_index.faiss_index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append(RetrievedChunk(chunk=paper_index.chunks[idx], similarity_score=float(score)))
    return results


def retrieve_context(
    paper_index: PaperIndex, queries: List[Tuple[str, int]], max_total_chunks: Optional[int] = None,
) -> Tuple[str, List[RetrievedChunk]]:
    """High-level helper for callers with more than one task-specific
    query against the same paper (e.g. Knowledge Extraction also pulling
    Research Gap + Future Work context for the same combined LLM call).

    queries = [(query_text, top_k), ...]. Retrieves each, deduplicates by
    chunk_id (keeping the highest score if a chunk matches more than one
    query), optionally caps the total chunk count, and assembles the
    survivors into one "[Section] chunk text" blob ready to drop into an
    LLM prompt.

    Returns ("", []) if nothing could be retrieved - callers must treat
    that as "no evidence available" (-> "Not Found in Retrieved Context"),
    never fall back to sending the whole document.
    """
    best_by_chunk_id: Dict[str, RetrievedChunk] = {}
    for query_text, top_k in queries:
        for rc in retrieve_chunks(paper_index, query_text, top_k):
            existing = best_by_chunk_id.get(rc.chunk.chunk_id)
            if existing is None or rc.similarity_score > existing.similarity_score:
                best_by_chunk_id[rc.chunk.chunk_id] = rc

    by_relevance = sorted(best_by_chunk_id.values(), key=lambda rc: rc.similarity_score, reverse=True)
    if max_total_chunks:
        by_relevance = by_relevance[:max_total_chunks]

    # Selection is relevance-ranked (above); presentation is document-order
    # (below) - the surviving chunks are reassembled in the order they
    # actually appear in the paper, not shuffled by score, so the LLM
    # reads evidence following the paper's own narrative flow.
    position = {chunk.chunk_id: i for i, chunk in enumerate(paper_index.chunks)}
    ranked = sorted(by_relevance, key=lambda rc: position.get(rc.chunk.chunk_id, 0))

    assembled = "\n\n".join(f"[{rc.chunk.section_name}] {rc.chunk.chunk_text}" for rc in ranked)
    return assembled, ranked
