"""
Dense semantic retrieval using BAAI/bge-large-en-v1.5.

Instead of matching keywords, we embed the user's query and every
candidate paper's title+abstract into the same vector space and rank
by cosine similarity - so a query like "Agentic AI for Cloud Resource
Allocation" can surface a paper titled "Autonomous multi-agent systems
for elastic datacenter scheduling" even though they share almost no
words.
"""

from typing import List, Tuple

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import HashingVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config
from utils import Paper


class _OfflineHashingEmbeddingModel:
    """Offline fallback used when no SentenceTransformer model is cached.

    It keeps the app runnable without network access by mapping texts into a
    fixed hashing vector space. The vectors are not as strong as a true
    sentence model, but they preserve the ranking / clustering pipeline.
    """

    def __init__(self) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=4096,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):  # noqa: D401, ARG002
        vectors = self.vectorizer.transform(texts).astype(np.float32)
        return vectors.toarray()


def _load_sentence_transformer(model_name: str, local_only: bool = False) -> SentenceTransformer:
    return SentenceTransformer(model_name, local_files_only=local_only)


@st.cache_resource(show_spinner="Loading embedding model (BAAI/bge-large-en-v1.5)...")
def get_embedding_model():
    """Load once per Streamlit session/process - this model is ~1.3GB and
    should never be reloaded on every rerun of the script.
    First tries local cache (fast), then downloads from HuggingFace if not cached,
    finally falls back to a lightweight hashing model if all else fails."""
    # 1. Try local cache first (fastest, works offline)
    try:
        return _load_sentence_transformer(config.EMBEDDING_MODEL_NAME, local_only=True)
    except Exception:
        pass
    # 2. Try downloading from HuggingFace (first-time setup)
    try:
        return _load_sentence_transformer(config.EMBEDDING_MODEL_NAME, local_only=False)
    except Exception:  # noqa: BLE001 - offline fallback must keep the app running
        return _OfflineHashingEmbeddingModel()


def embed_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    """Encode a batch of texts into L2-normalized embedding vectors.

    BGE models expect a retrieval-instruction prefix on the *query* side
    only; document/passage texts are embedded as-is.
    """
    model = get_embedding_model()
    if is_query:
        texts = [config.BGE_QUERY_INSTRUCTION + t for t in texts]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def rank_by_similarity(query: str, papers: List[Paper]) -> List[Paper]:
    """Embed the query and every paper's title+abstract, score each paper
    by cosine similarity to the query, and return papers sorted best-first.
    """
    if not papers:
        return []

    query_vec = embed_texts([query], is_query=True)  # shape (1, dim)
    paper_texts = [p.text_for_embedding() for p in papers]
    paper_vecs = embed_texts(paper_texts, is_query=False)  # shape (N, dim)

    similarities = cosine_similarity(paper_vecs, query_vec).flatten()  # shape (N,)

    for paper, score in zip(papers, similarities):
        paper.similarity_score = float(score)

    return sorted(papers, key=lambda p: p.similarity_score, reverse=True)


def cluster_texts(
    items: List[Tuple[str, str]], similarity_threshold: float,
) -> List[List[Tuple[str, str]]]:
    """Greedy semantic clustering shared by any module that needs to merge
    near-duplicate short texts (research gaps, future-work themes,
    research problems, ...) without exact string matching.

    items = [(label, text), ...]. Each text joins the first existing
    cluster whose running centroid (mean of all members added so far,
    re-normalized) is within `similarity_threshold`, else starts a new
    cluster. A running centroid is more stable than comparing only to a
    cluster's first member, which drifts as soon as a borderline item
    joins.
    """
    items = [(label, text) for label, text in items if text and text.strip()]
    if not items:
        return []

    vecs = embed_texts([text for _, text in items], is_query=False)

    cluster_member_idxs: List[List[int]] = []
    cluster_sums: List[np.ndarray] = []
    for i in range(len(items)):
        placed = False
        for c_idx in range(len(cluster_member_idxs)):
            centroid = cluster_sums[c_idx] / len(cluster_member_idxs[c_idx])
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            if float(np.dot(vecs[i], centroid)) >= similarity_threshold:
                cluster_member_idxs[c_idx].append(i)
                cluster_sums[c_idx] = cluster_sums[c_idx] + vecs[i]
                placed = True
                break
        if not placed:
            cluster_member_idxs.append([i])
            cluster_sums.append(vecs[i].copy())

    return [[items[i] for i in idxs] for idxs in cluster_member_idxs]
