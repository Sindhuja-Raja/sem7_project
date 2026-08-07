"""
Cross-encoder re-ranking using cross-encoder/ms-marco-MiniLM-L12-v2.

Bi-encoder cosine similarity (embedding.py) is fast but scores the
query and each paper independently. A cross-encoder instead reads the
query and paper text *together* in a single forward pass, which is
slower but produces a much sharper relevance judgement - so we only
run it on the top-K shortlist that survived the embedding stage.
"""

from typing import List

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import HashingVectorizer
from sentence_transformers import CrossEncoder

import config
from utils import Paper


class _OfflineCrossEncoder:
    """Deterministic fallback when the HF cross-encoder is unavailable."""

    def __init__(self) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=4096,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def predict(self, pairs):
        scores = []
        for query, paper_text in pairs:
            query_vec = self.vectorizer.transform([query]).toarray()[0]
            paper_vec = self.vectorizer.transform([paper_text]).toarray()[0]
            overlap = float(np.dot(query_vec, paper_vec))
            query_terms = {term for term in query.lower().split() if term}
            paper_terms = {term for term in paper_text.lower().split() if term}
            jaccard = len(query_terms & paper_terms) / max(len(query_terms | paper_terms), 1)
            scores.append(overlap + jaccard)
        return np.asarray(scores, dtype=np.float32)


@st.cache_resource(show_spinner="Loading cross-encoder re-ranking model...")
def get_cross_encoder():
    # 1. Try local cache first (fastest, works offline)
    try:
        return CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, local_files_only=True)
    except Exception:
        pass
    # 2. Try downloading from HuggingFace (first-time setup)
    try:
        return CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, local_files_only=False)
    except Exception:  # noqa: BLE001 - app must stay runnable without the model
        return _OfflineCrossEncoder()


def rerank_papers(query: str, papers: List[Paper]) -> List[Paper]:
    """Score each (query, paper) pair with the cross-encoder and return
    papers sorted best-first by that score."""
    if not papers:
        return []

    model = get_cross_encoder()
    pairs = [(query, p.text_for_embedding()) for p in papers]
    scores = model.predict(pairs)

    for paper, score in zip(papers, scores):
        paper.rerank_score = float(score)

    return sorted(papers, key=lambda p: p.rerank_score, reverse=True)
