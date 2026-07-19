"""
Cross-encoder re-ranking using cross-encoder/ms-marco-MiniLM-L12-v2.

Bi-encoder cosine similarity (embedding.py) is fast but scores the
query and each paper independently. A cross-encoder instead reads the
query and paper text *together* in a single forward pass, which is
slower but produces a much sharper relevance judgement - so we only
run it on the top-K shortlist that survived the embedding stage.
"""

from typing import List

import streamlit as st
from sentence_transformers import CrossEncoder

import config
from utils import Paper


@st.cache_resource(show_spinner="Loading cross-encoder re-ranking model...")
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(config.CROSS_ENCODER_MODEL_NAME)


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
