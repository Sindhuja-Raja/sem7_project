"""
Shared data model + helper functions used across the pipeline:

- Paper: the common record every academic source is normalized into.
- Title/DOI normalization helpers.
- Duplicate detection and merging across sources.

No network calls and no AI models live here - this module is pure,
synchronous, dependency-light logic so it stays easy to unit test.
"""

import re
import difflib
from dataclasses import dataclass, field
from typing import List, Optional

from config import TITLE_SIMILARITY_DEDUP_THRESHOLD


@dataclass
class Paper:
    """Normalized representation of a single paper, regardless of which
    academic API it came from."""

    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    source: str = ""            # e.g. "OpenAlex" - becomes "OpenAlex + Crossref" after merging
    doi: Optional[str] = None
    abstract: str = ""
    pdf_url: Optional[str] = None

    # populated later by the ranking pipeline
    similarity_score: float = 0.0
    rerank_score: float = 0.0
    ai_summary: str = ""
    ai_reason: str = ""

    def authors_display(self) -> str:
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) > 5:
            return ", ".join(self.authors[:5]) + f" et al. ({len(self.authors)} authors)"
        return ", ".join(self.authors)

    def text_for_embedding(self) -> str:
        """Title + abstract concatenation used for embedding / cross-encoder input."""
        if self.abstract:
            return f"{self.title}. {self.abstract}"
        return self.title


# ------------------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation/whitespace so titles can be compared
    regardless of formatting differences between APIs."""
    if not title:
        return ""
    title = title.lower()
    title = _PUNCT_RE.sub(" ", title)
    title = _WHITESPACE_RE.sub(" ", title).strip()
    return title


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Strip URL prefixes / whitespace / casing so DOIs from different
    APIs (e.g. "https://doi.org/10.1145/X" vs "10.1145/X") compare equal."""
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = doi.strip().strip("/")
    return doi or None


def title_similarity(title_a: str, title_b: str) -> float:
    """Fuzzy similarity ratio in [0, 1] between two normalized titles."""
    a, b = normalize_title(title_a), normalize_title(title_b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _authors_overlap(a_list: List[str], b_list: List[str]) -> bool:
    """True if the two author lists share at least one surname."""
    if not a_list or not b_list:
        return False

    def surnames(authors):
        return {a.strip().lower().split()[-1] for a in authors if a.strip()}

    return len(surnames(a_list) & surnames(b_list)) > 0


# ------------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------------

def _is_duplicate(a: Paper, b: Paper) -> bool:
    """Decide whether two Paper records refer to the same underlying work,
    using DOI, title similarity, authors and publication year as signals."""

    # Strongest signal: identical DOI. A DOI *mismatch* is not strong
    # evidence against being duplicates though - a preprint (e.g. an SSRN
    # or arXiv DOI) and its later published journal version commonly have
    # two different DOIs for the same underlying paper - so a mismatch
    # falls through to the title-similarity check below instead of
    # short-circuiting to "not a duplicate".
    doi_a, doi_b = normalize_doi(a.doi), normalize_doi(b.doi)
    if doi_a and doi_b and doi_a == doi_b:
        return True

    # No matching DOI -> fall back to title similarity, corroborated by
    # publication year and author overlap to avoid false positives
    # between distinct papers that merely share a generic title.
    sim = title_similarity(a.title, b.title)
    if sim >= TITLE_SIMILARITY_DEDUP_THRESHOLD:
        same_year = (a.year is None or b.year is None or abs(a.year - b.year) <= 1)
        return same_year

    # Borderline title similarity: only treat as duplicate if authors
    # and year both corroborate it.
    if sim >= 0.75:
        same_year = (a.year is not None and b.year is not None and a.year == b.year)
        return same_year and _authors_overlap(a.authors, b.authors)

    return False


def _merge_papers(a: Paper, b: Paper) -> Paper:
    """Combine two duplicate records, keeping the most complete field
    values and recording that the paper was found in multiple sources."""
    merged = Paper(
        title=a.title if len(a.title) >= len(b.title) else b.title,
        authors=a.authors if len(a.authors) >= len(b.authors) else b.authors,
        year=a.year or b.year,
        source=", ".join(sorted(set(a.source.split(", ") + b.source.split(", ")))),
        doi=normalize_doi(a.doi) or normalize_doi(b.doi),
        abstract=a.abstract if len(a.abstract) >= len(b.abstract) else b.abstract,
        pdf_url=a.pdf_url or b.pdf_url,
    )
    return merged


def deduplicate_papers(papers: List[Paper]) -> List[Paper]:
    """Merge duplicate records retrieved from different academic sources
    into a single Paper per underlying work."""
    unique: List[Paper] = []

    for candidate in papers:
        if not candidate.title:
            continue

        merged_into_existing = False
        for i, existing in enumerate(unique):
            if _is_duplicate(candidate, existing):
                unique[i] = _merge_papers(existing, candidate)
                merged_into_existing = True
                break

        if not merged_into_existing:
            unique.append(candidate)

    return unique
