"""
Full-PDF fetch + text extraction + structural section split.

Low-level layer consumed by rag_pipeline.py (chunking/embedding/FAISS
retrieval) - this module only acquires and cleans text, it does not decide
what gets sent to an LLM.

Uses PyMuPDF (fitz) to extract the COMPLETE text of every page (no page
cap) - academic PDFs vary wildly in layout (single/multi-column, running
headers/footers, figures interleaved with text), so this makes a best-
effort pass to split the extracted text into named sections (Abstract,
Introduction, Related Work, Methodology, Dataset, Experimental Setup,
Results, Discussion, Limitations, Future Work, Conclusion) by detecting
common heading patterns, and always drops References/Acknowledgements
before returning sections.

If no headings are confidently detected, the whole cleaned text is kept
as a single "Body" section rather than silently discarded - a PDF that
downloads and parses successfully always yields usable section text,
whether or not its structure could be recovered.

get_full_paper_text() is the entry point every caller should use (never
raw fetch_pdf_sections() + hand-rolled cleaning): it fetches, cleans, and
caches the complete extracted text to data/papers/<paper_id>.txt so a
paper is never re-downloaded/re-parsed on a later search, and prints a
verification line (title, char/word counts, first 1000 chars) so it's
always visible that the FULL paper - not just the abstract - was read.
A paper whose PDF can't be downloaded or parsed is logged as an ERROR and
returns {} so the caller can fall back to the abstract-only path; this
never raises and never stops the rest of the batch.
"""

import hashlib
import io
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import requests

import config
from utils import Paper, normalize_doi, normalize_title

logger = logging.getLogger(__name__)

PAPERS_DIR = Path(__file__).resolve().parent / "data" / "papers"

# Canonical section name -> heading text variants that map to it, matched
# case-insensitively against a standalone (optionally numbered) line.
# Deliberately coarse-grained: equivalent headings from very different
# paper conventions (physics proposals, EE/networking papers, ML theory
# papers, CS/AI papers) all fold onto the same canonical bucket, so a
# paper using "Experimental Setup" instead of "Methodology", or
# "Discussion" instead of "Conclusion", isn't treated as if that content
# were missing.
SECTION_ALIASES: Dict[str, List[str]] = {
    "Abstract": ["abstract"],
    "Introduction": ["introduction", "background"],
    "Related Work": ["related work", "related works", "literature review",
                      "prior work", "background and related work"],
    "Methodology": [
        "methodology", "method", "methods", "materials and methods",
        "proposed method", "proposed approach", "approach",
        "system design", "model architecture", "system architecture",
        "system model", "network model", "problem formulation",
        "problem statement", "algorithm", "proposed system", "framework",
        "experimental setup", "experimental design", "implementation",
    ],
    "Dataset": ["dataset", "datasets", "data description", "data collection"],
    "Results": ["results", "results and discussion", "experimental results", "evaluation",
                "results and analysis", "performance evaluation", "simulation results",
                "numerical results", "performance analysis", "numerical evaluation",
                "simulation", "simulation and results", "experiments"],
    "Limitations": ["limitations", "threats to validity"],
    "Conclusion": ["conclusion", "conclusions", "summary",
                   "conclusion and future work", "conclusions and future work",
                   "conclusion and future works", "summary and conclusion",
                   "discussion", "discussion and analysis"],
    "Future Work": ["future work", "future directions", "future research",
                     "future scope", "open challenges", "open problems"],
    "References": ["references", "bibliography"],
    "Acknowledgements": ["acknowledgements", "acknowledgments", "acknowledgement"],
}

_IGNORED_SECTIONS = {"References", "Acknowledgements"}

_ALIAS_TO_CANONICAL: Dict[str, str] = {
    alias: canonical
    for canonical, aliases in SECTION_ALIASES.items()
    for alias in aliases
}

# A heading line: optionally a roman/arabic numeral prefix, then 2-6 words
# of title-ish text, nothing else on the line.
_HEADING_LINE_RE = re.compile(
    r"^\s*(?:[IVXLC]+\.|\d+(?:\.\d+)*\.?)?\s*([A-Za-z][A-Za-z \-]{2,45})\s*$"
)

# IEEE-style run-in headings, e.g. "Abstract—This article presents...",
# "Abstract: We propose...", "Abstract. We propose..." - heading and the
# section's first sentence sharing one physical line instead of the
# heading standing alone. Extremely common in two-column IEEE-formatted
# papers; without this, _HEADING_LINE_RE's "nothing else on the line"
# requirement silently misses the entire section.
_RUN_IN_HEADING_RE = re.compile(r"^\s*([A-Za-z][A-Za-z \-]{1,30}?)\s*[\-—:.]\s*(\S.*)$")


def _canonical_section(line: str) -> Optional[str]:
    match = _HEADING_LINE_RE.match(line.strip())
    if not match:
        return None
    return _ALIAS_TO_CANONICAL.get(match.group(1).strip().lower())


def _run_in_heading(line: str) -> Optional[Tuple[str, str]]:
    """Returns (canonical_section, remainder_text_on_same_line) for a
    run-in heading, else None. Gated on the candidate actually being a
    known alias, so this doesn't misfire on ordinary sentences that
    happen to contain a colon/dash/period."""
    match = _RUN_IN_HEADING_RE.match(line.strip())
    if not match:
        return None
    candidate, remainder = match.group(1).strip(), match.group(2).strip()
    canonical = _ALIAS_TO_CANONICAL.get(candidate.lower())
    if not canonical:
        return None
    return canonical, remainder


_ABSTRACT_FALLBACK_MIN_WORDS = 40
_ABSTRACT_FALLBACK_MAX_WORDS = 250
_ABSTRACT_FALLBACK_MIN_WORDS_PER_SENTENCE = 15


def _guess_abstract_lines(body_lines: List[str]) -> Optional[Tuple[int, int]]:
    """Heuristic fallback for papers with no detectable "Abstract" heading
    anywhere in the PDF body - confirmed on real papers (older preprints,
    institutional proposal documents) that never use the word "Abstract"
    at all, just launch straight from the author/affiliation block into
    prose. Finds the first substantial paragraph in the pre-heading
    "Body" text, skipping short title/author/affiliation/email lines at
    the top. Returns the (start, end) line-index range to lift out as the
    guessed Abstract, or None if nothing plausible was found.

    A deliberate precision/coverage tradeoff, chosen over leaving these
    papers permanently excluded: this can occasionally pick up a stray
    non-abstract paragraph instead of the true one, which is why it only
    ever runs when heading-based detection found nothing at all."""
    start = None
    for i, line in enumerate(body_lines):
        stripped = line.strip()
        if not stripped or "@" in stripped:
            continue
        if len(stripped.split()) > 6:
            start = i
            break
    if start is None:
        return None

    end = start
    total_words = 0
    for i in range(start, len(body_lines)):
        stripped = body_lines[i].strip()
        if not stripped:
            if total_words > 0:
                break
            continue
        total_words += len(stripped.split())
        end = i + 1
        if total_words >= _ABSTRACT_FALLBACK_MAX_WORDS:
            break

    candidate = " ".join(l.strip() for l in body_lines[start:end] if l.strip())
    word_count = len(candidate.split())
    if word_count < _ABSTRACT_FALLBACK_MIN_WORDS:
        return None
    sentence_markers = len(re.findall(r"[.!?]", candidate))
    if sentence_markers < 2:
        return None
    # Author/affiliation blocks ("B. Fastrup, E. Pedersen University of
    # Aarhus, Institute of Physics...") are comma/period-dense but made of
    # short 2-5-word fragments - real abstract prose runs ~15-30 words per
    # sentence. Requiring a high average rejects name-list false positives
    # (confirmed on a real paper) without needing per-name pattern rules.
    if word_count / sentence_markers < _ABSTRACT_FALLBACK_MIN_WORDS_PER_SENTENCE:
        return None
    return start, end


def _split_into_sections(raw_text: str) -> Dict[str, str]:
    """Best-effort heading-based split. Anything before the first
    recognized heading (or all of it, if none are found) goes into
    "Body" rather than being dropped."""
    current = "Body"
    buckets: Dict[str, List[str]] = {current: []}

    for line in raw_text.splitlines():
        heading = _canonical_section(line)
        if heading:
            current = heading
            buckets.setdefault(current, [])
            continue

        run_in = _run_in_heading(line)
        if run_in:
            current, remainder = run_in
            buckets.setdefault(current, [])
            if remainder:
                buckets[current].append(remainder)
            continue

        buckets.setdefault(current, []).append(line)

    if "Abstract" not in buckets and "Body" in buckets:
        guess = _guess_abstract_lines(buckets["Body"])
        if guess:
            start, end = guess
            buckets["Abstract"] = buckets["Body"][start:end]
            buckets["Body"] = buckets["Body"][:start] + buckets["Body"][end:]

    return {
        name: text
        for name, chunks in buckets.items()
        if (text := "\n".join(chunks).strip())
    }


# --------------------------------------------------------------------------
# Cleaning - strip page numbers, repeated running headers/footers, collapse
# duplicate whitespace. References/Acknowledgements are already excluded
# by _split_into_sections()/fetch_pdf_sections() before this ever runs.
# --------------------------------------------------------------------------

_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$")


def clean_text(text: str) -> str:
    """Strip page numbers (standalone digit lines), repeated running
    headers/footers (short lines that recur 3+ times verbatim across the
    document - journal names, copyright/ISSN boilerplate, etc.), and
    collapse duplicate whitespace."""
    if not text:
        return ""

    from collections import Counter

    lines = text.split("\n")
    short_line_counts = Counter(
        stripped for line in lines
        if (stripped := line.strip()) and len(stripped.split()) <= 12
    )
    boilerplate = {line for line, count in short_line_counts.items() if count >= 3}

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if _PAGE_NUMBER_LINE_RE.match(stripped) or stripped in boilerplate:
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# PDF fetch + parse (PyMuPDF - fitz)
# --------------------------------------------------------------------------

def fetch_pdf_sections(pdf_url: str) -> Dict[str, str]:
    """Download + parse the COMPLETE PDF (every page, no cap) into named
    sections (References/Acknowledgements already dropped). Returns {} on
    any failure (network error, non-PDF response, unparsable content) so
    the caller can fall back to the abstract."""
    try:
        resp = requests.get(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0 (knowledge-extraction)"},
            timeout=config.PDF_FETCH_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
    except Exception as exc:  # noqa: BLE001 - network failures must never break analysis
        logger.error("PDF download failed for %s: %s", pdf_url, exc)
        return {}

    content_type = resp.headers.get("Content-Type", "")
    if resp.status_code != 200 or (b"%PDF" not in resp.content[:1024] and "pdf" not in content_type.lower()):
        logger.error(
            "PDF download did not return a PDF for %s (status %s, content-type %r)",
            pdf_url, resp.status_code, content_type,
        )
        return {}

    try:
        doc = fitz.open(stream=io.BytesIO(resp.content).read(), filetype="pdf")
        try:
            raw_text = "\n".join(page.get_text() or "" for page in doc)
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001 - a corrupt/unusual PDF must never break analysis
        logger.error("PDF parse failed for %s: %s", pdf_url, exc)
        return {}

    if not raw_text.strip():
        logger.error("PDF parsed to empty text for %s", pdf_url)
        return {}

    sections = _split_into_sections(raw_text)
    return {name: text for name, text in sections.items() if name not in _IGNORED_SECTIONS}


# --------------------------------------------------------------------------
# Full-text acquisition + on-disk cache - the entry point every caller
# (rag_pipeline.py) should use.
# --------------------------------------------------------------------------

def paper_id_for(paper: Paper) -> str:
    """Stable, filesystem-safe identifier for a paper - the DOI when
    available (most stable across sources/reruns), else the normalized
    title. Used both as the cache filename and as the id printed/logged."""
    key = normalize_doi(paper.doi) or normalize_title(paper.title) or paper.title
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _sections_to_document(sections: Dict[str, str]) -> str: 
    """Serializes a sections dict back into a flat document whose headings
    round-trip through _split_into_sections() unchanged, so a cached file
    can be reloaded with the exact same parser used for a fresh PDF - no
    separate cache format needed. "Body" (the catch-all bucket used when
    no headings were detected) is written with no heading line, matching
    _split_into_sections()'s own default starting bucket."""
    parts = []
    for name, text in sections.items():
        if name != "Body":
            parts.append(name)
        parts.append(text)
    return "\n\n".join(parts)


def _safe_print(text: str) -> None:
    """print() that never raises over a character the console's codepage
    can't encode (Windows' default cp1252 in particular chokes on the
    math symbols/footnote marks PDFs routinely contain) - unencodable
    characters are replaced rather than crashing the extraction."""
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def _print_extraction_summary(title: str, full_text: str) -> None:
    """Verification output (per the spec: title, total chars, total words,
    first 1000 chars) so it's always visible that the COMPLETE paper - not
    just the abstract - was read, before this text ever reaches an LLM.
    PDFs routinely contain Unicode (math symbols, footnote marks) the
    Windows console's default codepage can't encode - _safe_print() avoids
    crashing the whole extraction over a print statement."""
    _safe_print(f"[PDF Extraction] {title}")
    _safe_print(f"  Total characters: {len(full_text)}")
    _safe_print(f"  Total words: {len(full_text.split())}")
    _safe_print(f"  First 1000 characters:\n{full_text[:1000]}")


def _abstract_fallback(paper: Paper) -> Dict[str, str]:
    """Returns the paper's abstract as a minimal {section: text} dict when
    the full PDF is unavailable. Allows papers without PDFs to still pass
    validation and contribute to AI analysis via their abstract text."""
    abstract = getattr(paper, "abstract", None) or ""
    abstract = abstract.strip()
    if abstract:
        logger.info("Using abstract fallback for '%s' (no PDF available).", paper.title[:60])
        return {"Abstract": abstract}
    logger.info("No abstract and no PDF for '%s' - will be rejected.", paper.title[:60])
    return {}


def get_full_paper_text(paper: Paper) -> Dict[str, str]:
    """Returns {section_name: cleaned_text} for the paper's COMPLETE PDF
    content - the only function callers should use to get a paper's full
    text (never fetch_pdf_sections() + hand-rolled cleaning directly).

    Checks data/papers/<paper_id>.txt first (skips re-downloading/re-
    parsing a paper already fetched in a previous search); on a cache
    miss, downloads and parses the PDF, cleans every section, prints the
    verification summary, and writes the cache file before returning.

    Falls back to the paper's abstract when there is no pdf_url, the
    download fails, or the PDF yields no usable text — so papers without
    PDFs still pass validation and contribute to AI analysis via their
    abstract. Returns {} (never raises) only when both the PDF AND the
    abstract are unavailable."""
    paper_id = paper_id_for(paper)
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PAPERS_DIR / f"{paper_id}.txt"

    # ── 1. Disk cache hit ────────────────────────────────────────────────
    if cache_path.exists():
        cached_document = cache_path.read_text(encoding="utf-8", errors="ignore")
        sections = _split_into_sections(cached_document)
        sections = {name: text for name, text in sections.items() if name not in _IGNORED_SECTIONS}
        if sections:
            full_text = "\n\n".join(sections.values())
            _print_extraction_summary(f"{paper.title} (cached)", full_text)
            return sections
        logger.warning("Cached text for %s was empty/unparsable - re-fetching.", paper.title[:60])

    # ── 2. No PDF URL → abstract fallback ───────────────────────────────
    if not paper.pdf_url:
        fallback = _abstract_fallback(paper)
        if fallback:
            try:
                cache_path.write_text(
                    _sections_to_document(fallback), encoding="utf-8"
                )
            except OSError:
                pass
        return fallback

    # ── 3. Download & parse PDF ──────────────────────────────────────────
    try:
        raw_sections = fetch_pdf_sections(paper.pdf_url)
    except Exception as exc:  # noqa: BLE001 - PDF issues must never break indexing
        logger.error("PDF extraction FAILED for %s: %s — trying abstract fallback.", paper.title[:60], exc)
        return _abstract_fallback(paper)

    sections = {name: clean_text(text) for name, text in raw_sections.items() if text}
    sections = {name: text for name, text in sections.items() if text}
    if not sections:
        logger.error("PDF extraction FAILED for %s: no usable text after cleaning — trying abstract fallback.", paper.title[:60])
        return _abstract_fallback(paper)

    full_text = "\n\n".join(sections.values())
    _print_extraction_summary(paper.title, full_text)

    try:
        cache_path.write_text(_sections_to_document(sections), encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 - caching is an optimization, never fatal
        logger.warning("Could not write PDF text cache for %s: %s", paper.title[:60], exc)

    return sections

