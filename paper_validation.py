"""
Paper Validation Agent.

Runs immediately after a paper's full-PDF text has been extracted and
cleaned (pdf_extraction.get_full_paper_text) and BEFORE any chunking,
embedding, or FAISS indexing - deliberately independent of rag_pipeline.py
(no PaperIndex/FAISS dependency at all; this module only knows about
Dict[str, str] section text) so a paper that's going to be rejected never
costs a single embedding call or FAISS build. Every downstream agent
(Knowledge Extraction, Problem-Solution Analysis, Contradiction Detection,
Root Cause Analysis, Inventor Agent) only ever sees papers that passed
validation here.

Deliberately lenient - a paper is marked INVALID (with every failing
reason recorded, not just the first) ONLY when:
    - The PDF couldn't be read at all (pdf_extraction.get_full_paper_text
      returned {} - no pdf_url, download failed, or PyMuPDF couldn't parse
      it into usable text).
    - The extracted text is shorter than config.PAPER_VALIDATION_MIN_WORDS
      (summed across every extracted section) - genuinely too little
      content to extract anything meaningful from.
    - The extracted text is degenerate - fewer than
      config.PAPER_VALIDATION_MIN_UNIQUE_WORDS distinct words, the
      signature of a broken extraction (e.g. a font/encoding issue that
      yields the same token repeated thousands of times) rather than real
      prose, even though it may clear the raw word-count floor above.

A paper is NEVER rejected for a missing section (Future Work, Conclusion,
Dataset, Results, or any other specific section) - pdf_extraction.
SECTION_ALIASES already folds equivalent headings from very different
paper conventions onto the same canonical bucket (e.g. "Experimental
Setup"/"Implementation" -> Methodology, "Discussion" -> Conclusion), and
whatever sections genuinely aren't present just mean Knowledge Extraction
reports "Not Found in Retrieved Context" for the specific fields with no
evidence - never a reason to discard the whole paper. `sections_found` is
still recorded on every result for transparency, it just no longer gates
validity.

A paper that fails any check is rejected before any chunking/embedding/
FAISS-indexing work ever runs for it; a paper that passes proceeds
unchanged to rag_pipeline.build_paper_index_from_sections(). One paper's
validation failure never stops the rest from being validated.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import config
from utils import Paper

logger = logging.getLogger(__name__)

VALID = "VALID"
INVALID = "INVALID"


@dataclass
class PaperValidationResult:
    paper_title: str
    status: str  # VALID / INVALID
    reasons: List[str] = field(default_factory=list)  # empty when VALID
    word_count: int = 0
    sections_found: List[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "Paper Title": self.paper_title,
            "Validation Status": self.status,
            "Failure Reason": "; ".join(self.reasons) if self.reasons else "Not Applicable",
            "Word Count": self.word_count,
            "Sections Found": ", ".join(self.sections_found) if self.sections_found else "None",
        }


def _validate_text(paper: Paper, sections: Dict[str, str]) -> PaperValidationResult:
    if not sections:
        return PaperValidationResult(
            paper_title=paper.title, status=INVALID,
            reasons=["No retrievable content - PDF unavailable, download failed, or could not be parsed."],
        )

    word_count = sum(len(text.split()) for text in sections.values())
    sections_found = sorted(sections.keys())
    unique_word_count = len({
        word.lower() for text in sections.values() for word in text.split() if word.isalpha()
    })

    reasons: List[str] = []
    if word_count < config.PAPER_VALIDATION_MIN_WORDS:
        reasons.append(
            f"Extracted text is only {word_count} words (minimum {config.PAPER_VALIDATION_MIN_WORDS} required)."
        )
    if unique_word_count < config.PAPER_VALIDATION_MIN_UNIQUE_WORDS:
        reasons.append(
            f"No meaningful technical content detected (only {unique_word_count} distinct words - "
            "looks like a degenerate/garbled extraction rather than real prose)."
        )

    return PaperValidationResult(
        paper_title=paper.title, status=INVALID if reasons else VALID, reasons=reasons,
        word_count=word_count, sections_found=sections_found,
    )


def validate_paper_text(paper: Paper, sections: Dict[str, str]) -> PaperValidationResult:
    """Validates one paper's already-extracted, already-cleaned full-text
    sections (pdf_extraction.get_full_paper_text's return value) - the
    ONLY input this module depends on, so it never touches rag_pipeline,
    embeddings, or FAISS. Never raises: any internal error is caught and
    reported as an INVALID result with the error as its reason, so one
    paper's validation problem can never stop the rest of a batch."""
    try:
        return _validate_text(paper, sections)
    except Exception as exc:  # noqa: BLE001 - one paper's validation must never break the rest
        logger.error("Validation failed for %s: %s", paper.title[:60], exc)
        return PaperValidationResult(paper_title=paper.title, status=INVALID, reasons=[f"Validation error: {exc}"])
