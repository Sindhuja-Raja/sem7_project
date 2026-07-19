"""
LLM-powered stages of the pipeline, served by Groq (fast, free-tier
hosted inference for open models such as Llama 3.3 and Qwen 3):

- summarize_paper:    a short, plain-language summary of one paper.
- explain_relevance:  a one/two sentence explanation of why a specific
                      paper matches the user's query.

Query understanding (intent extraction + search query generation) lives
in query_understanding.py, which reuses _call_groq below.

If GROQ_API_KEY is not configured (or a call fails), every function
degrades gracefully to a sensible non-AI fallback instead of raising -
the rest of the app keeps working, just without the LLM enrichment.
"""

import json
import logging
import re
import time
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)

# Groq's free tier enforces a fairly low tokens-per-minute budget, so
# concurrent calls (e.g. knowledge_analysis.py's per-paper extraction)
# can legitimately get 429'd even under normal use - retry a couple of
# times before giving up, honoring the server's own reset hint.
_MAX_RATE_LIMIT_RETRIES = 3
_MAX_RETRY_DELAY_SECONDS = 15.0
_GROQ_DURATION_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?(?:([\d.]+)ms)?")


def _parse_groq_duration(text: str) -> Optional[float]:
    """Groq's x-ratelimit-reset-* headers look like '1h37m55.2s' or '229ms'."""
    match = _GROQ_DURATION_RE.fullmatch(text.strip())
    if not match:
        return None
    hours, minutes, secs, millis = match.groups()
    total = 0.0
    if hours:
        total += int(hours) * 3600
    if minutes:
        total += int(minutes) * 60
    if secs:
        total += float(secs)
    if millis:
        total += float(millis) / 1000
    return total if total > 0 else None


def _retry_delay_seconds(resp: requests.Response) -> float:
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    for header in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        parsed = _parse_groq_duration(resp.headers.get(header, ""))
        if parsed is not None:
            return parsed
    return 2.0


def is_available() -> bool:
    return bool(config.GROQ_API_KEY)


def parse_json(content: Optional[str]) -> Optional[object]:
    """Best-effort JSON parsing of an LLM response: try it as-is, then
    fall back to extracting the outermost {...}/[...] block in case the
    model wrapped it in markdown fences or a stray sentence."""
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _call_groq(
    messages: list,
    max_tokens: int = 200,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> Optional[str]:
    if not config.GROQ_API_KEY:
        return None

    payload = {
        "model": config.GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
        try:
            resp = requests.post(config.GROQ_API_URL, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001 - LLM failures must never break the search
            logger.warning("Groq call failed: %s", exc)
            return None

        if resp.status_code == 429 and attempt < _MAX_RATE_LIMIT_RETRIES:
            delay = min(_retry_delay_seconds(resp), _MAX_RETRY_DELAY_SECONDS)
            logger.warning("Groq rate-limited (attempt %d/%d) - retrying in %.1fs",
                            attempt + 1, _MAX_RATE_LIMIT_RETRIES + 1, delay)
            time.sleep(delay)
            continue

        try:
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001 - LLM failures must never break the search
            # Include Groq's actual error body (not just the generic "400/413
            # Client Error" exception text) - this is what actually reveals
            # causes like "Request too large ... tokens per minute", which
            # the bare exception string never showed.
            body = resp.text[:500] if "resp" in locals() else ""
            logger.warning("Groq call failed: %s | response body: %s", exc, body)
            return None

    return None


def summarize_paper(title: str, abstract: str) -> str:
    """Produce a short plain-language summary of a paper for display."""
    if not abstract:
        return "No abstract available from the source API."

    content = _call_groq(
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize academic papers in 2-3 plain-language "
                    "sentences for a researcher skimming search results. "
                    "Be concrete about the method and contribution. No preamble."
                ),
            },
            {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
        ],
        max_tokens=150,
        temperature=0.3,
    )
    if content:
        return content
    # Fallback with no LLM available: trim the raw abstract.
    return abstract[:300] + ("..." if len(abstract) > 300 else "")


def explain_relevance(query: str, title: str, abstract: str) -> str:
    """Explain in 1-2 sentences why this paper is relevant to the query."""
    content = _call_groq(
        messages=[
            {
                "role": "system",
                "content": (
                    "You explain in 1-2 sentences why a specific paper is "
                    "relevant to a researcher's search query. Reference the "
                    "concrete overlap between the query and the paper's "
                    "method/topic. No preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Search query: {query}\n\n"
                    f"Paper title: {title}\n\nPaper abstract: {abstract or '(no abstract available)'}"
                ),
            },
        ],
        max_tokens=100,
        temperature=0.3,
    )
    if content:
        return content
    return "AI relevance explanation unavailable (no LLM API key configured)."
