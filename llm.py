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

--------------------------------------------------------------------------
CENTRALIZED LLM MANAGER - this module is the ONLY place in the codebase
that talks to Groq. Every agent (Knowledge Extraction, Problem-Solution,
Contradiction Detection, Root Cause Analysis, Inventor Agent, Query
Understanding) calls _call_groq() below - none of them build an HTTP
request themselves. That single choke point is what makes the following
process-wide guarantees possible:

    Agents -> _call_groq() -> global lock/queue -> Groq API

1. GLOBAL SERIALIZATION: `_groq_call_lock` wraps every request (including
   any wait/retry inside it) so at most one Groq request is ever in
   flight for the whole process, regardless of how many agents/threads
   call in concurrently (e.g. Knowledge Extraction's ThreadPoolExecutor).
2. TPM-AWARE THROTTLING: a proactive wait based on Groq's own
   `x-ratelimit-remaining/reset-tokens` response headers, PLUS exponential
   backoff with jitter as a safeguard on top when a 429 still happens.
3. TPD (daily quota) is a DIFFERENT failure mode from TPM and is handled
   differently: it is detected from Groq's error message text, is NEVER
   retried (waiting cannot fix a quota that resets once a day), and
   raises GroqDailyQuotaExceeded so callers can stop gracefully instead
   of burning further retries/requests.
4. A local, persistent (survives process restarts), CROSS-PROCESS-SAFE
   running total of tokens used today is reserved BEFORE every request
   against config.GROQ_DAILY_SAFE_BUDGET (via an OS-level file lock on a
   sidecar .lock file, not just an in-process threading.Lock - two
   separate `python`/`streamlit run` processes sharing this file cannot
   both slip past the budget), then reconciled to the real total_tokens
   once the response is known, so this app never intentionally spends
   the account down to the last token itself.
5. Every successful, validly-parsed response is cached to disk
   (llm_cache.py) by (agent, paper/batch, prompt version, exact request
   content) - an identical request (e.g. from a Streamlit rerun) never
   re-hits Groq at all.
--------------------------------------------------------------------------
"""

import json
import logging
import re
import sys
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from random import uniform as _random_uniform
from typing import Optional

import requests

import config
import llm_cache

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


class GroqDailyQuotaExceeded(Exception):
    """Raised when this account's daily (TPD) Groq token quota is at or
    near its limit - either because our own tracked running total would
    cross config.GROQ_DAILY_SAFE_BUDGET, or because Groq's own error
    response says so directly. Unlike a per-minute (TPM) 429, this is
    NEVER retried automatically - a day-scoped quota does not refill
    within a pipeline run, so retrying only burns more of a budget that
    is already gone. Callers should catch this at their agent's outermost
    entry point, log it, and return whatever result they can build from
    already-completed/cached work instead of letting it crash the run."""


_TPD_MARKER_RE = re.compile(r"tokens per day\s*\(TPD\)", re.IGNORECASE)
_TPD_USAGE_RE = re.compile(
    r"tokens per day\s*\(TPD\):\s*Limit\s*(\d+),\s*Used\s*(\d+),\s*Requested\s*(\d+)", re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Daily (TPD) usage tracking - persisted to disk (survives process
# restarts/Streamlit reruns), reset when the UTC date rolls over. This is
# OUR OWN best-effort running total (Groq doesn't expose a "remaining
# TPD" response header the way it does for TPM - the day-scope limit only
# shows up in a 429's error body once already exceeded), reconciled
# against Groq's own authoritative "Used" figure whenever a TPD 429
# actually happens, so drift (e.g. from another process sharing the same
# key) self-corrects instead of accumulating.
#
# CROSS-PROCESS SAFETY: a plain threading.Lock only protects threads
# INSIDE one process - it does nothing for two separate `python`/
# `streamlit run` processes sharing this same file, which is exactly the
# scenario that let real usage reach 199,577 before the 170,000 soft
# budget stopped anything (each process's own read-check-write raced the
# other's). _FileLock below is a real OS-level advisory lock (msvcrt on
# Windows, flock on POSIX) on a tiny sidecar .lock file, held ONLY for
# the brief read-check-reserve-write below - never across the Groq
# network call itself, so unrelated work (PDF fetch, chunking, embedding,
# FAISS, other agents' retrieval) is never serialized by this.
#
# RESERVE -> COMMIT/RELEASE pattern (the actual fix for the race the
# audit describes): reserve_daily_tokens() atomically adds the ESTIMATE
# to the shared total and returns whether that stayed within budget,
# before any network call is made - so two processes checking at nearly
# the same instant can never both proceed past the limit, since the
# first one to acquire the file lock immediately "spends" its estimate
# for the other to see. After the real call completes, finalize_reservation()
# corrects the estimate to the real total_tokens Groq reports; if the
# call never actually got billed (network error, TPM retries exhausted),
# release_reservation() gives the estimate back.
# --------------------------------------------------------------------------
_USAGE_STATE_PATH = Path(__file__).resolve().parent / "data" / "groq_daily_usage.json"
_USAGE_LOCK_PATH = Path(__file__).resolve().parent / "data" / "groq_daily_usage.lock"
_daily_quota_exhausted = threading.Event()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class _FileLock:
    """Cross-process advisory lock on a small sidecar file. Blocks (with
    a short retry loop, never msvcrt's own hard 10s-then-error ceiling)
    until acquired. Used only around the few lines of read-check-write
    below - deliberately NOT held across any network I/O."""

    def __init__(self, path: Path):
        self._path = path
        self._fh = None

    def __enter__(self) -> "_FileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a+b")
        while True:
            try:
                if sys.platform == "win32":
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                time.sleep(0.02)

    def __exit__(self, *exc_info) -> None:
        try:
            if sys.platform == "win32":
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()


def _read_usage_state_unlocked() -> dict:
    """Always reads fresh from disk - no in-memory caching across calls.
    MUST be called only while holding _FileLock for any check-then-write
    sequence; safe to call standalone (e.g. get_used_today()) for a
    read-only snapshot, at the cost of a possible race with a concurrent
    writer that a caller doing only a read never needed atomicity for
    anyway."""
    try:
        loaded = json.loads(_USAGE_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        loaded = {}
    if loaded.get("date") != _today_utc():
        loaded = {"date": _today_utc(), "used_tokens": 0}
        _daily_quota_exhausted.clear()  # new day - budget is fresh again
    return loaded


def _write_usage_state_unlocked(state: dict) -> None:
    try:
        _USAGE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _USAGE_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 - usage tracking must never break the pipeline
        logger.warning("Failed to persist Groq daily usage state: %s", exc)


def get_used_today() -> int:
    return _read_usage_state_unlocked().get("used_tokens", 0)


def reserve_daily_tokens(estimated_tokens: int) -> bool:
    """Atomically (cross-process) checks whether `estimated_tokens` more
    would stay within config.GROQ_DAILY_SAFE_BUDGET and, if so, RESERVES
    it immediately (adds it to the shared total before any network call
    happens) - the actual fix for two processes both reading a safe-
    looking total and both proceeding. Returns False (reserves nothing)
    if the budget would be exceeded."""
    with _FileLock(_USAGE_LOCK_PATH):
        state = _read_usage_state_unlocked()
        used = state.get("used_tokens", 0)
        if used + estimated_tokens > config.GROQ_DAILY_SAFE_BUDGET:
            return False
        state["used_tokens"] = used + estimated_tokens
        _write_usage_state_unlocked(state)
        return True


def finalize_reservation(estimated_tokens: int, actual_tokens: int) -> None:
    """Replace a prior reservation's ESTIMATE with the REAL total_tokens
    Groq reported for that call, once it's known."""
    with _FileLock(_USAGE_LOCK_PATH):
        state = _read_usage_state_unlocked()
        state["used_tokens"] = max(0, state.get("used_tokens", 0) - estimated_tokens + actual_tokens)
        _write_usage_state_unlocked(state)


def release_reservation(estimated_tokens: int) -> None:
    """Give back a reservation for a call that never actually got billed
    (network error, or TPM retries exhausted with no successful
    response) - never called for a TPD rejection, which instead
    reconciles to Groq's own authoritative "Used" figure below."""
    with _FileLock(_USAGE_LOCK_PATH):
        state = _read_usage_state_unlocked()
        state["used_tokens"] = max(0, state.get("used_tokens", 0) - estimated_tokens)
        _write_usage_state_unlocked(state)


def _reconcile_daily_usage(groq_reported_used: int) -> None:
    """Trust Groq's own "Used" figure from a TPD 429 body over our local
    running total whenever it's higher - our tracker can only see this
    process's own calls, Groq sees everything against the key."""
    with _FileLock(_USAGE_LOCK_PATH):
        state = _read_usage_state_unlocked()
        if groq_reported_used > state.get("used_tokens", 0):
            state["used_tokens"] = groq_reported_used
            _write_usage_state_unlocked(state)


def is_daily_quota_exhausted() -> bool:
    """True once this process has observed (or locally projected) that
    the daily Groq quota is exhausted for today - agents should check
    this alongside is_available() to skip even attempting further Groq
    calls for the rest of this run, without needing to catch an
    exception from every single call site."""
    return _daily_quota_exhausted.is_set()


def daily_budget_remaining() -> int:
    return max(0, config.GROQ_DAILY_SAFE_BUDGET - get_used_today())


# Groq's free tier enforces a fairly low tokens-per-minute budget, so
# concurrent calls (e.g. knowledge_analysis.py's per-paper extraction)
# can legitimately get 429'd even under normal use - retry a couple of
# times before giving up, honoring the server's own reset hint.
#
# CONFIRMED via live probe against this project's actual Groq key: the
# real per-minute budget is 8,000 tokens (`x-ratelimit-limit-tokens: 8000`
# in every response header) - NOT the 12,000 figure some call sites in
# this codebase were tuned against. Also confirmed live: a single
# Knowledge Extraction call alone can need ~7,300-9,100 tokens - i.e.
# close to (or momentarily over) the ENTIRE per-minute budget - so after
# one such call the window needs close to the full 60s to refill, not a
# partial wait. An earlier version of this cap (25s) waited less than
# Groq's own `x-ratelimit-reset-tokens` hint reported, which fired the
# retry back into a still-depleted budget and guaranteed a second 429 -
# confirmed live (a request still failed after "waiting 25.0s" when the
# real reset needed was ~40s+). 65s covers Groq's full 60s rolling window
# with a small safety margin.
_MAX_RATE_LIMIT_RETRIES = 4
_MAX_RETRY_DELAY_SECONDS = 65.0
_GROQ_DURATION_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?(?:([\d.]+)ms)?")

# --------------------------------------------------------------------------
# Proactive TPM throttle - shared across every thread in this process.
#
# The reactive 429-retry loop above only reacts AFTER a request has
# already been rejected; under concurrent calls (e.g. Knowledge
# Extraction's ThreadPoolExecutor firing several papers' Groq calls at
# once) multiple threads independently retry into the same still-nearly-
# empty budget and collide again. This tracks the most recently observed
# `x-ratelimit-remaining-tokens` / `x-ratelimit-reset-tokens` headers (Groq
# returns them on every response, success or failure) and makes every
# subsequent call - regardless of which thread or module issued it - wait
# up front if the shared budget clearly can't cover it, instead of firing
# and hoping.
_rate_state_lock = threading.Lock()
_last_remaining_tokens: Optional[int] = None
_last_reset_tokens_seconds: Optional[float] = None
_last_observed_monotonic: Optional[float] = None

# --------------------------------------------------------------------------
# GLOBAL REQUEST SERIALIZATION - requirement #1: agents -> _call_groq() ->
# this one lock -> Groq API. Held for the ENTIRE duration of one logical
# _call_groq() invocation (every retry/wait inside it included), so two
# threads can never have a Groq HTTP request in flight at the same time,
# no matter how many agents/ThreadPoolExecutors call in concurrently. This
# is strictly stronger than (and now makes largely redundant, but still
# harmless as a lighter-weight pre-check) the proactive TPM throttle below.
# --------------------------------------------------------------------------
_groq_call_lock = threading.Lock()


def _update_rate_state(resp: "requests.Response") -> None:
    remaining = resp.headers.get("x-ratelimit-remaining-tokens")
    reset = resp.headers.get("x-ratelimit-reset-tokens")
    if remaining is None:
        return
    try:
        remaining_int = int(float(remaining))
    except ValueError:
        return
    reset_seconds = _parse_groq_duration(reset) if reset else None
    global _last_remaining_tokens, _last_reset_tokens_seconds, _last_observed_monotonic
    with _rate_state_lock:
        _last_remaining_tokens = remaining_int
        _last_reset_tokens_seconds = reset_seconds
        _last_observed_monotonic = time.monotonic()


def _estimate_request_tokens(messages: list, max_tokens: int) -> int:
    """Cheap chars/4 estimate of prompt size plus the full completion
    reservation - conservative on purpose (an overestimate just waits a
    little longer, an underestimate risks another 429), good enough to
    decide whether to wait, not to bill accurately."""
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    return (prompt_chars // 4) + max_tokens


def _throttle_if_needed(estimated_tokens: int) -> None:
    with _rate_state_lock:
        remaining = _last_remaining_tokens
        reset_seconds = _last_reset_tokens_seconds
        observed_at = _last_observed_monotonic
    if remaining is None or observed_at is None:
        return
    elapsed = time.monotonic() - observed_at
    time_until_reset = max(0.0, (reset_seconds or 0.0) - elapsed)
    if remaining < estimated_tokens and time_until_reset > 0:
        wait = min(time_until_reset + 0.5, _MAX_RETRY_DELAY_SECONDS)
        logger.info(
            "Throttling Groq call: shared TPM budget has ~%d tokens left, this call needs ~%d - waiting %.1fs",
            remaining, estimated_tokens, wait,
        )
        time.sleep(wait)


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


def _header_retry_delay_seconds(resp: requests.Response) -> Optional[float]:
    """The precise wait Groq itself tells us we need, when it tells us -
    None if neither header/field is present, so the caller can fall back
    to pure exponential backoff instead of a made-up default."""
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
    return None


def _retry_delay_seconds(resp: requests.Response, attempt: int) -> float:
    """TPM-aware wait: prefer Groq's own reset hint (the actual required
    wait for THIS rolling window) when available; exponential backoff
    with jitter is the SAFEGUARD for when it isn't (or as extra spacing
    on repeated collisions) - never a substitute for the real hint when
    we have one, per the "use actual response headers whenever available"
    requirement."""
    header_delay = _header_retry_delay_seconds(resp)
    backoff = min(2.0 * (2 ** attempt), _MAX_RETRY_DELAY_SECONDS)  # 2s, 4s, 8s, 16s, ...
    base = header_delay if header_delay is not None else backoff
    jitter = _random_uniform(0, base * 0.2)  # up to +20%, decorrelates concurrent retries
    return min(base + jitter, _MAX_RETRY_DELAY_SECONDS)


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


def _validate_for_cache(content: str, json_mode: bool) -> bool:
    """Never cache a malformed/empty result as a successful response
    (requirement: caching must not launder a bad answer into a fake
    cache HIT later) - a non-JSON call just needs non-empty content
    (already guaranteed by the empty-content check above this call), a
    JSON-mode call must additionally actually parse."""
    if not content:
        return False
    if json_mode:
        return parse_json(content) is not None
    return True


def _call_groq(
    messages: list,
    max_tokens: int = 200,
    temperature: float = 0.3,
    json_mode: bool = False,
    reasoning_effort: Optional[str] = "low",
    agent_name: str = "unknown",
    paper_id: str = "",
    cache_version: str = "v1",
    use_cache: bool = True,
    rag_chunk_count: Optional[int] = None,
) -> Optional[str]:
    """The single centralized entry point for every Groq call in this
    codebase (see module docstring). reasoning_effort defaults to "low":
    every call site here is a structured-extraction/classification task
    (JSON fields, labels, short summaries), never open-ended chain-of-
    thought the caller actually wants to see - and gpt-oss models spend
    real completion-token budget on a hidden `reasoning` field before
    ever writing `content`. CONFIRMED via live probe: the same trivial
    prompt used 35 reasoning tokens by default vs 8 with
    reasoning_effort="low". Pass reasoning_effort=None to opt out.

    agent_name/paper_id/cache_version together determine both the log
    tag and the persistent cache key (llm_cache.py) - pass a stable
    agent_name for every call site so [LLM] logs and caching both work;
    paper_id is optional (blank for a cross-paper batched call, where
    the message content itself already encodes exactly which papers/
    chunks were involved). Bump cache_version for a given agent whenever
    you change that agent's prompt in a way that should invalidate old
    cached answers.

    rag_chunk_count is purely for the [LLM] log line (how many retrieved
    chunks actually made it into this request) - pass it when the caller
    knows it, omit it otherwise.

    Raises GroqDailyQuotaExceeded (never retried - see the class
    docstring) when the daily token budget is exhausted; every other
    failure mode (network error, TPM exhaustion after retries, invalid
    JSON, empty content) degrades to returning None, as before."""
    tag = f"[LLM] Agent={agent_name}"
    if not config.GROQ_API_KEY:
        return None

    cache_key = llm_cache.build_key(agent_name, paper_id, cache_version, messages) if use_cache else None
    if cache_key:
        cached = llm_cache.get(cache_key)
        if cached is not None:
            logger.info("%s | Cache HIT | key=%s...", tag, cache_key[:48])
            return cached
        logger.info("%s | Cache MISS", tag)

    if is_daily_quota_exhausted():
        logger.warning("%s | Daily quota already exhausted - skipping call entirely", tag)
        raise GroqDailyQuotaExceeded(f"Daily Groq token budget ({config.GROQ_DAILY_SAFE_BUDGET}) already exhausted today.")

    payload = {
        "model": config.GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    input_tokens_est = _estimate_request_tokens(messages, 0)
    estimated_tokens = input_tokens_est + max_tokens
    logger.info(
        "%s | RAG chunks=%s | Estimated input tokens=%d | Requested output tokens=%d | Total estimated tokens=%d",
        tag, rag_chunk_count if rag_chunk_count is not None else "n/a",
        input_tokens_est, max_tokens, estimated_tokens,
    )

    # RESERVE first (cross-process safe - see reserve_daily_tokens'
    # docstring): this is what actually stops two processes both reading
    # a safe-looking total and both proceeding, unlike a check that only
    # reads-then-later-writes. Every exit path below either finalizes
    # this reservation to the real token count (a response was actually
    # billed) or releases it (the call never got billed) - exactly once.
    if not reserve_daily_tokens(estimated_tokens):
        _daily_quota_exhausted.set()
        logger.warning(
            "%s | Daily quota exhausted - stopping (used=%d, safe_budget=%d, this call needs ~%d)",
            tag, get_used_today(), config.GROQ_DAILY_SAFE_BUDGET, estimated_tokens,
        )
        raise GroqDailyQuotaExceeded(
            f"Daily Groq token budget ({config.GROQ_DAILY_SAFE_BUDGET}) would be exceeded "
            f"(this call needs ~{estimated_tokens})."
        )

    # GLOBAL SERIALIZATION: the entire attempt (every wait/retry inside
    # it) runs under one process-wide lock - see the lock's own docstring.
    # This is separate from (and does not replace) the cross-process file
    # lock above: this one serializes the actual NETWORK CALL within this
    # process; that one protects the tiny on-disk accounting critical
    # section across ALL processes. Neither is held across the other's
    # concern - the file lock is released well before this point.
    with _groq_call_lock:
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            _throttle_if_needed(estimated_tokens)
            logger.info("%s | Groq request started (attempt %d/%d)", tag, attempt + 1, _MAX_RATE_LIMIT_RETRIES + 1)
            try:
                resp = requests.post(config.GROQ_API_URL, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT_SECONDS)
            except Exception as exc:  # noqa: BLE001 - LLM failures must never break the search
                logger.warning("%s | Groq call failed: %s", tag, exc)
                release_reservation(estimated_tokens)  # never sent - give the estimate back
                return None

            _update_rate_state(resp)

            if resp.status_code == 429:
                body_text = resp.text
                if _TPD_MARKER_RE.search(body_text):
                    # Daily quota - a fundamentally different failure than a
                    # per-minute 429: waiting cannot fix it within this run,
                    # so this is NEVER retried (requirement: never retry TPD).
                    # The request was REJECTED (not billed) - release our own
                    # estimate, then reconcile the shared total up to Groq's
                    # own authoritative "Used" figure (the real ground truth,
                    # covering every process/session against this key).
                    release_reservation(estimated_tokens)
                    match = _TPD_USAGE_RE.search(body_text)
                    if match:
                        _, groq_used, _ = (int(g) for g in match.groups())
                        _reconcile_daily_usage(groq_used)
                    _daily_quota_exhausted.set()
                    logger.warning("%s | Daily quota exhausted (Groq-reported) - stopping, no retry | %s",
                                   tag, body_text[:300])
                    raise GroqDailyQuotaExceeded(body_text[:300])

                if attempt < _MAX_RATE_LIMIT_RETRIES:
                    delay = _retry_delay_seconds(resp, attempt)
                    logger.warning("%s | TPM wait=%.1fs (attempt %d/%d)",
                                    tag, delay, attempt + 1, _MAX_RATE_LIMIT_RETRIES + 1)
                    time.sleep(delay)
                    continue
                logger.warning("%s | Exhausted all retries on TPM rate limit", tag)
                release_reservation(estimated_tokens)  # never billed - give the estimate back
                return None

            try:
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # noqa: BLE001 - LLM failures must never break the search
                # Include Groq's actual error body (not just the generic "400/413
                # Client Error" exception text) - this is what actually reveals
                # causes like "Request too large ... tokens per minute", which
                # the bare exception string never showed.
                body = resp.text[:500] if "resp" in locals() else ""
                logger.warning("%s | Groq call failed: %s | response body: %s", tag, exc, body)
                release_reservation(estimated_tokens)  # never billed - give the estimate back
                return None

            # A response was actually returned - Groq DID bill for it
            # (even the empty-content case below still consumed real
            # completion tokens on reasoning) - finalize to the ACTUAL
            # total_tokens Groq reports, never just the pre-request
            # estimate, per "audit token accounting: prefer actual usage
            # over estimation whenever the response provides it."
            usage = data.get("usage", {})
            total_tokens = usage.get("total_tokens") or estimated_tokens
            finalize_reservation(estimated_tokens, total_tokens)
            logger.info(
                "%s | Groq request completed | Tokens used=%d (prompt=%s, completion=%s) | Daily budget remaining=%d",
                tag, total_tokens, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                daily_budget_remaining(),
            )

            if not content:
                # A 200 OK with empty content is a REAL failure mode (not "no
                # data available") - almost always max_tokens exhausted by the
                # hidden reasoning trace before any content was written. Must
                # be logged and NEVER cached - silently returning "" here (as
                # this used to do) made every downstream caller's `if
                # content:` check treat it identically to "call never
                # happened", so this failure never showed up in any log.
                logger.warning(
                    "%s | EMPTY content (finish_reason=%s, reasoning_tokens=%s, completion_tokens=%s) "
                    "- max_tokens=%d was likely exhausted by hidden reasoning before any answer was written.",
                    tag, data["choices"][0].get("finish_reason"),
                    usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
                    usage.get("completion_tokens"), max_tokens,
                )
                return None

            if cache_key:
                if _validate_for_cache(content, json_mode):
                    llm_cache.set(cache_key, content)
                else:
                    logger.warning("%s | Response failed validation - NOT caching malformed output", tag)

            return content

    return None


@lru_cache(maxsize=512)
def summarize_paper(title: str, abstract: str) -> str:
    """Produce a short plain-language summary of a paper for display.
    Results are cached in-memory by (title, abstract) for this process
    (this lru_cache) AND persisted to disk via _call_groq's cache_key
    (survives process restarts/Streamlit reruns) — same paper never hits
    Groq twice either way."""
    if not abstract:
        return "No abstract available from the source API."

    try:
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
            agent_name="PaperSummary",
            paper_id=title,
        )
    except GroqDailyQuotaExceeded:
        content = None
    if content:
        return content
    # Fallback with no LLM available (or daily quota exhausted): trim the raw abstract.
    return abstract[:300] + ("..." if len(abstract) > 300 else "")


@lru_cache(maxsize=512)
def explain_relevance(query: str, title: str, abstract: str) -> str:
    """Explain in 1-2 sentences why this paper is relevant to the query.
    Results are cached in-memory by (query, title, abstract) for this
    process AND persisted to disk (survives restarts/reruns) — no
    duplicate Groq calls either way."""
    try:
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
            agent_name="PaperRelevance",
            paper_id=title,
        )
    except GroqDailyQuotaExceeded:
        content = None
    if content:
        return content
    return "AI relevance explanation unavailable (no LLM API key configured, or daily quota exhausted)."


def clear_llm_cache() -> None:
    """Clear both LRU caches — call between unrelated searches if memory is a concern."""
    summarize_paper.cache_clear()
    explain_relevance.cache_clear()

