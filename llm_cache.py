"""
Persistent, disk-backed cache for LLM (Groq) responses.

Why this exists: Streamlit reruns the whole script on every interaction,
and a fresh process restart (or a second terminal running the app) starts
with an empty in-memory cache - neither is safe against a Groq account
with a hard daily token quota. A response cached here survives both, so
the SAME (paper, agent, prompt version, request content) never re-hits
Groq once it has already succeeded.

Backing store: one JSON file (data/llm_cache.json), loaded lazily and
kept in memory for the life of the process, written through on every
`set()`. This app's realistic cache size (tens to low hundreds of
entries, each a few KB of JSON text) makes a single file simpler and
more portable (no extra dependency, works identically on Windows) than
a real key-value store, while still being genuinely persistent.

Callers never build cache keys by hand - llm.py's _call_groq() derives
one automatically from (agent_name, paper_id, cache_version, messages),
so adding caching to a new call site is just passing agent_name.
"""

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).resolve().parent / "data" / "llm_cache.json"
_lock = threading.Lock()
_store: Optional[dict] = None  # lazily loaded, then kept in memory


def _load() -> dict:
    global _store
    if _store is not None:
        return _store
    try:
        _store = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(_store, dict):
            _store = {}
    except (OSError, json.JSONDecodeError):
        _store = {}
    return _store


def _save() -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(_store, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 - a cache write failure must never break the pipeline
        logger.warning("Failed to persist LLM cache to disk: %s", exc)


def build_key(agent_name: str, paper_id: str, cache_version: str, messages: list) -> str:
    """Deterministic key from what actually determines the response: which
    agent asked, which paper/batch (paper_id - empty string for a
    cross-paper batched call, which is fine since the message content
    itself already encodes exactly which papers/chunks were involved),
    the prompt version (bump this whenever a prompt changes meaning, to
    invalidate stale cached answers), and a hash of the exact message
    content (system + user prompts) - so a different retrieved-chunk set
    for the "same" paper (e.g. after a chunking/RAG-query change) never
    collides with an old cached answer."""
    raw = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    prefix = f"{paper_id}::" if paper_id else ""
    return f"{prefix}{agent_name}::{cache_version}::{content_hash}"


def get(key: str) -> Optional[str]:
    with _lock:
        return _load().get(key)


def set(key: str, content: str) -> None:  # noqa: A001 - matches dict.get/set naming, not shadowing builtin usage here
    with _lock:
        store = _load()
        store[key] = content
        _save()


def clear() -> None:
    """Wipe the entire persistent cache - e.g. after a prompt-version bump
    that should already have invalidated everything via the key itself,
    or for a manual reset during development."""
    global _store
    with _lock:
        _store = {}
        _save()


def stats() -> dict:
    with _lock:
        store = _load()
        return {"entries": len(store), "path": str(_CACHE_PATH)}
