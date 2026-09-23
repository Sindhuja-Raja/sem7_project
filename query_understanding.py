"""
LLM-powered Query Understanding Agent.

Problem this solves: academic APIs (OpenAlex, Crossref, arXiv, CORE,
Semantic Scholar) mostly do lexical/keyword matching, not intent
understanding. Sending a raw research title straight to those APIs
retrieves a lot of noise - papers that share surface words with the
title but not its actual research intent.

This module sits between the user's title and retrieve.py. It asks an
LLM to *understand* the title first - domain, problem, technique(s),
synonyms, exclusions - then turns that understanding into search
queries tuned to how each individual academic API actually parses
queries (see build_source_queries).

Flow:
    raw title -> understand_query() -> QueryUnderstanding (structured)
              -> build_source_queries() -> {source_name: query_string}
              -> retrieve.fetch_all_sources(per_source_queries)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List

import config
import llm

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# JSON schema (deliverable #3) - documents the exact contract the LLM must
# fill in. Also used by _coerce() below for lightweight structural
# validation without pulling in an extra `jsonschema` dependency.
# --------------------------------------------------------------------------
QUERY_UNDERSTANDING_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "QueryUnderstanding",
    "type": "object",
    "properties": {
        "research_domain": {
            "type": "string",
            "description": "The broad academic field the title belongs to, e.g. 'Healthcare', 'Cybersecurity'.",
        },
        "research_problem": {
            "type": "string",
            "description": "The specific problem being solved, e.g. 'Medical Diagnosis'.",
        },
        "primary_ai_technique": {
            "type": "string",
            "description": "The main AI method named or implied by the title, e.g. 'Agentic AI'.",
        },
        "secondary_ai_techniques": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Supporting AI methods mentioned alongside the primary technique.",
        },
        "application_domain": {
            "type": "string",
            "description": "The specific real-world setting the technique is applied to, e.g. 'Clinical Decision-Making'.",
        },
        "related_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Technical terms closely related to the topic, useful for widening lexical search.",
        },
        "synonyms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Near-identical alternative names for the primary AI technique.",
        },
        "alternative_technical_terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Broader academic phrasings/jargon used for this topic in the literature, distinct from strict synonyms.",
        },
        "exclude_domains": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Unrelated application domains that commonly share vocabulary with this topic and should be filtered out.",
        },
        "search_intent": {
            "type": "string",
            "description": "One-sentence statement of what the researcher is actually trying to find.",
        },
        "search_query": {
            "type": "string",
            "description": "A boolean academic search query: OR within a concept group, AND between concept groups.",
        },
    },
    "required": [
        "research_domain", "research_problem", "primary_ai_technique",
        "secondary_ai_techniques", "application_domain", "related_keywords",
        "synonyms", "alternative_technical_terms", "exclude_domains",
        "search_intent", "search_query",
    ],
    "additionalProperties": False,
}


@dataclass
class QueryUnderstanding:
    raw_title: str
    research_domain: str = ""
    research_problem: str = ""
    primary_ai_technique: str = ""
    secondary_ai_techniques: List[str] = field(default_factory=list)
    application_domain: str = ""
    related_keywords: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    alternative_technical_terms: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    search_intent: str = ""
    search_query: str = ""
    llm_generated: bool = False  # False means the no-LLM fallback path was used

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Prompt (deliverable #1)
# --------------------------------------------------------------------------

_WORKED_EXAMPLE_INPUT = "Agentic AI for Medical Diagnosis using Multi-Agent Systems"
_WORKED_EXAMPLE_OUTPUT = {
    "research_domain": "Healthcare",
    "research_problem": "Medical Diagnosis",
    "primary_ai_technique": "Agentic AI",
    "secondary_ai_techniques": ["Multi-Agent Systems"],
    "application_domain": "Clinical Decision-Making",
    "related_keywords": [
        "Healthcare AI", "Clinical Decision Support", "Medical Diagnosis",
        "Autonomous AI Agents", "LLM Agents",
    ],
    "synonyms": ["AI Agents", "Autonomous Agents", "Intelligent Agents"],
    "alternative_technical_terms": [
        "LLM-based Agents", "Cognitive Agent Architecture", "Autonomous Multi-Agent Framework",
    ],
    "exclude_domains": ["Autonomous Vehicles", "Finance", "Agriculture", "Manufacturing", "Robotics"],
    "search_intent": (
        "Find papers that apply agentic AI / multi-agent systems specifically to medical "
        "diagnosis and clinical decision support, excluding unrelated application domains "
        "of agentic AI such as robotics, finance or autonomous vehicles."
    ),
    "search_query": (
        "(Agentic AI OR AI Agents OR Autonomous Agents OR LLM-based Agents) "
        "AND (Medical Diagnosis OR Healthcare OR Clinical Decision Support) "
        "AND (Multi-Agent Systems)"
    ),
}

SYSTEM_PROMPT = f"""You are an academic Query Understanding Agent for a research paper \
retrieval system. Academic search APIs (OpenAlex, Crossref, arXiv, CORE, Semantic \
Scholar) mostly do lexical keyword matching, not intent understanding - so your job is \
to read a research title, understand its actual research intent, and turn that \
understanding into a structured search plan.

Given a research title, analyze it and identify:
- research_domain: the primary academic field (e.g. "Healthcare", "Cybersecurity").
- research_problem: the specific problem being addressed.
- primary_ai_technique: the main AI method named or implied.
- secondary_ai_techniques: supporting AI methods mentioned alongside it.
- application_domain: the specific real-world setting the technique is applied to.
- related_keywords: technical terms closely tied to the topic (3-6 items).
- synonyms: near-identical alternative names for the primary AI technique (2-5 items).
- alternative_technical_terms: broader academic phrasings/jargon for this topic that are
  NOT strict synonyms of the primary technique (2-5 items).
- exclude_domains: unrelated application domains that commonly share vocabulary with this
  topic and would pollute results if not filtered out (3-6 items).
- search_intent: one sentence stating what the researcher actually wants to find.
- search_query: a boolean query string using OR to join terms within one concept group
  and AND between concept groups, e.g. "(A OR B) AND (C OR D)". Build it from the
  primary technique + its synonyms, the research problem + application domain, and any
  secondary techniques. This must read as an academic literature search query, not a
  natural-language web-search query.

Rules:
- Extract only meaningful technical/academic concepts. Ignore stop words and filler.
- Never invent an unrelated research domain - stay grounded in what the title implies.
- Keep every array field short and non-redundant (no duplicate or near-duplicate terms).
- Respond with ONLY a single valid JSON object matching this exact schema - no markdown
  fences, no commentary before or after it:
{json.dumps(QUERY_UNDERSTANDING_SCHEMA['properties'], indent=2)}

Example.
Input title: "{_WORKED_EXAMPLE_INPUT}"
Output JSON:
{json.dumps(_WORKED_EXAMPLE_OUTPUT, indent=2)}
"""


# --------------------------------------------------------------------------
# understand_query (deliverable #2, part 1)
# --------------------------------------------------------------------------

def _as_str_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce(data: dict, raw_title: str) -> QueryUnderstanding:
    """Turn a (possibly incomplete/malformed) LLM JSON dict into a valid
    QueryUnderstanding, filling any missing field with a safe default
    instead of raising - the schema in QUERY_UNDERSTANDING_SCHEMA
    documents the intended shape, this is the defensive runtime check."""
    return QueryUnderstanding(
        raw_title=raw_title,
        research_domain=str(data.get("research_domain") or "").strip(),
        research_problem=str(data.get("research_problem") or "").strip(),
        primary_ai_technique=str(data.get("primary_ai_technique") or "").strip(),
        secondary_ai_techniques=_as_str_list(data.get("secondary_ai_techniques")),
        application_domain=str(data.get("application_domain") or "").strip(),
        related_keywords=_as_str_list(data.get("related_keywords")),
        synonyms=_as_str_list(data.get("synonyms")),
        alternative_technical_terms=_as_str_list(data.get("alternative_technical_terms")),
        exclude_domains=_as_str_list(data.get("exclude_domains")),
        search_intent=str(data.get("search_intent") or "").strip(),
        search_query=str(data.get("search_query") or "").strip() or raw_title,
        llm_generated=True,
    )


def _fallback(raw_title: str) -> QueryUnderstanding:
    """No LLM available (or it failed) - fall back to searching with the
    raw title verbatim, same behavior as directly hitting the APIs."""
    return QueryUnderstanding(
        raw_title=raw_title,
        primary_ai_technique=raw_title,
        search_intent="LLM unavailable - falling back to the raw title as the search query.",
        search_query=raw_title,
        llm_generated=False,
    )


def understand_query(title: str) -> QueryUnderstanding:
    """Analyze a research title with the LLM and return a structured
    QueryUnderstanding. Degrades gracefully to _fallback() if the LLM is
    unavailable or returns something unparseable - retrieval always has
    a usable search_query either way."""
    if llm.is_daily_quota_exhausted():
        return _fallback(title)
    try:
        content = llm._call_groq(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": title},
            ],
            max_tokens=700,
            temperature=0.2,
            json_mode=True,
            agent_name="QueryUnderstanding",
            paper_id=title,
            cache_version=config.LLM_CACHE_VERSION,
        )
    except llm.GroqDailyQuotaExceeded as exc:
        logger.warning("Query understanding: daily Groq quota exhausted - %s", exc)
        return _fallback(title)
    if not content:
        return _fallback(title)

    data = llm.parse_json(content)
    if not isinstance(data, dict):
        logger.warning("Query Understanding Agent returned unparseable JSON: %s", content[:200])
        return _fallback(title)

    return _coerce(data, title)


# --------------------------------------------------------------------------
# build_source_queries (deliverable #2, part 2 + #4 integration)
# --------------------------------------------------------------------------

def _dedupe(terms: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in terms:
        t = (t or "").strip()
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _quote_if_multiword(term: str) -> str:
    return f'"{term}"' if " " in term else term


def _boolean_group(terms: List[str], field_name: str = "all", max_terms: int = 5) -> str:
    """Build an OR-group like (all:"a" OR all:"b") for engines with real
    boolean query grammar (arXiv, CORE)."""
    terms = _dedupe(terms)[:max_terms]
    if not terms:
        return ""
    parts = [f"{field_name}:{_quote_if_multiword(t)}" for t in terms]
    return parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"


def build_source_queries(qu: QueryUnderstanding) -> Dict[str, str]:
    """Turn one QueryUnderstanding into a per-source query string.

    Different academic APIs parse queries very differently:
      - arXiv's search_query param has real boolean grammar (AND/OR plus
        field prefixes like all:"term"), so it gets a properly built
        boolean query.
      - CORE's `q` param is Elasticsearch-query-string based and also
        understands AND/OR/parentheses, so it gets the LLM's canonical
        boolean search_query directly.
      - OpenAlex, Crossref and Semantic Scholar do relevance-ranked free
        text search on the raw string rather than parsing boolean syntax
        - sending them literal "AND"/"OR"/parentheses would just search
        for those words and hurt relevance. They instead get a flattened,
        deduplicated bag of the most important terms.
    """
    primary_terms = [qu.primary_ai_technique] + qu.synonyms + qu.alternative_technical_terms
    problem_terms = [qu.research_problem, qu.application_domain] + qu.related_keywords[:3]
    secondary_terms = list(qu.secondary_ai_techniques)

    arxiv_groups = [
        _boolean_group(primary_terms),
        _boolean_group(problem_terms),
        _boolean_group(secondary_terms),
    ]
    arxiv_query = " AND ".join(g for g in arxiv_groups if g) or qu.raw_title

    core_query = qu.search_query or qu.raw_title

    flattened_terms = _dedupe(
        [qu.primary_ai_technique, qu.research_problem]
        + qu.secondary_ai_techniques
        + qu.related_keywords[:5]
    )
    flattened_query = " ".join(flattened_terms) or qu.raw_title

    return {
        config.SOURCE_OPENALEX: flattened_query,
        config.SOURCE_CROSSREF: flattened_query,
        config.SOURCE_SEMANTIC_SCHOLAR: flattened_query,
        config.SOURCE_CORE: core_query,
        config.SOURCE_ARXIV: arxiv_query,
    }
