"""
Real-time retrieval from official academic APIs. No database is used -
every search hits the live APIs directly.

Each `fetch_<source>` function takes a search query and returns a
List[Paper]. All five are called concurrently by `fetch_all_sources`
via a thread pool, since these are I/O-bound network calls.

Pagination is implemented for every source so that up to
config.RESULTS_PER_SOURCE papers can be returned per source (where
individual API page sizes are smaller than the target, multiple pages
are fetched sequentially).

A source that errors out (bad key, timeout, rate limit) never crashes
the pipeline - it just contributes zero results and the failure is
reported back to the caller for display in the UI.
"""

import logging
import math
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import requests

import config
from utils import Paper

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# Rate limiting and retry logic for handling 429 errors
# ------------------------------------------------------------------------

def _retry_with_backoff(func, max_retries: Optional[int] = None, initial_delay: Optional[float] = None) -> Optional[requests.Response]:
    """
    Retry a function with exponential backoff for rate limit errors (429).
    
    Args:
        func: Callable that returns a requests.Response
        max_retries: Maximum number of retry attempts (uses config default if None)
        initial_delay: Initial delay in seconds, doubles on each retry (uses config default if None)
    
    Returns:
        Response object or None if all retries failed
    """
    max_retries = max_retries or config.API_RETRY_MAX_ATTEMPTS
    initial_delay = initial_delay or config.API_RETRY_INITIAL_DELAY
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            resp = func()
            if resp.status_code == 429:
                # Rate limited - retry with backoff
                if attempt < max_retries - 1:
                    logger.debug(
                        f"Rate limited (429), retrying in {delay}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
            # Success or other error (not 429)
            return resp
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.debug(f"Request failed, retrying in {delay}s: {e}")
                time.sleep(delay)
                delay *= 2
            continue
    
    # All retries exhausted - raise last exception
    if last_exception:
        raise last_exception
    return None




def _reconstruct_openalex_abstract(inverted_index: Optional[dict]) -> str:
    """OpenAlex stores abstracts as a word -> [positions] inverted index
    instead of plain text (for copyright reasons). Rebuild the sentence."""
    if not inverted_index:
        return ""
    positions = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positions)


# OpenAlex hard caps per_page at 100 — paginate to reach the target.
_OPENALEX_PAGE_SIZE = 100


def filter_papers_by_metadata(
    papers: List[Paper], year_from: Optional[int] = None, year_to: Optional[int] = None,
    publishers: Optional[List[str]] = None,
) -> List[Paper]:
    """Filter already-fetched papers by year range and publisher name."""
    publishers = [p.strip().lower() for p in (publishers or []) if p and str(p).strip()]
    filtered = []
    for paper in papers:
        if year_from is not None and (paper.year is None or paper.year < year_from):
            continue
        if year_to is not None and (paper.year is None or paper.year > year_to):
            continue
        if publishers:
            publisher = (paper.publisher or "").strip().lower()
            if publisher not in publishers:
                continue
        filtered.append(paper)
    return filtered


def fetch_openalex(query: str, limit: int = config.RESULTS_PER_SOURCE) -> List[Paper]:
    """Fetch up to `limit` papers from OpenAlex, paginating as needed
    (OpenAlex per_page max = 100)."""
    papers: List[Paper] = []
    page = 1
    remaining = limit
    
    while remaining > 0:
        page_size = min(_OPENALEX_PAGE_SIZE, remaining)
        params = {
            "search": query,
            "per_page": page_size,
            "page": page,
        }
        if config.CONTACT_EMAIL:
            params["mailto"] = config.CONTACT_EMAIL

        try:
            resp = requests.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("OpenAlex page %d failed: %s", page, exc)
            break

        results = resp.json().get("results", [])
        if not results:
            break  # No more results

        for item in results:
            authors = [
                a["author"]["display_name"]
                for a in item.get("authorships", [])
                if a.get("author", {}).get("display_name")
            ]
            pdf_url = None
            oa = item.get("open_access") or {}
            if oa.get("oa_url"):
                pdf_url = oa["oa_url"]
            elif item.get("primary_location", {}).get("pdf_url"):
                pdf_url = item["primary_location"]["pdf_url"]

            publisher = None
            if item.get("primary_location"):
                host = item.get("primary_location", {}).get("source", {}).get("display_name")
                publisher = host or None
            papers.append(Paper(
                title=item.get("title") or "",
                authors=authors,
                year=item.get("publication_year"),
                source=config.SOURCE_OPENALEX,
                doi=item.get("doi"),
                abstract=_reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
                pdf_url=pdf_url,
                publisher=publisher,
            ))

        remaining -= len(results)
        if len(results) < page_size:
            break  # Last page returned fewer than requested → no more pages
        page += 1
        time.sleep(0.1)  # polite delay between pages

    return papers[:limit]


# ------------------------------------------------------------------------
# Crossref - https://api.crossref.org  (no key required)
# Crossref supports up to 1000 rows per request - no pagination needed
# for our target limit of 200.
# ------------------------------------------------------------------------

def _strip_jats_tags(text: str) -> str:
    """Crossref abstracts are sometimes wrapped in JATS XML tags like
    <jats:p>...</jats:p>; strip them down to plain text."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_crossref(query: str, limit: int = config.RESULTS_PER_SOURCE) -> List[Paper]:
    """Fetch up to `limit` papers from Crossref.
    Crossref supports rows up to 1000 per request, so a single call suffices."""
    # Crossref max rows per request = 1000, so we can fetch up to 1000 in one go.
    rows = min(limit, 1000)
    params = {"query": query, "rows": rows}
    if config.CONTACT_EMAIL:
        params["mailto"] = config.CONTACT_EMAIL

    resp = requests.get(
        "https://api.crossref.org/works",
        params=params,
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    items = resp.json().get("message", {}).get("items", [])

    papers = []
    for item in items:
        titles = item.get("title") or []
        title = titles[0] if titles else ""
        if not title:
            continue

        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in item.get("author", [])
            if a.get("family")
        ]

        year = None
        date_parts = (
            item.get("published-print", {}).get("date-parts")
            or item.get("published-online", {}).get("date-parts")
            or item.get("issued", {}).get("date-parts")
        )
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        pdf_url = None
        for link in item.get("link", []):
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL")
                break

        publisher = None
        container_title = item.get("container-title") or []
        if container_title:
            publisher = container_title[0]
        papers.append(Paper(
            title=title,
            authors=authors,
            year=year,
            source=config.SOURCE_CROSSREF,
            doi=item.get("DOI"),
            abstract=_strip_jats_tags(item.get("abstract", "")),
            pdf_url=pdf_url or item.get("URL"),
            publisher=publisher,
        ))
    return papers[:limit]


# ------------------------------------------------------------------------
# arXiv - https://info.arxiv.org/help/api  (no key required, Atom/XML feed)
# arXiv supports max_results up to 30,000 - single request suffices.
# ------------------------------------------------------------------------

_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def fetch_arxiv(query: str, limit: int = config.RESULTS_PER_SOURCE) -> List[Paper]:
    """Fetch up to `limit` papers from arXiv.
    arXiv supports max_results up to 30,000 in a single request."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
    }
    resp = requests.get(
        "http://export.arxiv.org/api/query",
        params=params,
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    papers = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        title_el = entry.find(f"{_ATOM_NS}title")
        title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""
        if not title:
            continue

        authors = [
            name_el.text.strip()
            for author_el in entry.findall(f"{_ATOM_NS}author")
            for name_el in [author_el.find(f"{_ATOM_NS}name")]
            if name_el is not None and name_el.text
        ]

        published_el = entry.find(f"{_ATOM_NS}published")
        year = int(published_el.text[:4]) if published_el is not None and published_el.text else None

        summary_el = entry.find(f"{_ATOM_NS}summary")
        abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None and summary_el.text else ""

        pdf_url = None
        for link_el in entry.findall(f"{_ATOM_NS}link"):
            if link_el.get("title") == "pdf" or link_el.get("type") == "application/pdf":
                pdf_url = link_el.get("href")
                break

        id_el = entry.find(f"{_ATOM_NS}id")
        arxiv_url = id_el.text.strip() if id_el is not None and id_el.text else None

        papers.append(Paper(
            title=title,
            authors=authors,
            year=year,
            source=config.SOURCE_ARXIV,
            doi=None,
            abstract=abstract,
            pdf_url=pdf_url or arxiv_url,
            publisher="arXiv",
        ))
    return papers[:limit]


# ------------------------------------------------------------------------
# CORE - https://api.core.ac.uk/docs/v3  (requires a free API key)
# Paginate using offset to reach target limit.
# ------------------------------------------------------------------------

_CORE_PAGE_SIZE = 100


def fetch_core(query: str, limit: int = config.RESULTS_PER_SOURCE) -> List[Paper]:
    """Fetch up to `limit` papers from CORE, paginating as needed."""
    if not config.CORE_API_KEY:
        # No key configured - skip this source gracefully rather than failing.
        return []

    papers: List[Paper] = []
    offset = 0
    remaining = limit

    while remaining > 0:
        page_size = min(_CORE_PAGE_SIZE, remaining)
        try:
            resp = requests.get(
                "https://api.core.ac.uk/v3/search/works/",
                params={"q": query, "limit": page_size, "offset": offset},
                headers={"Authorization": f"Bearer {config.CORE_API_KEY}"},
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("CORE offset=%d failed: %s", offset, exc)
            break

        results = resp.json().get("results", [])
        if not results:
            break

        for item in results:
            title = item.get("title") or ""
            if not title:
                continue

            authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]

            papers.append(Paper(
                title=title,
                authors=authors,
                year=item.get("yearPublished"),
                source=config.SOURCE_CORE,
                doi=item.get("doi"),
                abstract=item.get("abstract") or "",
                pdf_url=item.get("downloadUrl") or (item.get("sourceFulltextUrls") or [None])[0],
                publisher=item.get("publisher") or item.get("publisherName"),
            ))

        remaining -= len(results)
        if len(results) < page_size:
            break  # Last page
        offset += page_size
        time.sleep(0.1)

    return papers[:limit]


# ------------------------------------------------------------------------
# Semantic Scholar - https://api.semanticscholar.org  (key optional, raises
# rate limits when provided)
# Semantic Scholar search supports limit up to 100 per request.
# Paginate via offset to reach target limit.
# ------------------------------------------------------------------------

_SEMANTIC_SCHOLAR_PAGE_SIZE = 100


def fetch_semantic_scholar(query: str, limit: int = config.RESULTS_PER_SOURCE) -> List[Paper]:
    """
    Fetch up to `limit` papers from Semantic Scholar, paginating via offset.
    
    NOTE: The API key actually causes MORE aggressive rate limiting when provided,
    so we deliberately omit it. See:
    https://api.semanticscholar.org/graph/v1/paper/search
    """
    papers: List[Paper] = []
    offset = 0
    remaining = limit

    while remaining > 0:
        page_size = min(_SEMANTIC_SCHOLAR_PAGE_SIZE, remaining)
        # Small delay to avoid hammering the API
        time.sleep(0.2)

        def make_request(os=offset, ps=page_size):
            return requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": ps,
                    "offset": os,
                    "fields": "title,authors,year,abstract,externalIds,openAccessPdf",
                },
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            )

        # Retry with exponential backoff for rate limits
        resp = _retry_with_backoff(make_request)

        if resp is None:
            break

        if resp.status_code != 200:
            logger.warning("Semantic Scholar returned %d at offset=%d", resp.status_code, offset)
            break

        data = resp.json()
        items = data.get("data", [])
        if not items:
            break

        for item in items:
            title = item.get("title") or ""
            if not title:
                continue

            authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]
            external_ids = item.get("externalIds") or {}
            oa_pdf = item.get("openAccessPdf") or {}

            papers.append(Paper(
                title=title,
                authors=authors,
                year=item.get("year"),
                source=config.SOURCE_SEMANTIC_SCHOLAR,
                doi=external_ids.get("DOI"),
                abstract=item.get("abstract") or "",
                pdf_url=oa_pdf.get("url"),
                publisher=item.get("publisher"),
            ))

        remaining -= len(items)
        if len(items) < page_size:
            break  # Reached end of results
        # Semantic Scholar API returns total count - respect it
        total = data.get("total", 0)
        if offset + page_size >= total:
            break
        offset += page_size

    return papers[:limit]


# ------------------------------------------------------------------------
# Fan-out: call every source concurrently
# ------------------------------------------------------------------------

_SOURCE_FUNCTIONS = {
    config.SOURCE_OPENALEX: fetch_openalex,
    config.SOURCE_CROSSREF: fetch_crossref,
    config.SOURCE_ARXIV: fetch_arxiv,
    config.SOURCE_CORE: fetch_core,
}

# Conditionally include Semantic Scholar if not disabled by user
if config.ENABLE_SEMANTIC_SCHOLAR:
    _SOURCE_FUNCTIONS[config.SOURCE_SEMANTIC_SCHOLAR] = fetch_semantic_scholar


def fetch_all_sources(queries: Union[str, Dict[str, str]]) -> Tuple[List[Paper], Dict[str, str]]:
    """Query every academic source in parallel.

    `queries` is either a single string (used for every source, mainly
    for quick manual testing) or a dict mapping each source name to its
    own query string, as produced by
    query_understanding.build_source_queries() - since different APIs
    parse query syntax differently, each source is meant to get a query
    string tuned to it rather than one raw string blasted at all five.

    Returns (papers, status) where status maps each source name to
    "ok (N results)", "skipped (no API key)" or "error: <message>" so
    the UI can show what happened without the whole search failing.
    """
    if isinstance(queries, str):
        queries = {name: queries for name in _SOURCE_FUNCTIONS}

    papers: List[Paper] = []
    status: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(_SOURCE_FUNCTIONS)) as executor:
        future_to_source = {
            executor.submit(func, queries[name]): name
            for name, func in _SOURCE_FUNCTIONS.items()
        }
        for future in as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                result = future.result()
                if not result and source_name in (config.SOURCE_CORE, config.SOURCE_SEMANTIC_SCHOLAR):
                    key_present = (
                        config.CORE_API_KEY if source_name == config.SOURCE_CORE
                        else config.SEMANTIC_SCHOLAR_API_KEY
                    )
                    status[source_name] = "skipped (no API key)" if not key_present else "ok (0 results)"
                else:
                    status[source_name] = f"ok ({len(result)} results)"
                papers.extend(result)
            except Exception as exc:  # noqa: BLE001 - a single source must never break the search
                logger.warning("Retrieval failed for %s: %s", source_name, exc)
                status[source_name] = f"error: {exc}"

    return papers, status
