"""Agent tools for the RAG "Agent mode": ``rag_search`` and ``web_search``.

Both tools return a structured result dict and NEVER raise -- a failure
(timeout, network error, empty index, missing provider dependency) is
reported as ``{"success": False, "error": ...}`` so a calling LangGraph node
can degrade gracefully (e.g. fall back to a RAG-only answer) instead of
crashing the whole request. This is the core lesson from the OmniGent
analysis this design is based on: a sub-agent/tool failure must never leave
the caller with nothing. See docs/adr/0005-rag-agent-supervisor-pattern.md.
"""

from __future__ import annotations

import concurrent.futures
import logging
from abc import ABC, abstractmethod
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.config import get_settings
from app.core.retriever import retrieve_with_scores

_logger = logging.getLogger(__name__)


def rag_search(query: str, top_k: int = 4) -> dict[str, Any]:
    """Search the local knowledge base. Never raises.

    :param query: The search query.
    :param top_k: Number of documents to retrieve.
    :returns: On success, ``{"success": True, "results": [{"content", "source",
        "score"}, ...], "min_distance": float}``. ``score``/``min_distance`` is
        the **ChromaDB L2 distance** -- lower means more similar, this is a
        distance, not a similarity score. On failure (e.g. empty/uninitialized
        index), ``{"success": False, "error": str, "results": [], "min_distance": inf}``.
    """
    try:
        pairs = retrieve_with_scores(query, top_k=top_k)
    except Exception as e:
        _logger.warning("rag_search failed for query=%r: %s", query, e)
        return {"success": False, "error": str(e), "results": [], "min_distance": float("inf")}

    results = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": score,
        }
        for doc, score in pairs
    ]
    min_distance = min((r["score"] for r in results), default=float("inf"))
    return {"success": True, "results": results, "min_distance": min_distance}


class WebSearchProvider(ABC):
    """A pluggable web search backend (mirrors ``app.sources.base.DataSource``).

    See docs/adr/0006-web-search-provider-choice.md for why the default is
    free/no-API-key and how to switch providers.
    """

    #: short id used in config / the registry, e.g. "duckduckgo"
    name: str = "base"

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Return ``[{"title": ..., "snippet": ..., "url": ...}, ...]``."""
        raise NotImplementedError


class DuckDuckGoProvider(WebSearchProvider):
    """Free, no-API-key web search via the ``ddgs`` package. Default provider."""

    name = "duckduckgo"

    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        from ddgs import DDGS

        with DDGS() as client:
            hits = list(client.text(query, max_results=max_results))
        return [
            {
                "title": h.get("title", ""),
                "snippet": h.get("body", ""),
                "url": h.get("href", ""),
            }
            for h in hits
        ]


class TavilyProvider(WebSearchProvider):
    """Higher-quality paid alternative. Requires ``TAVILY_API_KEY``."""

    name = "tavily"

    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        from tavily import TavilyClient

        settings = get_settings()
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("content", ""),
                "url": r.get("url", ""),
            }
            for r in response.get("results", [])
        ]


_PROVIDERS: dict[str, type[WebSearchProvider]] = {
    "duckduckgo": DuckDuckGoProvider,
    "tavily": TavilyProvider,
}


def _get_provider() -> WebSearchProvider:
    settings = get_settings()
    provider_cls = _PROVIDERS.get(settings.web_search_provider, DuckDuckGoProvider)
    return provider_cls()


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _search_with_retry(
    provider: WebSearchProvider, query: str, max_results: int
) -> list[dict[str, str]]:
    return provider.search(query, max_results)


def web_search(query: str, max_results: int | None = None) -> dict[str, Any]:
    """Search the web via the configured provider. Never raises.

    Bounded retry (2 attempts, 1s apart) for transient errors, then a hard
    wall-clock timeout enforced via a worker thread -- a hung provider call
    must not block the agent loop forever. Any failure (timeout, exhausted
    retries, missing optional dependency) degrades to a structured error
    result instead of propagating, so the caller can fall back to a
    RAG-only answer.

    :param query: The search query.
    :param max_results: Number of results to return; defaults to
        ``settings.web_search_max_results``.
    :returns: ``{"success": True, "results": [...], "provider": str}`` or
        ``{"success": False, "error": str, "results": []}``.
    """
    settings = get_settings()
    n = max_results or settings.web_search_max_results
    provider = _get_provider()

    # Deliberately NOT a `with ThreadPoolExecutor(...)` block: its __exit__
    # calls shutdown(wait=True), which would block here until the hung
    # worker thread finishes -- silently defeating the timeout below (the
    # exact failure mode this function exists to prevent). shutdown(wait=False)
    # on the timeout path lets this function return promptly; the orphaned
    # worker thread finishes on its own and its result is simply discarded.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_search_with_retry, provider, query, n)
        results = future.result(timeout=settings.web_search_timeout_s)
    except concurrent.futures.TimeoutError:
        _logger.warning(
            "web_search timed out for query=%r after %ss", query, settings.web_search_timeout_s
        )
        pool.shutdown(wait=False, cancel_futures=True)
        return {
            "success": False,
            "error": f"timed out after {settings.web_search_timeout_s}s",
            "results": [],
        }
    except Exception as e:
        _logger.warning("web_search failed for query=%r: %s", query, e)
        pool.shutdown(wait=False, cancel_futures=True)
        return {"success": False, "error": str(e), "results": []}

    pool.shutdown(wait=False)
    return {"success": True, "results": results, "provider": provider.name}
