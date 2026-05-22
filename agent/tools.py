"""Search tools for the study partner agent.

All tool descriptions are intentionally concise to minimize token usage —
tool schemas are a hidden token sink since they're included in every LLM call.

DuckDuckGo is the default (free, no API key). If SERPER_API_KEY is set,
Google search via Serper is used instead for higher-quality results.
"""

import os
import logging
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── Data directory for saving study plans ──────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "study_plans"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_search_engine():
    """Return the appropriate search function based on available API keys."""
    serper_key = os.getenv("SERPER_API_KEY", "")
    if serper_key:
        logger.info("Using Serper (Google) for web search")
        return _search_serper
    logger.info("Using DuckDuckGo for web search (free, no API key)")
    return _search_duckduckgo


def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """Search using DuckDuckGo (free, no API key needed)."""
    from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        logger.error("DuckDuckGo search failed: %s", e)
        return []


def _search_serper(query: str, max_results: int = 5) -> list[dict]:
    """Search using Serper (Google results, requires API key)."""
    import json
    import urllib.request

    api_key = os.getenv("SERPER_API_KEY", "")
    try:
        data = json.dumps({"q": query, "num": max_results}).encode()
        req = urllib.request.Request(
            "https://google.serper.dev/search",
            data=data,
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
        organic = body.get("organic", [])
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", ""),
            }
            for r in organic[:max_results]
        ]
    except Exception as e:
        logger.error("Serper search failed: %s", e)
        return []


# ── Tools exposed to the LangGraph agent ───────────────────────────────


@tool
def web_search(query: str) -> str:
    """Search the web for learning resources, courses, and information."""
    search_fn = _get_search_engine()
    results = search_fn(query, max_results=5)
    if not results:
        return "No search results found. Try a different query."
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r['title']}\n    URL: {r['url']}\n    {r['snippet']}"
        )
    return "\n\n".join(formatted)


@tool
def reddit_search(query: str) -> str:
    """Search Reddit for community recommendations and discussions."""
    search_fn = _get_search_engine()
    reddit_query = f"{query} site:reddit.com"
    results = search_fn(reddit_query, max_results=5)
    if not results:
        return "No Reddit discussions found. Try a different query."
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r['title']}\n    URL: {r['url']}\n    {r['snippet']}"
        )
    return "\n\n".join(formatted)


@tool
def save_study_plan(topic: str, content: str) -> str:
    """Save a study plan to a local markdown file."""
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)
    safe_name = safe_name.strip().replace(" ", "_").lower()
    if not safe_name:
        safe_name = "study_plan"
    filepath = DATA_DIR / f"{safe_name}.md"

    filepath.write_text(content, encoding="utf-8")
    return f"Study plan saved to {filepath}"


# ── Utility: raw search for state (used by nodes, not by LLM) ─────────


def raw_web_search(query: str, max_results: int = 5) -> list[dict]:
    """Perform a raw web search and return structured results.

    This is called directly by nodes (not via tool calling) to store
    results in the agent state for verification purposes.
    """
    search_fn = _get_search_engine()
    return search_fn(query, max_results=max_results)
