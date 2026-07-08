import asyncio
import logging

import config

log = logging.getLogger("sentient_bot.web")

# ddgs (formerly duckduckgo_search): free DuckDuckGo results, no API key required.
try:
    from ddgs import DDGS
except ImportError:  # keep the bot runnable even if the optional dep isn't installed
    DDGS = None


def _search_sync(query: str, max_results: int, region: str) -> list[dict]:
    """Blocking DuckDuckGo text search. Runs in a thread executor (ddgs is synchronous)."""
    results: list[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region=region, max_results=max_results):
            results.append({
                "title": (r.get("title") or "").strip(),
                "url": (r.get("href") or "").strip(),
                "snippet": (r.get("body") or "").strip(),
            })
    return results


async def search(query: str, max_results: int | None = None,
                 region: str | None = None) -> list[dict]:
    """Run a DuckDuckGo text search off the event loop.

    Returns a list of {title, url, snippet} dicts, or [] on any failure/timeout so callers can
    treat "no web context" as a normal, non-fatal case.
    """
    if DDGS is None:
        log.warning("ddgs not installed -- web search disabled")
        return []
    if not query:
        return []
    max_results = max_results or config.WEB_SEARCH_MAX_RESULTS
    region = region or config.WEB_SEARCH_REGION
    loop = asyncio.get_running_loop()
    log.info(f"Web search started: query={query!r} region={region!r} max_results={max_results}")
    try:
        results = await asyncio.wait_for(
            loop.run_in_executor(None, _search_sync, query, max_results, region),
            timeout=config.WEB_SEARCH_TIMEOUT_SECONDS,
        )
        log.info(f"Web search completed: query={query!r} results={len(results)}")
        return results
    except asyncio.TimeoutError:
        log.warning(f"Web search timed out for query: {query!r}")
    except Exception as e:
        log.warning(f"Web search failed for {query!r}: {e}")
    return []


def format_results(results: list[dict], snippet_chars: int = 300) -> str:
    """Compact, prompt-friendly rendering of search results."""
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        snippet = " ".join(r["snippet"].split())
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars].rstrip() + "..."
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {snippet}")
    return "\n".join(lines)
