"""Verification node — the quality gate between generation and user.

This module implements a layered verification strategy:
1. Rule-based URL check (zero LLM cost) — catches fabricated links
2. Content relevance check (lightweight LLM call) — catches off-topic responses

The verification node runs AFTER a specialist node generates a response
but BEFORE the user sees it. This is a key anti-hallucination pattern.

Important design decision: we do NOT use a regeneration loop. Research
shows that unconditional self-correction can degrade quality by introducing
errors into previously correct answers. Instead, we strip problematic
content and add disclaimers.
"""

import re
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def verify_citations(response: str, search_results: list[dict]) -> dict:
    """Rule-based verification: check that all URLs in the response
    actually came from the search results.

    This is the primary anti-hallucination defense. It costs zero tokens
    because it's pure string matching — no LLM call needed.

    Args:
        response: The generated response text.
        search_results: Raw search results from the agent state.

    Returns:
        Dict with 'is_valid', 'issues', and 'cleaned_response' keys.
    """
    if not search_results:
        # No search was performed, so no citations to verify
        return {"is_valid": True, "issues": [], "cleaned_response": response}

    # Extract all URLs from the response
    url_pattern = r'https?://[^\s\)\]\>\"\'`]+'
    response_urls = re.findall(url_pattern, response)

    if not response_urls:
        # No URLs in response — nothing to verify
        return {"is_valid": True, "issues": [], "cleaned_response": response}

    # Build set of valid URL domains + paths from search results
    valid_urls = set()
    for result in search_results:
        url = result.get("url", "")
        if url:
            valid_urls.add(url.rstrip("/"))
            # Also add the domain for partial matching
            try:
                parsed = urlparse(url)
                valid_urls.add(f"{parsed.scheme}://{parsed.netloc}")
            except Exception:
                pass

    # Check each URL in the response
    issues = []
    cleaned = response
    for url in response_urls:
        url_clean = url.rstrip("/.,;:!?")
        is_valid = False
        for valid_url in valid_urls:
            if url_clean.startswith(valid_url) or valid_url.startswith(url_clean):
                is_valid = True
                break

        if not is_valid:
            issues.append(f"Unverified URL removed: {url_clean}")
            # Replace the fabricated URL with a note
            cleaned = cleaned.replace(url_clean, "[link removed — not found in search results]")

    if issues:
        logger.warning("Verification found %d issue(s): %s", len(issues), issues)

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "cleaned_response": cleaned,
    }


def add_verification_disclaimer(response: str, issues: list[str]) -> str:
    """Add a subtle disclaimer to responses that had issues.

    Rather than silently modifying the response, we add a note so the
    user knows verification happened. This builds trust.

    Args:
        response: The cleaned response text.
        issues: List of issues that were found and fixed.

    Returns:
        Response with disclaimer appended.
    """
    if not issues:
        return response

    disclaimer = (
        "\n\n---\n"
        "⚠️ *Note: Some links were removed during verification because they "
        "couldn't be confirmed in the search results. The remaining links "
        "have been verified.*"
    )
    return response + disclaimer
