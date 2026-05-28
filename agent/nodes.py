"""Node implementations for the study partner agent.

Each node is a function that takes the agent state and returns a partial
state update. Nodes are the "workers" in the LangGraph — each one handles
a specific responsibility.

Architecture:
  summarize → router → [specialist] → verify → respond

Quality patterns used here:
- Structured outputs (Pydantic schemas) for routing
- Source grounding (tagged documents for citations)
- Confidence scoring (router knows when it's uncertain)
- Conversation summarization (token optimization)
- Quiz generation with active recall
"""

import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent.prompts import (
    ROUTER_PROMPT,
    RESOURCE_FINDER_PROMPT,
    STUDY_PLANNER_PROMPT,
    PROFESSOR_PROMPT,
    QUIZ_MASTER_PROMPT,
    OFF_TOPIC_PROMPT,
    SUMMARIZE_PROMPT,
)
from agent.schemas import RouterDecision
from agent.tools import raw_web_search, save_study_plan
from agent.memory import trim_conversation
from agent.verification import verify_citations, add_verification_disclaimer
from agent.llm import get_llm

logger = logging.getLogger(__name__)

# ── Execution guardrails ───────────────────────────────────────────────
MAX_SEARCH_QUERIES = 6  # Max search queries per turn (raised for multi-source)
MAX_MESSAGES_BEFORE_SUMMARY = 10  # Trigger summarization at this count


# ── Cached LLM instance ───────────────────────────────────────────────
_llm = None


def _get_llm():
    """Get or create the cached LLM instance."""
    global _llm
    if _llm is None:
        _llm = get_llm(streaming=True)
    return _llm


def _perform_search(topic: str, queries: list[str]) -> list[dict]:
    """Shared search logic: run queries and deduplicate results."""
    all_results = []
    seen_urls = set()
    for query in queries[:MAX_SEARCH_QUERIES]:
        results = raw_web_search(query, max_results=4)
        for r in results:
            if r.get("url") and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                all_results.append(r)
    return all_results


def _multi_source_search(topic: str, context: str = "") -> list[dict]:
    """Google-AI-style multi-source search across platforms.

    Performs targeted searches across YouTube, Reddit, educational
    platforms, Stack Exchange, and general web — then deduplicates
    and merges results. This emulates how Google AI goes into multiple
    websites to assemble the best resources.

    Args:
        topic: The academic topic to search for.
        context: Additional context (e.g., "beginner", "advanced").

    Returns:
        Deduplicated list of search results from all sources.
    """
    qt = _quote_topic(topic)
    ctx = f" {context}" if context else ""

    # Platform-specific queries — each targets a different source
    # NOTE: DuckDuckGo doesn't support OR between site: operators well,
    # so we target specific platforms. To avoid rate limits, we use fewer queries.
    platform_queries = [
        # Educational platforms and courses
        f"{topic}{ctx} course tutorial coursera edx khan academy",
        # YouTube — specifically educational channels/lectures
        f"{topic}{ctx} full course lecture tutorial site:youtube.com",
        # Reddit — community recommendations
        f"{topic}{ctx} best resources site:reddit.com",
        # Open courseware and generic guides
        f"{topic}{ctx} tutorial guide free textbook PDF",
    ]

    import time
    all_results = []
    seen_urls = set()

    for query in platform_queries[:MAX_SEARCH_QUERIES]:
        results = raw_web_search(query, max_results=4)
        for r in results:
            if r.get("url") and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                # Tag the source platform for better organization
                r["source_platform"] = _detect_platform(r["url"])
                all_results.append(r)
        
        # Add a delay to prevent DuckDuckGo rate limiting
        time.sleep(1.0)

    logger.info(
        "Multi-source search for '%s': %d total results from %d queries",
        topic, len(all_results), len(platform_queries),
    )
    return all_results


def _detect_platform(url: str) -> str:
    """Detect which platform a URL belongs to for result tagging."""
    url_lower = url.lower()
    platform_map = {
        "youtube.com": "youtube", "youtu.be": "youtube",
        "reddit.com": "reddit",
        "coursera.org": "mooc", "edx.org": "mooc",
        "khanacademy.org": "mooc", "ocw.mit.edu": "mooc",
        "udemy.com": "mooc", "udacity.com": "mooc",
        "stackoverflow.com": "qa", "stackexchange.com": "qa",
        "math.stackexchange.com": "qa",
        "arxiv.org": "paper", "scholar.google.com": "paper",
        "github.com": "code", "freecodecamp.org": "code",
        "wikipedia.org": "reference",
        "geeksforgeeks.org": "tutorial",
        "medium.com": "article", "towardsdatascience.com": "article",
    }
    for domain, platform in platform_map.items():
        if domain in url_lower:
            return platform
    return "web"


def _format_tagged_docs(results: list[dict]) -> str:
    """Format search results as tagged XML documents for source grounding."""
    return "\n\n".join(
        f'<doc id="{i+1}">\n'
        f"  <title>{r['title']}</title>\n"
        f"  <url>{r['url']}</url>\n"
        f"  <content>{r['snippet']}</content>\n"
        f"</doc>"
        for i, r in enumerate(results)
    )


def _track_topic(topic: str, existing_topics: list[str]) -> list[str]:
    """Add a topic to the covered topics list (deduplicated)."""
    if not topic or topic == "general":
        return existing_topics
    normalized = topic.lower().strip()
    if normalized not in [t.lower() for t in existing_topics]:
        return existing_topics + [topic]
    return existing_topics


# Conversational phrases to strip when building search queries
_NOISE_PHRASES = [
    "i'm having trouble understanding",
    "i am having trouble understanding",
    "i don't understand",
    "i do not understand",
    "could you explain",
    "can you explain",
    "please explain",
    "help me understand",
    "help me with",
    "i need help with",
    "i'm confused about",
    "i am confused about",
    "tell me about",
    "what exactly is",
    "what exactly are",
    "what is a",
    "what is an",
    "what is",
    "what are",
    "how does",
    "how do",
    "why does",
    "why do",
    "explain to me",
    "explain",
    "like i'm a beginner",
    "like i am a beginner",
    "in simple terms",
    "simply",
]


def _clean_search_query(user_message: str, topic: str) -> str:
    """Extract the core concept from a conversational question.

    Strips common conversational phrases so the search engine gets
    a clean, concept-focused query instead of a full English sentence.

    Example:
        "I'm having trouble understanding eigen values, could you explain it?"
        → "eigen values"
    """
    cleaned = user_message.lower().strip()

    # Remove punctuation at the end
    cleaned = cleaned.rstrip("?!.,;:")

    # Strip noise phrases (longest first to avoid partial matches)
    for phrase in sorted(_NOISE_PHRASES, key=len, reverse=True):
        cleaned = cleaned.replace(phrase, "")

    # Remove filler words that survive
    cleaned = cleaned.replace(" it ", " ").replace(" it", "")

    # Collapse whitespace
    cleaned = " ".join(cleaned.split()).strip()

    # Remove leading/trailing commas and conjunctions
    cleaned = cleaned.strip(",- ")

    # If cleaning removed everything, fall back to topic
    if not cleaned or len(cleaned) < 3:
        return topic

    return cleaned


def _quote_topic(topic: str) -> str:
    """Wrap multi-word topics in quotes for better search disambiguation.

    'linear algebra' → '"linear algebra"' so search engines treat it
    as a phrase and don't return results about 'Linear' the product.
    Single-word topics are returned as-is.
    """
    if " " in topic.strip():
        return f'"{ topic.strip()}"'
    return topic.strip()


# Domains that almost never contain educational content
_NOISE_DOMAINS = {
    # Social media
    "instagram.com", "tiktok.com", "facebook.com", "twitter.com",
    "x.com", "pinterest.com", "snapchat.com", "threads.net",
    "linkedin.com",
    # App stores
    "play.google.com", "apps.apple.com",
    # Product / SaaS pages (common false positives)
    "linear.app", "notion.so", "figma.com", "slack.com",
    # Dictionaries / translation (common for short topic names)
    "rae.es", "wordreference.com", "linguee.com", "deepl.com",
    "translate.google.com", "dictionary.com", "merriam-webster.com",
    # Shopping
    "amazon.com", "ebay.com", "etsy.com", "walmart.com",
    "alibaba.com", "aliexpress.com",
    # News / entertainment (rarely educational for study topics)
    "imdb.com", "yelp.com", "tripadvisor.com", "buzzfeed.com",
    "netflix.com", "spotify.com", "twitch.tv",
    # Lifestyle / non-academic
    "allrecipes.com", "food.com", "epicurious.com",
    "webmd.com", "healthline.com",  # medical, not academic
}

# Domains that are strong educational signals — results from these
# pass the relevance filter more easily
_EDUCATIONAL_DOMAINS = {
    # Academic / research
    "arxiv.org", "scholar.google.com", "paperswithcode.com",
    "semanticscholar.org", "researchgate.net",
    # MOOCs / educational platforms
    "coursera.org", "edx.org", "khanacademy.org", "ocw.mit.edu",
    "freecodecamp.org", "brilliant.org", "openstax.org",
    "udacity.com", "codecademy.com", "datacamp.com",
    # Community / Q&A
    "reddit.com", "stackoverflow.com", "stackexchange.com",
    "math.stackexchange.com", "cs.stackexchange.com",
    "physics.stackexchange.com",
    # Reference / tutorial
    "wikipedia.org", "mathworld.wolfram.com", "geeksforgeeks.org",
    "towardsdatascience.com", "medium.com", "realpython.com",
    "w3schools.com", "tutorialspoint.com",
    # Video (with title validation)
    "youtube.com",
    # Code / tools
    "github.com", "docs.python.org", "developer.mozilla.org",
    "scikit-learn.org", "pytorch.org", "tensorflow.org",
    "huggingface.co", "numpy.org", "scipy.org",
}


def _filter_relevant_results(results: list[dict], topic: str, concept: str) -> list[dict]:
    """Remove search results that are obviously off-topic.

    Uses a multi-signal approach:
    1. Block known noise domains (social media, product pages, dictionaries)
    2. YouTube-specific validation: require topic keywords in video TITLE
    3. Auto-pass results from known educational domains
    4. For remaining results, require keyword overlap — at least 2 keywords
       for multi-word topics, 1 for single-word topics
    """
    keywords = set()
    for term in (topic.lower().split() + concept.lower().split()):
        if len(term) > 2:  # Skip tiny words like 'is', 'to', 'a'
            keywords.add(term)

    # For multi-word topics, require stricter matching
    min_keyword_hits = 2 if len(keywords) >= 2 else 1

    filtered = []
    for r in results:
        url = r.get("url", "").lower()
        title = r.get("title", "").lower()
        snippet = r.get("snippet", "").lower()

        # Skip known noise domains
        if any(domain in url for domain in _NOISE_DOMAINS):
            logger.debug("Filtered out noise domain: %s", url)
            continue

        text = f"{title} {snippet}"
        keyword_hits = sum(1 for kw in keywords if kw in text)

        # YouTube-specific: require keyword in the VIDEO TITLE (not just snippet)
        # This prevents irrelevant YouTube channels from leaking through
        is_youtube = "youtube.com" in url or "youtu.be" in url
        if is_youtube:
            title_hits = sum(1 for kw in keywords if kw in title)
            if title_hits < 1:
                logger.debug(
                    "Filtered YouTube (no topic in title): %s", title,
                )
                continue

        is_educational = any(domain in url for domain in _EDUCATIONAL_DOMAINS)

        if is_educational and keyword_hits >= 1:
            filtered.append(r)
        elif keyword_hits >= min_keyword_hits:
            filtered.append(r)
        else:
            logger.debug(
                "Filtered out irrelevant result (%d/%d keywords): %s",
                keyword_hits, min_keyword_hits, r.get('title', ''),
            )

    return filtered


def _search_academic_resources(topic: str, concept: str = "") -> tuple[list[dict], list[dict]]:
    """Search for academic papers and books related to a topic.

    Performs targeted searches for arxiv papers, seminal research,
    and recommended textbooks. Returns two lists: (papers, books).
    Results are relevance-filtered — only returns non-empty lists
    when genuinely relevant results are found.

    Args:
        topic: The broad study topic (e.g. "machine learning").
        concept: The specific concept being discussed (e.g. "eigenvalues").
            If empty, searches at the topic level.

    Returns:
        Tuple of (papers, books) where each is a list of search result dicts.
    """
    quoted_topic = _quote_topic(topic)
    search_term = f"{quoted_topic} {concept}".strip() if concept else quoted_topic

    # Search for papers — use specific academic terms
    paper_queries = [
        f"{search_term} research paper site:arxiv.org",
        f"{search_term} seminal foundational paper survey",
    ]
    paper_results = []
    seen_urls = set()
    for q in paper_queries:
        for r in raw_web_search(q, max_results=3):
            if r.get("url") and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                paper_results.append(r)

    # Search for books — use specific book terms
    book_queries = [
        f"{search_term} best textbook recommended",
        f"{search_term} must read book site:reddit.com",
    ]
    book_results = []
    for q in book_queries:
        for r in raw_web_search(q, max_results=3):
            if r.get("url") and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                book_results.append(r)

    # Filter for relevance
    papers = _filter_relevant_results(paper_results, topic, concept or topic)
    books = _filter_relevant_results(book_results, topic, concept or topic)

    logger.info(
        "Academic search for '%s': %d papers, %d books",
        search_term, len(papers), len(books),
    )

    return papers, books


def _format_academic_section(papers: list[dict], books: list[dict]) -> str:
    """Format academic papers and books as a tagged section for the LLM.

    Returns an empty string if no relevant papers or books were found,
    so the section is omitted entirely from the prompt.
    """
    if not papers and not books:
        return ""

    sections = []

    if papers:
        paper_docs = "\n\n".join(
            f'<paper id="p{i+1}">\n'
            f"  <title>{r['title']}</title>\n"
            f"  <url>{r['url']}</url>\n"
            f"  <summary>{r['snippet']}</summary>\n"
            f"</paper>"
            for i, r in enumerate(papers)
        )
        sections.append(f"ACADEMIC PAPERS:\n{paper_docs}")

    if books:
        book_docs = "\n\n".join(
            f'<book id="b{i+1}">\n'
            f"  <title>{r['title']}</title>\n"
            f"  <url>{r['url']}</url>\n"
            f"  <summary>{r['snippet']}</summary>\n"
            f"</book>"
            for i, r in enumerate(books)
        )
        sections.append(f"RECOMMENDED BOOKS:\n{book_docs}")

    return "\n\n".join(sections)


# ══════════════════════════════════════════════════════════════════════
# NODE: Summarize — Compress old messages to save tokens
# ══════════════════════════════════════════════════════════════════════


def summarize_node(state: dict) -> dict:
    """Compress older conversation messages into a summary.

    This is a key token optimization pattern. When the conversation
    exceeds MAX_MESSAGES_BEFORE_SUMMARY, older messages are condensed
    into a 2-3 sentence summary stored in state['summary']. The raw
    messages are then trimmed.

    This means the LLM sees: system prompt + summary + recent messages
    instead of the entire conversation history.
    """
    messages = state.get("messages", [])
    existing_summary = state.get("summary", "")

    # Only summarize if conversation is getting long
    # Count non-system messages
    conversation_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(conversation_msgs) <= MAX_MESSAGES_BEFORE_SUMMARY:
        return {}  # No changes needed

    llm = _get_llm()

    # Build summarization prompt
    summary_input = f"Existing summary: {existing_summary}\n\n" if existing_summary else ""
    # Take the older messages (everything except the last few)
    older_msgs = conversation_msgs[:-6]  # Keep last 6 verbatim
    if not older_msgs:
        return {}

    conversation_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in older_msgs
        if hasattr(m, "content") and m.content
    )

    summary_prompt = [
        SystemMessage(content=SUMMARIZE_PROMPT),
        HumanMessage(content=f"{summary_input}Conversation to summarize:\n{conversation_text}"),
    ]

    response = llm.invoke(summary_prompt)
    new_summary = response.content

    # Trim the messages — keep only recent ones
    trimmed = trim_conversation(messages, max_messages=8)

    logger.info("Summarized %d old messages into summary", len(older_msgs))

    return {
        "summary": new_summary,
        "messages": trimmed,
    }


# ══════════════════════════════════════════════════════════════════════
# NODE: Router — Classify user intent with structured output
# ══════════════════════════════════════════════════════════════════════


def router_node(state: dict) -> dict:
    """Classify the user's intent using structured output.

    Uses the RouterDecision Pydantic schema to force the LLM into
    a predictable classification format with confidence scoring.
    Low confidence triggers a "clarification" response instead of
    guessing wrong.
    """
    messages = state.get("messages", [])
    current_topic = state.get("current_topic", "")
    summary = state.get("summary", "")

    if not messages:
        return {"intent": "general_chat", "current_topic": "general"}

    # Get the last user message
    last_msg = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_msg = m.content
            break

    if not last_msg:
        return {"intent": "general_chat", "current_topic": current_topic or "general"}

    llm = _get_llm()

    # Build context for the router
    context = ""
    if summary:
        context += f"Conversation context: {summary}\n"
    if current_topic:
        context += f"Current topic: {current_topic}\n"

    router_messages = [
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(
            content=f"{context}\nUser message to classify: {last_msg}"
        ),
    ]

    # Use structured output for reliable classification
    try:
        structured_llm = llm.with_structured_output(RouterDecision)
        decision: RouterDecision = structured_llm.invoke(router_messages)

        logger.info(
            "Router: intent=%s, topic=%s, confidence=%.2f",
            decision.intent,
            decision.topic,
            decision.confidence,
        )

        return {
            "intent": decision.intent,
            "current_topic": decision.topic,
        }
    except Exception as e:
        logger.warning("Structured routing failed (%s), falling back to general", e)
        return {
            "intent": "general_chat",
            "current_topic": current_topic or "general",
        }


# ══════════════════════════════════════════════════════════════════════
# NODE: Resource Finder — Search web + Reddit for learning resources
# ══════════════════════════════════════════════════════════════════════


def resource_finder_node(state: dict) -> dict:
    """Search for learning resources using multiple targeted queries.

    Performs up to MAX_SEARCH_QUERIES searches across general web
    and Reddit. Results are tagged with document IDs for source
    grounding — the LLM can only cite URLs that appear in these
    tagged documents.
    """
    topic = state.get("current_topic", "")
    messages = state.get("messages", [])
    summary = state.get("summary", "")
    topics_covered = state.get("topics_covered", [])

    if not topic or topic == "general":
        return {
            "response_draft": "I'd be happy to find resources! Could you tell me what topic you'd like to learn about?",
            "search_results": [],
        }

    # Perform multi-source search across platforms (YouTube, Reddit, MOOCs, etc.)
    all_results = _multi_source_search(topic)

    # Filter out irrelevant results (including YouTube title validation)
    all_results = _filter_relevant_results(all_results, topic, topic)

    if not all_results:
        return {
            "response_draft": (
                f"I couldn't find search results for **{topic}** right now. "
                "This might be a temporary issue with the search service. "
                "Please try again in a moment, or try rephrasing your topic."
            ),
            "search_results": [],
        }

    # Format results as tagged documents for source grounding
    tagged_docs = _format_tagged_docs(all_results)

    # Search for academic papers and books (supplementary)
    papers, books = _search_academic_resources(topic)
    academic_section = _format_academic_section(papers, books)

    # Merge all results for verification
    combined_results = all_results + papers + books

    llm = _get_llm()

    # Build the prompt with grounding context
    context = ""
    if summary:
        context += f"Previous context: {summary}\n\n"

    prompt_content = (
        f"{context}"
        f"The student wants to learn about: **{topic}**\n\n"
        f"Here are the search results. ONLY use URLs from these documents:\n\n"
        f"{tagged_docs}"
    )

    if academic_section:
        prompt_content += (
            f"\n\nThe following academic papers and books were also found. "
            f"Include a '📄 Recommended Reading' section at the end ONLY if "
            f"these are genuinely relevant to the topic. "
            f"If they are NOT relevant, do NOT include this section at all "
            f"— do not even mention that papers were provided:\n\n"
            f"{academic_section}"
        )

    resource_messages = [
        SystemMessage(content=RESOURCE_FINDER_PROMPT),
        HumanMessage(content=prompt_content),
    ]

    response = llm.invoke(resource_messages)

    return {
        "response_draft": response.content,
        "search_results": combined_results,
        "topics_covered": _track_topic(topic, topics_covered),
    }


# ══════════════════════════════════════════════════════════════════════
# NODE: Study Planner — Generate structured study plans
# ══════════════════════════════════════════════════════════════════════


def study_planner_node(state: dict) -> dict:
    """Generate a structured study plan with timeline and resources.

    First searches for resources, then uses those results to build
    a comprehensive plan with phases, milestones, and self-assessment
    checkpoints. The plan is also saved to a local markdown file.
    """
    topic = state.get("current_topic", "")
    messages = state.get("messages", [])
    summary = state.get("summary", "")
    topics_covered = state.get("topics_covered", [])

    if not topic or topic == "general":
        return {
            "response_draft": "I'd love to create a study plan! What topic would you like to study? And how much time can you dedicate per week?",
            "search_results": [],
        }

    # Multi-source search across platforms
    all_results = _multi_source_search(topic, context="roadmap curriculum")

    # Filter out irrelevant results
    all_results = _filter_relevant_results(all_results, topic, topic)

    # Format as tagged documents
    tagged_docs = _format_tagged_docs(all_results)

    # Search for academic papers and books (supplementary)
    papers, books = _search_academic_resources(topic)
    academic_section = _format_academic_section(papers, books)

    # Merge all results for verification
    combined_results = all_results + papers + books

    llm = _get_llm()

    # Extract any time preferences from the user's message
    last_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_msg = m.content
            break

    context = ""
    if summary:
        context += f"Previous context: {summary}\n\n"

    prompt_content = (
        f"{context}"
        f"Student's request: {last_msg}\n"
        f"Topic: **{topic}**\n\n"
        f"Available resources (ONLY use URLs from these documents):\n\n"
        f"{tagged_docs}"
    )

    if academic_section:
        prompt_content += (
            f"\n\nThe following academic papers and books were also found. "
            f"If relevant, include a '📄 Recommended Reading' section in the "
            f"plan suggesting these as supplementary materials. "
            f"If they are NOT relevant, do NOT include this section at all "
            f"— do not even mention that papers were provided:\n\n"
            f"{academic_section}"
        )

    planner_messages = [
        SystemMessage(content=STUDY_PLANNER_PROMPT),
        HumanMessage(content=prompt_content),
    ]

    response = llm.invoke(planner_messages)

    # Save the study plan to a file
    try:
        save_result = save_study_plan.invoke(
            {"topic": topic, "content": response.content}
        )
        logger.info(save_result)
        saved_note = f"\n\n💾 *{save_result}*"
    except Exception as e:
        logger.error("Failed to save study plan: %s", e)
        saved_note = ""

    return {
        "response_draft": response.content + saved_note,
        "search_results": combined_results,
        "topics_covered": _track_topic(topic, topics_covered),
    }


# ══════════════════════════════════════════════════════════════════════
# NODE: Professor — Q&A with web-sourced citations
# ══════════════════════════════════════════════════════════════════════


def professor_node(state: dict) -> dict:
    """Answer questions and explain concepts with web-sourced citations.

    Searches the web for supporting sources using cleaned search queries
    (not the raw user message). Filters out irrelevant results. Falls back
    gracefully to general knowledge when search results are poor.
    """
    messages = state.get("messages", [])
    summary = state.get("summary", "")
    topic = state.get("current_topic", "")
    topics_covered = state.get("topics_covered", [])

    # Get the student's question
    last_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_msg = m.content
            break

    # Extract the core concept from the conversational question
    concept = _clean_search_query(last_msg, topic)
    logger.info("Professor search concept: '%s' (from: '%s')", concept, last_msg[:80])

    # Build targeted search queries — quoted topic for disambiguation
    qt = _quote_topic(topic)
    queries = [
        f"{qt} {concept} explanation tutorial",
        f"{qt} {concept} site:reddit.com",
        f"{qt} {concept} guide examples",
    ]

    all_results = _perform_search(topic, queries)

    # Filter out obviously irrelevant results (Instagram, app stores, etc.)
    relevant_results = _filter_relevant_results(all_results, topic, concept)
    logger.info(
        "Professor search: %d raw results → %d relevant",
        len(all_results), len(relevant_results),
    )

    tagged_docs = _format_tagged_docs(relevant_results) if relevant_results else ""

    # Search for academic papers and books (supplementary)
    papers, books = _search_academic_resources(topic, concept)
    academic_section = _format_academic_section(papers, books)

    # Merge all results for verification
    combined_results = relevant_results + papers + books

    llm = _get_llm()

    # Build the professor's context
    system_content = PROFESSOR_PROMPT
    if summary:
        system_content += f"\n\nConversation context: {summary}"
    if topic:
        system_content += f"\nCurrent study topic: {topic}"

    difficulty = state.get("difficulty_level", "beginner")
    system_content += f"\nStudent's current level: {difficulty}"

    if topics_covered:
        system_content += f"\nTopics already covered: {', '.join(topics_covered)}"

    # Use the recent conversation for context
    recent_msgs = trim_conversation(messages, max_messages=8)
    professor_messages = [SystemMessage(content=system_content)] + [
        m for m in recent_msgs if not isinstance(m, SystemMessage)
    ]

    # Build grounding context with academic supplements
    academic_instruction = ""
    if academic_section:
        academic_instruction = (
            f"\n\nThe following academic papers and books were also found. "
            f"If they are genuinely relevant to this specific explanation, "
            f"mention them at the end as '📄 Recommended Reading'. "
            f"If they are NOT relevant, do NOT include a Recommended Reading "
            f"section at all — do not even mention that papers were provided "
            f"or that none were relevant. Simply omit it entirely:\n\n"
            f"{academic_section}"
        )

    # Inject search results as grounding context, or tell the LLM to use knowledge
    if tagged_docs:
        professor_messages.append(
            HumanMessage(
                content=(
                    f"Here are relevant web sources to support your answer. "
                    f"Cite them using [title](URL) format where applicable:\n\n"
                    f"{tagged_docs}"
                    f"{academic_instruction}\n\n"
                    f"Now answer the student's question: {last_msg}"
                )
            )
        )
    else:
        professor_messages.append(
            HumanMessage(
                content=(
                    f"No relevant web sources were found for this question. "
                    f"Give a thorough, high-quality explanation from your knowledge. "
                    f"Do NOT apologize about missing sources — just teach the concept well. "
                    f"If you think the student should verify specific claims, suggest "
                    f"they search for a specific term."
                    f"{academic_instruction}\n\n"
                    f"Answer the student's question: {last_msg}"
                )
            )
        )

    response = llm.invoke(professor_messages)

    return {
        "response_draft": response.content,
        "search_results": combined_results,
        "topics_covered": _track_topic(topic, topics_covered),
    }


# ══════════════════════════════════════════════════════════════════════
# NODE: Quiz Master — Generate quizzes for active recall
# ══════════════════════════════════════════════════════════════════════


def quiz_master_node(state: dict) -> dict:
    """Generate interactive quizzes to reinforce learning.

    Creates multiple-choice questions based on the current topic,
    grounded in web search results for factual accuracy. Adapts
    difficulty based on the student's tracked level.
    """
    topic = state.get("current_topic", "")
    messages = state.get("messages", [])
    summary = state.get("summary", "")
    difficulty = state.get("difficulty_level", "beginner")
    topics_covered = state.get("topics_covered", [])

    if not topic or topic == "general":
        return {
            "response_draft": (
                "I'd love to quiz you! What topic should I test you on? "
                "You can say something like *\"Quiz me on Python basics\"* "
                "or *\"Test my understanding of gradient descent\"*."
            ),
            "search_results": [],
        }

    # Search for factual content to base questions on
    qt = _quote_topic(topic)
    queries = [
        f"{qt} key concepts explained",
        f"{qt} common interview questions answers",
        f"{qt} quiz practice questions site:reddit.com",
        f"{qt} beginner mistakes misconceptions",
    ]

    all_results = _perform_search(topic, queries)
    relevant_results = _filter_relevant_results(all_results, topic, topic)
    tagged_docs = _format_tagged_docs(relevant_results) if relevant_results else ""

    llm = _get_llm()

    # Get user's specific quiz request if any
    last_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_msg = m.content
            break

    # Build quiz context
    context = ""
    if summary:
        context += f"Previous context: {summary}\n\n"
    if topics_covered:
        context += f"Topics the student has covered: {', '.join(topics_covered)}\n"

    quiz_messages = [
        SystemMessage(content=QUIZ_MASTER_PROMPT),
        HumanMessage(
            content=(
                f"{context}"
                f"Student's request: {last_msg}\n"
                f"Topic: **{topic}**\n"
                f"Student difficulty level: **{difficulty}**\n\n"
                f"Use these search results to create factually grounded questions. "
                f"Include source URLs where applicable:\n\n"
                f"{tagged_docs}"
            )
        ),
    ]

    response = llm.invoke(quiz_messages)

    return {
        "response_draft": response.content,
        "search_results": relevant_results,
    }


# ══════════════════════════════════════════════════════════════════════
# NODE: General Chat — Handle greetings and meta-questions
# ══════════════════════════════════════════════════════════════════════


def general_chat_node(state: dict) -> dict:
    """Handle general conversation, greetings, and meta-questions.

    This is the fallback node for messages that don't fit the other
    categories. It provides friendly responses and guides the user
    toward the agent's academic capabilities.
    """
    messages = state.get("messages", [])
    summary = state.get("summary", "")
    topics_covered = state.get("topics_covered", [])

    llm = _get_llm()

    system_content = (
        "You are a friendly ACADEMIC study partner AI. You help students learn "
        "university-level subjects like math, science, computer science, "
        "languages, and other academic disciplines.\n\n"
        "If the user greets you or asks what you can do, explain your capabilities:\n"
        "1. 🔍 Find learning resources (courses, books, videos, tutorials) with verified links\n"
        "2. 📅 Create structured study plans with timelines and milestones\n"
        "3. 🎓 Answer questions and explain academic concepts with supporting sources\n"
        "4. 🧠 Quiz you on any academic topic to reinforce learning through active recall\n\n"
        "IMPORTANT: You ONLY help with academic/educational topics. If asked about "
        "non-academic things (cooking, entertainment, fitness, etc.), politely redirect "
        "to academic study.\n\n"
        "Be warm, encouraging, and concise. If the student has been studying, "
        "proactively suggest a quiz or next topic to explore."
    )
    if summary:
        system_content += f"\n\nConversation context: {summary}"
    if topics_covered:
        system_content += f"\nTopics covered so far: {', '.join(topics_covered)}"

    recent_msgs = trim_conversation(messages, max_messages=6)
    chat_messages = [SystemMessage(content=system_content)] + [
        m for m in recent_msgs if not isinstance(m, SystemMessage)
    ]

    response = llm.invoke(chat_messages)

    return {
        "response_draft": response.content,
        "search_results": [],
    }


# ══════════════════════════════════════════════════════════════════════
# NODE: Off-Topic — Politely refuse non-academic queries
# ══════════════════════════════════════════════════════════════════════


def off_topic_node(state: dict) -> dict:
    """Handle non-academic queries by politely refusing and redirecting.

    Uses a lightweight LLM call to generate a personalized refusal
    that acknowledges what the user asked about and suggests related
    academic topics they could explore instead.
    """
    messages = state.get("messages", [])

    # Get the user's message for context
    last_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_msg = m.content
            break

    llm = _get_llm()

    off_topic_messages = [
        SystemMessage(content=OFF_TOPIC_PROMPT),
        HumanMessage(content=f"The user said: {last_msg}"),
    ]

    response = llm.invoke(off_topic_messages)

    return {
        "response_draft": response.content,
        "search_results": [],
    }


# ══════════════════════════════════════════════════════════════════════
# NODE: Verify — Quality gate before user sees the response
# ══════════════════════════════════════════════════════════════════════


def verify_node(state: dict) -> dict:
    """Verification quality gate — runs before the response reaches the user.

    Performs rule-based citation checking (zero LLM cost) to catch
    fabricated URLs. Strips invalid content and adds disclaimers
    rather than regenerating (to avoid the over-correction trap).
    """
    draft = state.get("response_draft", "")
    search_results = state.get("search_results", [])

    if not draft:
        return {"messages": [AIMessage(content="I'm not sure how to help with that. Could you rephrase your question?")]}

    # Run citation verification
    result = verify_citations(draft, search_results)

    if result["is_valid"]:
        # All good — pass through
        final_response = draft
    else:
        # Issues found — use cleaned response + disclaimer
        final_response = add_verification_disclaimer(
            result["cleaned_response"], result["issues"]
        )

    return {"messages": [AIMessage(content=final_response)]}
