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
- Few-shot prompting (consistent output quality)
"""

import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent.prompts import (
    ROUTER_PROMPT,
    RESOURCE_FINDER_PROMPT,
    STUDY_PLANNER_PROMPT,
    PROFESSOR_PROMPT,
    SUMMARIZE_PROMPT,
)
from agent.schemas import RouterDecision
from agent.tools import raw_web_search, save_study_plan
from agent.memory import trim_conversation
from agent.verification import verify_citations, add_verification_disclaimer
from agent.llm import get_llm

logger = logging.getLogger(__name__)

# ── Execution guardrails ───────────────────────────────────────────────
MAX_SEARCH_QUERIES = 4  # Max search queries per turn
MAX_MESSAGES_BEFORE_SUMMARY = 10  # Trigger summarization at this count


# ── Cached LLM instance ───────────────────────────────────────────────
_llm = None


def _get_llm():
    """Get or create the cached LLM instance."""
    global _llm
    if _llm is None:
        _llm = get_llm(streaming=True)
    return _llm


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
# NODE: Resource Finder — Search web + Reddit + experts
# ══════════════════════════════════════════════════════════════════════


def resource_finder_node(state: dict) -> dict:
    """Search for learning resources using multiple targeted queries.

    Performs up to MAX_SEARCH_QUERIES searches across general web,
    Reddit, and expert-focused queries. Results are tagged with document
    IDs for source grounding — the LLM can only cite URLs that appear
    in these tagged documents.
    """
    topic = state.get("current_topic", "")
    messages = state.get("messages", [])
    summary = state.get("summary", "")

    if not topic or topic == "general":
        return {
            "response_draft": "I'd be happy to find resources! Could you tell me what topic you'd like to learn about?",
            "search_results": [],
        }

    # Perform targeted searches
    queries = [
        f"{topic} best free online courses tutorials 2025 2026",
        f"{topic} best free resources recommendations site:reddit.com",
        f"{topic} most influential professors authors YouTube educators",
        f"{topic} best books for beginners free PDF",
    ]

    all_results = []
    seen_urls = set()
    for query in queries[:MAX_SEARCH_QUERIES]:
        results = raw_web_search(query, max_results=4)
        for r in results:
            if r.get("url") and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                all_results.append(r)

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
    tagged_docs = "\n\n".join(
        f'<doc id="{i+1}">\n'
        f"  <title>{r['title']}</title>\n"
        f"  <url>{r['url']}</url>\n"
        f"  <content>{r['snippet']}</content>\n"
        f"</doc>"
        for i, r in enumerate(all_results)
    )

    llm = _get_llm()

    # Build the prompt with grounding context
    context = ""
    if summary:
        context += f"Previous context: {summary}\n\n"

    resource_messages = [
        SystemMessage(content=RESOURCE_FINDER_PROMPT),
        HumanMessage(
            content=(
                f"{context}"
                f"The student wants to learn about: **{topic}**\n\n"
                f"Here are the search results. ONLY use URLs from these documents:\n\n"
                f"{tagged_docs}"
            )
        ),
    ]

    response = llm.invoke(resource_messages)

    return {
        "response_draft": response.content,
        "search_results": all_results,
    }


# ══════════════════════════════════════════════════════════════════════
# NODE: Study Planner — Generate structured study plans
# ══════════════════════════════════════════════════════════════════════


def study_planner_node(state: dict) -> dict:
    """Generate a structured study plan with timeline and resources.

    First searches for resources, then uses those results to build
    a comprehensive plan with phases, milestones, and expert recommendations.
    The plan is also saved to a local markdown file.
    """
    topic = state.get("current_topic", "")
    messages = state.get("messages", [])
    summary = state.get("summary", "")

    if not topic or topic == "general":
        return {
            "response_draft": "I'd love to create a study plan! What topic would you like to study? And how much time can you dedicate per week?",
            "search_results": [],
        }

    # Search for resources to include in the plan
    queries = [
        f"{topic} complete learning roadmap beginner to advanced 2025 2026",
        f"{topic} best free courses structured curriculum",
        f"{topic} study plan recommendations site:reddit.com",
        f"{topic} influential professors educators to follow",
    ]

    all_results = []
    seen_urls = set()
    for query in queries[:MAX_SEARCH_QUERIES]:
        results = raw_web_search(query, max_results=4)
        for r in results:
            if r.get("url") and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                all_results.append(r)

    # Format as tagged documents
    tagged_docs = "\n\n".join(
        f'<doc id="{i+1}">\n'
        f"  <title>{r['title']}</title>\n"
        f"  <url>{r['url']}</url>\n"
        f"  <content>{r['snippet']}</content>\n"
        f"</doc>"
        for i, r in enumerate(all_results)
    )

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

    planner_messages = [
        SystemMessage(content=STUDY_PLANNER_PROMPT),
        HumanMessage(
            content=(
                f"{context}"
                f"Student's request: {last_msg}\n"
                f"Topic: **{topic}**\n\n"
                f"Available resources (ONLY use URLs from these documents):\n\n"
                f"{tagged_docs}"
            )
        ),
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
        "search_results": all_results,
    }


# ══════════════════════════════════════════════════════════════════════
# NODE: Professor — Q&A and teaching
# ══════════════════════════════════════════════════════════════════════


def professor_node(state: dict) -> dict:
    """Answer questions and explain concepts in professor mode.

    Uses the full conversation context (summary + recent messages)
    to provide contextual, pedagogical responses. The professor
    prompt enforces Socratic teaching and honest uncertainty.
    """
    messages = state.get("messages", [])
    summary = state.get("summary", "")
    topic = state.get("current_topic", "")

    llm = _get_llm()

    # Build the professor's context
    system_content = PROFESSOR_PROMPT
    if summary:
        system_content += f"\n\nConversation context: {summary}"
    if topic:
        system_content += f"\nCurrent study topic: {topic}"

    # Use the recent conversation for context
    recent_msgs = trim_conversation(messages, max_messages=8)
    professor_messages = [SystemMessage(content=system_content)] + [
        m for m in recent_msgs if not isinstance(m, SystemMessage)
    ]

    response = llm.invoke(professor_messages)

    return {
        "response_draft": response.content,
        "search_results": [],  # Professor doesn't search — uses knowledge
    }


# ══════════════════════════════════════════════════════════════════════
# NODE: General Chat — Handle greetings and meta-questions
# ══════════════════════════════════════════════════════════════════════


def general_chat_node(state: dict) -> dict:
    """Handle general conversation, greetings, and meta-questions.

    This is the fallback node for messages that don't fit the other
    categories. It provides friendly responses and guides the user
    toward the agent's capabilities.
    """
    messages = state.get("messages", [])
    summary = state.get("summary", "")

    llm = _get_llm()

    system_content = (
        "You are a friendly study partner AI. You help people learn any topic "
        "by finding resources, creating study plans, and answering questions.\n\n"
        "If the user greets you or asks what you can do, explain your capabilities:\n"
        "1. 🔍 Find learning resources (courses, books, videos, tutorials)\n"
        "2. 📅 Create structured study plans with timelines\n"
        "3. 🎓 Answer questions and explain concepts like a professor\n"
        "4. 👥 Recommend experts and creators to follow\n\n"
        "Be warm, encouraging, and concise."
    )
    if summary:
        system_content += f"\n\nConversation context: {summary}"

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
