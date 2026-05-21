"""Chainlit application — the web UI for the study partner agent.

This is the entry point that connects the LangGraph agent to a beautiful
chat interface. It handles:
- Session management (thread IDs for persistent memory)
- Token-by-token streaming (ChatGPT-like typing effect)
- Status indicators (searching, planning, verifying...)
- Markdown rendering for study plans and resource lists
"""

import logging
import uuid

import chainlit as cl
from langchain_core.messages import HumanMessage

from agent.graph import build_graph

# ── Configure logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Build the agent graph once at startup ──────────────────────────────
agent = build_graph()


@cl.on_chat_start
async def on_start():
    """Initialize a new chat session.

    Creates a unique thread ID for this session so that the SQLite
    checkpointer can persist and retrieve conversation history.
    """
    # Generate a persistent thread ID for this session
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)

    logger.info("New session started: %s", thread_id)

    # Send welcome message
    await cl.Message(
        content=(
            "👋 **Welcome to Study Partner!**\n\n"
            "I'm your AI study companion. Here's what I can do:\n\n"
            "- 🔍 **Find Resources** — courses, books, videos, tutorials (prioritizing free ones!)\n"
            "- 📅 **Create Study Plans** — structured roadmaps with timelines\n"
            "- 🎓 **Teach & Explain** — answer questions like a professor\n"
            "- 👥 **Recommend Experts** — professors, creators, and authors to follow\n\n"
            "**Try saying:**\n"
            '- *"Find me free resources to learn machine learning"*\n'
            '- *"Create a 3-month study plan for web development"*\n'
            '- *"Explain gradient descent like I\'m a beginner"*\n'
            '- *"Help me learn Japanese"*\n\n'
            "What would you like to learn today? 📚"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages.

    Streams the agent's response token-by-token for a real-time
    typing effect. Shows status indicators for long-running operations.
    """
    thread_id = cl.user_session.get("thread_id")

    # Configure the graph invocation with the thread ID
    config = {"configurable": {"thread_id": thread_id}}

    # Create the response message (will be streamed into)
    msg = cl.Message(content="")
    await msg.send()

    # Track which step we're on for status updates
    current_step = None

    try:
        # Stream events from the LangGraph
        final_content = ""

        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=message.content)]},
            version="v2",
            config=config,
        ):
            event_type = event.get("event", "")
            event_name = event.get("name", "")

            # ── Status indicators ──────────────────────────────────
            if event_type == "on_chain_start":
                step_name = event_name
                if step_name == "resource_finder" and current_step != "resource_finder":
                    current_step = "resource_finder"
                    async with cl.Step(name="🔍 Searching for resources..."):
                        pass
                elif step_name == "study_planner" and current_step != "study_planner":
                    current_step = "study_planner"
                    async with cl.Step(name="📅 Creating your study plan..."):
                        pass
                elif step_name == "professor" and current_step != "professor":
                    current_step = "professor"
                elif step_name == "verify" and current_step != "verify":
                    current_step = "verify"

            # ── Stream tokens ──────────────────────────────────────
            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", None)
                if chunk and hasattr(chunk, "content") and chunk.content:
                    # Only stream tokens from the final response (verify node)
                    # The verify node produces the AIMessage via direct construction,
                    # so we capture tokens from the last LLM call in specialist nodes
                    parent_ids = event.get("parent_ids", [])
                    tags = event.get("tags", [])

                    # Stream all model output tokens
                    await msg.stream_token(chunk.content)
                    final_content += chunk.content

        # If streaming didn't produce content, get the final state
        if not final_content:
            # Fallback: get the response from the final state
            state = agent.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        msg.content = last_msg.content

        await msg.update()

    except Exception as e:
        logger.error("Error processing message: %s", e, exc_info=True)
        msg.content = (
            "❌ Sorry, I encountered an error processing your message. "
            "Please try again. If the issue persists, check the logs.\n\n"
            f"*Error: {str(e)[:200]}*"
        )
        await msg.update()


@cl.on_chat_resume
async def on_resume(thread: dict):
    """Resume a previous chat session.

    This is called when a user returns to a previous conversation.
    The thread_id is restored so the agent can access its checkpointed state.
    """
    thread_id = thread.get("id", str(uuid.uuid4()))
    cl.user_session.set("thread_id", thread_id)
    logger.info("Session resumed: %s", thread_id)
