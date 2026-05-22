"""Chainlit application — the web UI for the study partner agent.

This is the entry point that connects the LangGraph agent to a beautiful
chat interface. It handles:
- Session management (thread IDs for persistent memory)
- Token-by-token streaming (ChatGPT-like typing effect)
- Status indicators (searching, planning, quizzing, verifying...)
- Markdown rendering for study plans, quizzes, and resource lists
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

# We will build the agent graph dynamically inside the message handler
# so we can use the async checkpointer.


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
            "- 🔍 **Find Resources** — courses, books, videos, tutorials (with verified links!)\n"
            "- 📅 **Create Study Plans** — structured roadmaps with timelines and milestones\n"
            "- 🎓 **Explain Concepts** — answer questions with supporting sources\n"
            "- 🧠 **Quiz You** — test your knowledge with interactive quizzes\n\n"
            "**Try saying:**\n"
            '- *"Find me free resources to learn machine learning"*\n'
            '- *"Create a 3-month study plan for web development"*\n'
            '- *"Explain gradient descent like I\'m a beginner"*\n'
            '- *"Quiz me on Python basics"*\n\n'
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

    # Nodes whose LLM output should be streamed to the user
    STREAMABLE_NODES = {"resource_finder", "study_planner", "professor", "quiz_master", "general_chat"}

    try:
        from agent.memory import get_async_checkpointer
        async with get_async_checkpointer() as checkpointer:
            agent = build_graph(checkpointer=checkpointer)

            # Stream events from the LangGraph
            final_content = ""
            active_node = None

            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=message.content)]},
                version="v2",
                config=config,
            ):
                event_type = event.get("event", "")
                event_name = event.get("name", "")

                # ── Track which node is currently active ───────────────
                if event_type == "on_chain_start":
                    if event_name in STREAMABLE_NODES:
                        active_node = event_name
                    elif event_name in ("router", "summarize", "verify"):
                        active_node = event_name

                    # Show status indicators for long-running nodes
                    if event_name == "resource_finder" and current_step != "resource_finder":
                        current_step = "resource_finder"
                        async with cl.Step(name="🔍 Searching for resources..."):
                            pass
                    elif event_name == "study_planner" and current_step != "study_planner":
                        current_step = "study_planner"
                        async with cl.Step(name="📅 Creating your study plan..."):
                            pass
                    elif event_name == "professor" and current_step != "professor":
                        current_step = "professor"
                        async with cl.Step(name="🎓 Researching your question..."):
                            pass
                    elif event_name == "quiz_master" and current_step != "quiz_master":
                        current_step = "quiz_master"
                        async with cl.Step(name="🧠 Generating quiz questions..."):
                            pass
                    elif event_name == "verify" and current_step != "verify":
                        current_step = "verify"

                elif event_type == "on_chain_end":
                    if event_name == active_node:
                        active_node = None

                # ── Stream tokens only from specialist nodes ───────────
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", None)
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # Only stream from specialist nodes, not router/summarizer
                        if active_node in STREAMABLE_NODES:
                            await msg.stream_token(chunk.content)
                            final_content += chunk.content

            # If streaming didn't produce content, get the final state
            if not final_content:
                state = await agent.aget_state(config)
                if state and state.values:
                    messages = state.values.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "content") and last_msg.content:
                            msg.content = last_msg.content
            else:
                # Check if verification modified the response
                state = await agent.aget_state(config)
                if state and state.values:
                    messages = state.values.get("messages", [])
                    if messages:
                        verified_content = messages[-1].content if hasattr(messages[-1], "content") else ""
                        # If verified response differs (e.g. URLs were stripped),
                        # replace streamed content with the verified version
                        if verified_content and verified_content != final_content:
                            msg.content = verified_content
                            final_content = verified_content

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
