"""Memory management: SQLite persistence and token optimization.

This module handles two critical concerns:
1. Persistent memory via SQLite checkpointer — conversations survive restarts.
2. Token optimization via trim_messages — prevents context window overflow.
"""

import sqlite3
import logging
from pathlib import Path

from langchain_core.messages import (
    trim_messages,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "data" / "memory.sqlite"
MAX_MESSAGES_IN_CONTEXT = 10  # Keep last N messages verbatim


def get_checkpointer() -> SqliteSaver:
    """Create and return a SQLite checkpointer for LangGraph persistence.

    The checkpointer automatically saves the agent's state (messages,
    topic, search results, etc.) after every node transition. When the
    agent restarts, it resumes from the last checkpoint.

    Returns:
        A SqliteSaver instance connected to the local SQLite database.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    logger.info("Memory database at %s", DB_PATH)
    return SqliteSaver(conn)


def trim_conversation(messages: list, max_messages: int = MAX_MESSAGES_IN_CONTEXT) -> list:
    """Trim the conversation history to the last N messages.

    Preserves:
    - The system message (always kept)
    - The most recent `max_messages` messages
    - Ensures the trimmed history starts with a HumanMessage
      (required by most LLMs)

    Args:
        messages: The full message list from state.
        max_messages: Maximum number of messages to keep.

    Returns:
        Trimmed message list.
    """
    if len(messages) <= max_messages:
        return messages

    # Separate system messages from conversation
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    conversation = [m for m in messages if not isinstance(m, SystemMessage)]

    # Keep only the last N conversation messages
    trimmed = conversation[-max_messages:]

    # Ensure we start with a HumanMessage (LLM requirement)
    while trimmed and isinstance(trimmed[0], AIMessage):
        trimmed = trimmed[1:]

    return system_msgs + trimmed
