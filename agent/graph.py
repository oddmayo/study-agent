"""LangGraph assembly — the agent's state machine.

This module wires all the nodes together into a StateGraph:

  START → summarize → router → [conditional] → specialist → verify → END

The graph is compiled with a SQLite checkpointer for persistence,
meaning conversation state survives app restarts.
"""

import logging

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    summarize_node,
    router_node,
    resource_finder_node,
    study_planner_node,
    professor_node,
    general_chat_node,
    verify_node,
)
from agent.memory import get_checkpointer

logger = logging.getLogger(__name__)


def _route_by_intent(state: dict) -> str:
    """Conditional edge: route to the appropriate specialist node
    based on the router's intent classification.

    Returns the name of the next node to execute.
    """
    intent = state.get("intent", "general_chat")

    routing_map = {
        "search_resources": "resource_finder",
        "create_plan": "study_planner",
        "ask_question": "professor",
        "general_chat": "general_chat",
    }

    next_node = routing_map.get(intent, "general_chat")
    logger.info("Routing to: %s (intent: %s)", next_node, intent)
    return next_node


def build_graph():
    """Build and compile the study partner agent graph.

    Graph structure:
        __start__
            │
            ▼
        summarize ──→ router
                        │
                        ├──→ resource_finder ──→ verify ──→ __end__
                        ├──→ study_planner ───→ verify ──→ __end__
                        ├──→ professor ───────→ verify ──→ __end__
                        └──→ general_chat ───→ verify ──→ __end__

    Returns:
        A compiled LangGraph with SQLite persistence.
    """
    # Initialize the graph with our state schema
    builder = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────
    builder.add_node("summarize", summarize_node)
    builder.add_node("router", router_node)
    builder.add_node("resource_finder", resource_finder_node)
    builder.add_node("study_planner", study_planner_node)
    builder.add_node("professor", professor_node)
    builder.add_node("general_chat", general_chat_node)
    builder.add_node("verify", verify_node)

    # ── Add edges ──────────────────────────────────────────────────

    # Entry point: always start with summarization
    builder.set_entry_point("summarize")

    # Summarize → Router (always)
    builder.add_edge("summarize", "router")

    # Router → Specialist (conditional based on intent)
    builder.add_conditional_edges(
        "router",
        _route_by_intent,
        {
            "resource_finder": "resource_finder",
            "study_planner": "study_planner",
            "professor": "professor",
            "general_chat": "general_chat",
        },
    )

    # All specialists → Verify
    builder.add_edge("resource_finder", "verify")
    builder.add_edge("study_planner", "verify")
    builder.add_edge("professor", "verify")
    builder.add_edge("general_chat", "verify")

    # Verify → End
    builder.add_edge("verify", END)

    # ── Compile with persistence ───────────────────────────────────
    checkpointer = get_checkpointer()

    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Agent graph compiled successfully")
    return graph
