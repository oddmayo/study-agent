"""Agent state schema.

The state is a TypedDict that flows through every node in the LangGraph.
Using `add_messages` as the reducer for the messages field means new
messages are appended rather than replacing the entire list.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state for the study partner agent.

    Attributes:
        messages: Full conversation history. The `add_messages` reducer
            automatically appends new messages and handles deduplication.
        summary: Compressed summary of older messages (for token optimization).
            When the conversation gets long, older messages are summarized
            into this field and then trimmed from `messages`.
        current_topic: The active study topic being discussed. Persists
            across turns so the agent remembers what you're studying.
        intent: The router's classification of the user's latest message.
            One of: search_resources, create_plan, ask_question, take_quiz, general_chat.
        search_results: Raw search results from the most recent web search.
            Stored here so the verification node can cross-reference cited URLs.
        response_draft: The specialist node's response before verification.
            The verification node checks this and either passes it through
            or cleans it before sending to the user.
        difficulty_level: Tracks the student's current level (beginner,
            intermediate, advanced) — adapts explanations and quiz difficulty.
        topics_covered: List of sub-topics the student has explored so far.
            Used by the quiz generator and for suggesting next topics.
    """

    messages: Annotated[list, add_messages]
    summary: str
    current_topic: str
    intent: str
    search_results: list[dict]
    response_draft: str
    difficulty_level: str
    topics_covered: list[str]
