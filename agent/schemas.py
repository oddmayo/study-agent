"""Pydantic schemas for structured agent outputs.

Using structured outputs forces the LLM into predictable formats,
making responses verifiable, displayable, and saveable. This is
a key anti-hallucination pattern — the model can't weave fabricated
info into free-form prose when constrained to a schema.
"""

from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    """Structured output for the router node's intent classification."""

    intent: str = Field(
        description="One of: search_resources, create_plan, ask_question, take_quiz, off_topic, general_chat"
    )
    topic: str = Field(
        description="The study topic being discussed, e.g. 'machine learning', 'Japanese'"
    )
    confidence: float = Field(
        description="Confidence in the classification from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )


class Resource(BaseModel):
    """A single learning resource found via web search."""

    title: str = Field(description="Name of the resource")
    url: str = Field(description="URL — must come from search results")
    resource_type: str = Field(
        description="One of: course, book, video, article, tutorial, tool"
    )
    cost: str = Field(description="One of: free, paid, freemium")
    description: str = Field(description="1-2 sentence description of the resource")
    recommended_by: str = Field(
        default="",
        description="Where it was recommended, e.g. 'Reddit r/learnpython'",
    )


class QuizQuestion(BaseModel):
    """A single quiz question for testing knowledge."""

    question: str = Field(description="The quiz question")
    options: list[str] = Field(
        description="4 multiple-choice options labeled A, B, C, D"
    )
    correct_answer: str = Field(
        description="The correct option letter: A, B, C, or D"
    )
    explanation: str = Field(
        description="Brief explanation of why this is the correct answer"
    )
    difficulty: str = Field(
        description="One of: beginner, intermediate, advanced"
    )
    source_url: str = Field(
        default="",
        description="URL that supports the correct answer, if available",
    )


class StudyPhase(BaseModel):
    """A single phase within a study plan."""

    name: str = Field(description="Phase name, e.g. 'Foundation', 'Core Skills'")
    duration_weeks: int = Field(description="How many weeks this phase lasts")
    goals: list[str] = Field(description="Learning objectives for this phase")
    resources: list[str] = Field(
        description="Specific resources to use (titles with URLs)"
    )
    milestones: list[str] = Field(
        description="Checkpoints to verify progress"
    )


class StudyPlan(BaseModel):
    """A complete structured study plan."""

    topic: str = Field(description="The study topic")
    total_weeks: int = Field(description="Total duration in weeks")
    hours_per_week: float = Field(description="Recommended hours per week")
    prerequisites: list[str] = Field(
        default_factory=list,
        description="What the student should know beforehand",
    )
    phases: list[StudyPhase] = Field(description="Ordered list of study phases")
    tips: list[str] = Field(
        default_factory=list,
        description="General tips for studying this topic",
    )


class VerificationResult(BaseModel):
    """Result of the verification node's quality check."""

    is_valid: bool = Field(description="Whether the response passed verification")
    issues: list[str] = Field(
        default_factory=list,
        description="List of issues found (empty if valid)",
    )
    cleaned_response: str = Field(
        default="",
        description="The corrected response (only if issues were found)",
    )
