from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field


AgentName = Literal["researcher", "analyst", "writer", "supervisor"]


class RetrievedDocument(BaseModel):
    """A single piece of evidence returned from Qdrant retrieval."""

    content: str = Field(default="")
    score: float = Field(default=0.0, ge=-1.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchContext(BaseModel):
    """Context extracted by the Researcher agent before analysis."""

    search_query: str = Field(default="")
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    evidence_summary: str = Field(default="")
    missing_information: list[str] = Field(default_factory=list)


class AnalystFindings(BaseModel):
    """Structured summary of the Analyst's evaluation."""

    likely_root_cause: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """LangGraph state passed between supervisor and specialist agents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[AnyMessage] = Field(default_factory=list)
    user_request: str = Field(default="")
    current_agent: AgentName | None = None
    route_reason: str = Field(default="")
    research_context: ResearchContext | None = None
    analyst_findings: AnalystFindings | None = None
    final_answer: str = Field(default="")
    iteration_count: int = Field(default=0, ge=0)
    next_agent: AgentName | None = None
    errors: list[str] = Field(default_factory=list)


class SupervisorDecision(BaseModel):
    """Decision payload produced by the supervisor when choosing the next agent."""

    next_agent: AgentName
    reasoning: str = Field(default="")
    should_finalize: bool = Field(default=False)


def agent_state_update(state: AgentState, key: str, value: object) -> AgentState:
    """Convenience helper to update AgentState while preserving typed access."""
    updated = state.model_copy(deep=True)
    setattr(updated, key, value)
    return updated


AnnotatedState = Annotated[AgentState, "Agent state for the LangGraph workflow"]
