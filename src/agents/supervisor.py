from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from src.agents.specialists import analyst_node, researcher_node, writer_node
from src.agents.state import AgentState, SupervisorDecision
from src.tools.langfuse_tracing import get_langfuse_handler

logger = logging.getLogger(__name__)


class SupervisorRuntimeError(RuntimeError):
    """Raised when supervisor routing cannot determine the next action."""


def supervisor_node(state: AgentState) -> AgentState:
    """Initial routing node for the multi-agent graph."""
    if not state.user_request:
        raise SupervisorRuntimeError("User request is required before supervisor routing can start.")

    decision = SupervisorDecision(
        next_agent="researcher",
        reasoning="Starting with the Researcher to gather evidence before analysis or drafting.",
        should_finalize=False,
    )

    state.current_agent = "supervisor"
    state.next_agent = decision.next_agent
    state.route_reason = decision.reasoning
    state.iteration_count = max(state.iteration_count, 0)
    return state


def build_graph() -> StateGraph:
    """Construct the LangGraph workflow connecting the supervisor to specialist nodes."""
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", END)

    return workflow


def run_workflow(user_request: str) -> AgentState:
    """Execute the supervisor workflow for a single user request."""
    if not user_request or not user_request.strip():
        raise SupervisorRuntimeError("A non-empty user request is required.")

    # Initialize Langfuse callback handler
    handler = get_langfuse_handler()
    config = {"callbacks": [handler]} if handler else {}

    graph = build_graph().compile()
    initial_state = AgentState(user_request=user_request, iteration_count=0)
    
    # Run graph with Langfuse tracing active
    result = graph.invoke(initial_state, config=config)
    
    # Flush pending traces to Langfuse backend
    if handler and hasattr(handler, "flush"):
        handler.flush()
    
    if isinstance(result, dict):
        return AgentState(**result)
    return result