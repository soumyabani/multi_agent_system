from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.state import AgentState, AnalystFindings, ResearchContext, RetrievedDocument
from src.config import SYSTEM_PROMPTS, settings
from src.tools.embeddings import get_embedding_model
from src.tools.langfuse_tracing import get_langfuse_handler
from src.tools.qdrant_client import QdrantKnowledgeClient

logger = logging.getLogger(__name__)


class SpecialistAgentError(RuntimeError):
    """Raised when a specialist agent fails during instruction or retrieval."""


def _normalize_messages(messages: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for message in messages:
        if message is None:
            continue
        normalized.append(message)
    return normalized


def _append_error(state: AgentState, message: str) -> None:
    if message and message not in state.errors:
        state.errors.append(message)


def _coerce_analyst_findings(raw: Any) -> AnalystFindings:
    if isinstance(raw, AnalystFindings):
        return raw
    if isinstance(raw, dict):
        return AnalystFindings(**raw)
    return AnalystFindings(
        likely_root_cause=str(raw) if raw else "The model could not produce structured findings.",
        confidence=0.0,
        risks=["The model output could not be structured reliably."],
        recommendations=["Review the raw evidence and confirm the diagnosis before taking action."],
        open_questions=["What additional evidence is necessary to validate the conclusion?"],
    )


def researcher_node(state: AgentState) -> AgentState:
    """Gather facts and evidence from the local vector store for the user request."""
    user_request = (state.user_request or "").strip()
    if not user_request:
        raise SpecialistAgentError("Researcher node requires a non-empty user request.")

    state.current_agent = "researcher"
    state.errors = list(state.errors or [])

    try:
        embedding_model = get_embedding_model()
        qdrant_client = QdrantKnowledgeClient()
        qdrant_client.ensure_collection(vector_size=embedding_model.dimension)
        query_vector = embedding_model.embed_query(user_request)
        relevant_results = qdrant_client.search(
            query_embedding=query_vector,
            limit=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        )
    except Exception as exc:  # pragma: no cover - infrastructure dependent
        logger.exception("Researcher retrieval failed: %s", exc)
        relevant_results = []
        _append_error(state, f"retrieval failure: {exc}")

    retrieved_documents = [
        RetrievedDocument(content=result.content, score=result.score, metadata=result.metadata)
        for result in relevant_results
        if result.content and result.content.strip()
    ]

    if retrieved_documents:
        evidence_summary = (
            f"Retrieved {len(retrieved_documents)} document(s) with the highest similarity to the request."
        )
        missing_information: list[str] = []
    else:
        evidence_summary = "No relevant documents were retrieved from the local knowledge base."
        missing_information = [
            "No exact matching incident records were found in the local knowledge base.",
            "The workflow should present the answer with explicit uncertainty.",
        ]

    context = ResearchContext(
        search_query=user_request,
        retrieved_documents=retrieved_documents,
        evidence_summary=evidence_summary,
        missing_information=missing_information,
    )

    state.research_context = context
    state.messages = _normalize_messages(
        [
            *state.messages,
            SystemMessage(content=SYSTEM_PROMPTS["researcher"]),
            HumanMessage(content=user_request),
            AIMessage(content=evidence_summary),
        ]
    )
    return state


def analyst_node(state: AgentState) -> AgentState:
    """Evaluate evidence and translate it into root-cause and risk analysis."""
    research_context = state.research_context or ResearchContext()
    evidence = research_context.retrieved_documents

    state.current_agent = "analyst"
    state.errors = list(state.errors or [])

    if not evidence:
        findings = AnalystFindings(
            likely_root_cause="Insufficient evidence to establish a reliable root cause.",
            confidence=0.0,
            risks=["The available evidence is too weak to support a confident diagnosis."],
            recommendations=["Collect additional system telemetry, logs, or historical incident records before making a final decision."],
            open_questions=["What evidence is missing?"],
        )
        state.analyst_findings = findings
        state.messages = _normalize_messages(
            [
                *state.messages,
                SystemMessage(content=SYSTEM_PROMPTS["analyst"]),
                AIMessage(content="No reliable root cause can be established from the current evidence."),
            ]
        )
        return state

    llm = settings.get_chat_model()
    structured_llm = llm.with_structured_output(AnalystFindings) if hasattr(llm, "with_structured_output") else llm
    evidence_text = "\n".join(
        f"Document {index + 1}:\n{document.content}\nMetadata: {document.metadata}\nScore: {document.score:.4f}"
        for index, document in enumerate(evidence)
    )
    prompt = (
        "Use only the evidence below. Do not invent facts. "
        "Identify likely root cause, estimate confidence from 0 to 1, list risks, make recommendations, and list open questions.\n\n"
        f"User request: {state.user_request}\n\nRetrieved evidence:\n{evidence_text}"
    )

    try:
        response = structured_llm.invoke(prompt)
        findings = _coerce_analyst_findings(response)
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        logger.exception("Analyst LLM failed: %s", exc)
        _append_error(state, f"analyst failure: {exc}")
        findings = AnalystFindings(
            likely_root_cause="The model could not structure the analysis from retrieved evidence.",
            confidence=0.0,
            risks=["Decision quality is limited by missing model output."],
            recommendations=["Review the evidence manually and validate using telemetry before acting."],
            open_questions=["What additional validation is required?"],
        )

    state.analyst_findings = findings
    state.messages = _normalize_messages(
        [
            *state.messages,
            SystemMessage(content=SYSTEM_PROMPTS["analyst"]),
            AIMessage(content=f"Analysis: {findings.likely_root_cause}"),
        ]
    )
    return state


def writer_node(state: AgentState) -> AgentState:
    """Generate the final response using the configured LLM."""
    research_context = state.research_context or ResearchContext()
    findings = state.analyst_findings or AnalystFindings()
    state.current_agent = "writer"

    evidence_text = "\n".join(
        f"- {document.content}\n  score={document.score:.4f}\n  metadata={document.metadata}"
        for document in research_context.retrieved_documents
    ) or "No retrieved evidence available."

    prompt = (
        "You are writing the final answer for a support or incident-response workflow. "
        "Base the answer only on the supplied evidence and analyst findings. If the evidence is weak, say so clearly.\n\n"
        f"User request: {state.user_request}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Analyst findings:\n{findings.model_dump_json(indent=2)}\n\n"
        "Write a clear, professional response with: a short summary, confidence, key risks, recommended actions, and uncertainty notes."
    )

    try:
        llm = settings.get_chat_model()
        response = llm.invoke(prompt)
        state.final_answer = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # pragma: no cover - runtime dependent
        logger.exception("Writer LLM failed: %s", exc)
        _append_error(state, f"writer failure: {exc}")
        state.final_answer = (
            "I could not generate a final answer because the configured model could not produce output. "
            "Please verify the LLM service is running and the model configuration is valid."
        )

    state.messages = _normalize_messages(
        [
            *state.messages,
            SystemMessage(content=SYSTEM_PROMPTS["writer"]),
            AIMessage(content=state.final_answer),
        ]
    )
    return state


def route_specialist_for_request(state: AgentState) -> AgentState:
    """Simple routing wrapper used by the supervisor or tests."""
    if not state.user_request:
        raise SpecialistAgentError("Cannot route without a user request.")

    state.next_agent = "researcher"
    state.current_agent = "supervisor"
    state.route_reason = "Initial routing to the Researcher for evidence gathering."
    return state


def get_specialist_node(agent_name: Literal["researcher", "analyst", "writer"]) -> Any:
    """Return the node function for a named specialist agent."""
    mapping = {
        "researcher": researcher_node,
        "analyst": analyst_node,
        "writer": writer_node,
    }
    return mapping[agent_name]
