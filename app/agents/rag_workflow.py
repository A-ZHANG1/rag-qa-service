"""LangGraph supervisor + specialist workflow for the RAG "Agent mode".

Architecture (see docs/adr/0005-rag-agent-supervisor-pattern.md):

    rag_specialist -> [route: local docs good enough?]
                         |-> synthesis                  (local docs sufficient)
                         `-> web_specialist -> synthesis (local docs insufficient)

Two independent guards make sure this always terminates with a real answer
instead of hanging or raising -- the core reliability lesson from the
OmniGent multi-agent analysis this design is based on:

1. A hard step cap (``settings.agent_max_steps``) bounds how many specialist
   nodes can run, regardless of how the routing decision would otherwise go.
2. A web_specialist failure (timeout, network error -- see
   ``app.agents.rag_tools.web_search``, which never raises) does not abort
   the request: it sets ``degraded=True`` and synthesis falls back to
   whatever RAG context is available, telling the user honestly what
   happened, instead of the caller getting an exception and nothing else.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.rag_tools import rag_search, web_search
from app.config import get_settings
from app.core.chain import _get_llm
from app.core.telemetry import get_tracer

_logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

_SYNTHESIS_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions using retrieved context.

You may be given context from two sources:
- LOCAL DOCS: retrieved from the local knowledge base
- WEB RESULTS: retrieved from a live web search (only present when the local docs were insufficient)

Answer the user's question using the provided context. If you use information from a
source, make clear in your answer whether it came from the local documents or from the
web. If neither source has enough information, say so honestly rather than guessing.
{degraded_note}"""

_DEGRADED_NOTE = (
    "\n\nIMPORTANT: A web search was needed to fully answer this question, but the web "
    "search is currently unavailable. Answer using only the local documents below, and "
    "clearly tell the user that live web results could not be retrieved so the answer "
    "may be incomplete."
)


class AgentState(TypedDict):
    """State threaded through the supervisor/specialist graph.

    :param question: The user's original question.
    :param rag_result: Output of :func:`app.agents.rag_tools.rag_search`, set
        after the rag_specialist node runs.
    :param web_result: Output of :func:`app.agents.rag_tools.web_search`, or
        ``None`` if the web_specialist node was never triggered.
    :param step_count: Number of specialist steps taken so far, bounded by
        ``settings.agent_max_steps`` so the graph always terminates.
    :param trace: Ordered list of per-step decision dicts -- a human-readable
        parallel to the OTel spans emitted alongside (see the observability
        integration), useful for the API response's ``agent_trace`` field.
    :param answer: Final synthesized answer, set by the synthesis node.
    :param sources: Final source list; each entry is tagged with
        ``origin: "rag"`` or ``"web"`` so callers can attribute claims.
    :param degraded: True when the web specialist was needed but failed or
        timed out, so the answer had to fall back to RAG-only context.
    """

    question: str
    rag_result: dict[str, Any] | None
    web_result: dict[str, Any] | None
    step_count: int
    trace: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    degraded: bool


def _rag_specialist(state: AgentState) -> AgentState:
    """Search the local knowledge base. Always the first node to run."""
    with _tracer.start_as_current_span("rag_specialist") as span:
        result = rag_search(state["question"])
        span.set_attribute("success", result["success"])
        span.set_attribute("min_distance", result.get("min_distance") or -1.0)
        span.set_attribute("num_results", len(result.get("results", [])))
    trace = [
        *state["trace"],
        {
            "step": "rag_specialist",
            "success": result["success"],
            "min_distance": result.get("min_distance"),
        },
    ]
    return {**state, "rag_result": result, "step_count": state["step_count"] + 1, "trace": trace}


def _route_after_rag(state: AgentState) -> Literal["web_specialist", "synthesis"]:
    """Decide whether local docs were good enough, or web_specialist is needed.

    The hard step cap is checked first and always wins -- this is what
    guarantees the graph terminates even if a future change adds a
    loop-back edge (e.g. a retry-with-refined-query path) that could
    otherwise run indefinitely.
    """
    settings = get_settings()
    if state["step_count"] >= settings.agent_max_steps:
        _logger.info("agent_max_steps reached (%d); skipping web_specialist", settings.agent_max_steps)
        return "synthesis"
    rag_result = state["rag_result"]
    assert rag_result is not None, "_route_after_rag called before _rag_specialist"
    with _tracer.start_as_current_span("supervisor_route") as span:
        if not rag_result["success"] or rag_result["min_distance"] > settings.agent_rag_distance_threshold:
            span.set_attribute("decision", "trigger_web_search")
            return "web_specialist"
        span.set_attribute("decision", "local_docs_sufficient")
        return "synthesis"


def _web_specialist(state: AgentState) -> AgentState:
    """Search the web. Only reached when local docs were insufficient.

    A failure/timeout here is recorded in ``web_result`` (never raised, per
    ``app.agents.rag_tools.web_search``'s contract) and the ``degraded`` flag
    is set so synthesis can fall back to RAG-only instead of crashing.
    """
    with _tracer.start_as_current_span("web_specialist") as span:
        result = web_search(state["question"])
        span.set_attribute("success", result["success"])
        span.set_attribute("num_results", len(result.get("results", [])))
        if not result["success"]:
            span.set_attribute("error", result.get("error", ""))
    trace = [
        *state["trace"],
        {"step": "web_specialist", "success": result["success"], "error": result.get("error")},
    ]
    return {
        **state,
        "web_result": result,
        "step_count": state["step_count"] + 1,
        "trace": trace,
        "degraded": state["degraded"] or not result["success"],
    }


def _format_rag_context(rag_result: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    """Build the LOCAL DOCS context block and source entries from a rag_search result."""
    if not rag_result["success"] or not rag_result["results"]:
        return None, []
    block = "LOCAL DOCS:\n" + "\n\n".join(
        f"[{r['source']}]\n{r['content']}" for r in rag_result["results"]
    )
    sources = [
        {"origin": "rag", "source": r["source"], "content": r["content"][:200]}
        for r in rag_result["results"]
    ]
    return block, sources


def _format_web_context(web_result: dict[str, Any] | None) -> tuple[str | None, list[dict[str, Any]]]:
    """Build the WEB RESULTS context block and source entries from a web_search result."""
    if not web_result or not web_result["success"] or not web_result["results"]:
        return None, []
    block = "WEB RESULTS:\n" + "\n\n".join(
        f"[{r['url']}] {r['title']}\n{r['snippet']}" for r in web_result["results"]
    )
    sources = [
        {"origin": "web", "source": r["url"], "content": r["snippet"]} for r in web_result["results"]
    ]
    return block, sources


def _synthesis(state: AgentState) -> AgentState:
    """Combine whatever context is available into a final answer + sources.

    Never raises: runs even when both specialists came up empty (e.g. an
    uninitialized index and a failed web search), in which case the LLM is
    asked to answer from whatever fragments exist and be honest about the
    gap -- the request still returns a real response, never an unhandled
    exception.
    """
    rag_result = state["rag_result"] or {"success": False, "results": []}
    rag_block, rag_sources = _format_rag_context(rag_result)
    web_block, web_sources = _format_web_context(state["web_result"])

    context_parts = [b for b in (rag_block, web_block) if b]
    context = "\n\n---\n\n".join(context_parts) if context_parts else "No context could be retrieved."
    degraded_note = _DEGRADED_NOTE if state["degraded"] else ""
    system_prompt = _SYNTHESIS_SYSTEM_PROMPT.format(degraded_note=degraded_note)

    with _tracer.start_as_current_span("synthesis") as span:
        span.set_attribute("degraded", state["degraded"])
        span.set_attribute("num_sources", len(rag_sources) + len(web_sources))
        llm = _get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"),
            ]
        )
    trace = [*state["trace"], {"step": "synthesis", "degraded": state["degraded"]}]
    return {
        **state,
        "answer": response.content,
        "sources": [*rag_sources, *web_sources],
        "trace": trace,
    }


def build_agent_workflow() -> CompiledStateGraph:
    """Build the supervisor + specialist StateGraph.

    ``rag_specialist`` always runs first (fast, local, free); ``web_specialist``
    only runs when local docs are insufficient and the step budget allows it.
    Both paths converge on ``synthesis``, which never raises even if every
    specialist failed.
    """
    graph = StateGraph(AgentState)
    graph.add_node("rag_specialist", _rag_specialist)
    graph.add_node("web_specialist", _web_specialist)
    graph.add_node("synthesis", _synthesis)

    graph.set_entry_point("rag_specialist")
    graph.add_conditional_edges(
        "rag_specialist",
        _route_after_rag,
        {"web_specialist": "web_specialist", "synthesis": "synthesis"},
    )
    graph.add_edge("web_specialist", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()


_graph: CompiledStateGraph | None = None


def get_agent_graph() -> CompiledStateGraph:
    """Return the module-level compiled graph singleton, building it on first use."""
    global _graph
    if _graph is None:
        _graph = build_agent_workflow()
    return _graph


def run_agent(question: str) -> dict[str, Any]:
    """Run the supervisor+specialist agent workflow for one question.

    Never raises: any specialist failure degrades to a partial/RAG-only
    answer rather than propagating an exception to the caller.

    :param question: The user's question.
    :returns: ``{"answer": str, "sources": [...], "degraded": bool, "trace": [...]}``.
    """
    with _tracer.start_as_current_span("chat_agent") as span:
        span.set_attribute("question_length", len(question))
        graph = get_agent_graph()
        initial_state: AgentState = {
            "question": question,
            "rag_result": None,
            "web_result": None,
            "step_count": 0,
            "trace": [],
            "answer": "",
            "sources": [],
            "degraded": False,
        }
        final_state = graph.invoke(initial_state)
        span.set_attribute("degraded", final_state["degraded"])
    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "degraded": final_state["degraded"],
        "trace": final_state["trace"],
    }
