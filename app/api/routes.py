import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from app.api.models import ChatRequest, ChatResponse, HealthResponse, AgentChatResponse
from app.core.chain import ask, ask_stream
from app.agents.rag_workflow import run_agent

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.2.0")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG question answering endpoint."""
    result = ask(request.query)
    return ChatResponse(**result)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming RAG endpoint using Server-Sent Events."""

    async def event_generator():
        async for chunk in ask_stream(request.query):
            yield {"event": "message", "data": json.dumps({"chunk": chunk})}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


@router.post("/chat/agent", response_model=AgentChatResponse)
async def chat_agent(request: ChatRequest):
    """Agent-mode RAG endpoint: supervisor routes between local-doc retrieval
    and live web search, falling back gracefully if the web search fails.

    Unlike ``/chat``, the response includes ``degraded`` (whether the web
    specialist was needed but unavailable) and ``agent_trace`` (the ordered
    per-step decision log), so callers can see why an answer might be
    incomplete instead of only getting the final text.
    """
    result = run_agent(request.query)
    return AgentChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        degraded=result["degraded"],
        agent_trace=result["trace"],
    )
