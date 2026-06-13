import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from app.api.models import ChatRequest, ChatResponse, HealthResponse
from app.core.chain import ask, ask_stream

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.1.0")


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
