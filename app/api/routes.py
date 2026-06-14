import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from app.api.models import ChatRequest, ChatResponse, HealthResponse, AgentRequest, AgentResponse
from app.core.chain import ask, ask_stream
from app.agents.workflow import run_agents

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


@router.post("/agent", response_model=AgentResponse)
async def agent_endpoint(request: AgentRequest):
    """Multi-agent PR workflow endpoint.

    Three agents available (auto-routed by intent):
    - **催 Review Agent**: monitors PRs, sends review reminders
    - **Fix Comment Agent**: auto-fixes PR review comments
    - **周报 Agent**: generates weekly report for manager

    Examples:
    - "帮我催一下 review"
    - "把 PR comments 都修了"
    - "生成本周周报给 M1"
    """
    result = run_agents(request.message)
    return AgentResponse(**result)
