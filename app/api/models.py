from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of documents to retrieve")


class SourceDoc(BaseModel):
    content: str
    source: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]


class HealthResponse(BaseModel):
    status: str
    version: str


class AgentSourceDoc(BaseModel):
    origin: str = Field(..., description="Where this source came from: 'rag' or 'web'")
    source: str = Field(..., description="File path (rag) or URL (web) of the source")
    content: str


class AgentChatResponse(BaseModel):
    answer: str
    sources: list[AgentSourceDoc]
    degraded: bool = Field(
        ...,
        description=(
            "True when the web specialist was needed but failed/timed out, so the "
            "answer had to fall back to local-docs-only context."
        ),
    )
    agent_trace: list[dict[str, Any]] = Field(
        ...,
        description="Ordered per-step decision log (rag_specialist/web_specialist/synthesis).",
    )
