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


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message to the agent system")


class ToolCallLog(BaseModel):
    tool: str
    pr: int | None = None
    result: str | None = None
    reviewer: str | None = None
    count: int | None = None
    file: str | None = None
    line: int | None = None
    fixes: int | None = None
    state: str | None = None
    commits_found: int | None = None
    sections: int | None = None


class AgentResponse(BaseModel):
    agent_used: str
    result: str
    tool_calls_log: list[dict]
