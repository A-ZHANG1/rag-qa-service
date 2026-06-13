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
