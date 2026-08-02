from fastapi import FastAPI
from app.api.routes import router
from app.core.telemetry import setup_telemetry

setup_telemetry()

app = FastAPI(
    title="RAG QA Service",
    description="Retrieval-Augmented Generation question answering service",
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "RAG QA Service is running. Visit /docs for API documentation."}
