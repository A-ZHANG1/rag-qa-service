# rag-qa-service

A production-style RAG (Retrieval-Augmented Generation) question answering service built with FastAPI, LangChain, and ChromaDB.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐     ┌─────────┐
│  User Query  │────▶│  FastAPI API  │────▶│  Retriever  │────▶│ ChromaDB │
└─────────────┘     └──────┬───────┘     └─────┬──────┘     └─────────┘
                           │                    │
                           ▼                    ▼
                    ┌──────────────┐     ┌────────────┐
                    │  LLM (GPT)   │◀────│  Context    │
                    └──────┬───────┘     └────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Response   │
                    └──────────────┘
```

## Features

- **RAG Pipeline**: Document loading → chunking → embedding → vector store → retrieval → generation
- **REST API**: FastAPI with streaming (SSE) support
- **Observability**: OpenTelemetry tracing for full request lifecycle
- **Containerized**: Docker + docker-compose for one-command startup
- **Tested**: pytest with unit and integration tests

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Web Framework | FastAPI |
| LLM Orchestration | LangChain |
| LLM | Ollama (local, free) / OpenAI / Azure OpenAI |
| Embedding | nomic-embed-text (Ollama) / text-embedding-3-small (OpenAI) |
| Vector Store | ChromaDB |
| Observability | OpenTelemetry |
| Containerization | Docker |
| Testing | pytest |

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed (default, free, no API key needed)

### Local Development

```bash
# Clone the repo
git clone https://github.com/A-ZHANG1/rag-qa-service.git
cd rag-qa-service

# Pull Ollama models (free, runs locally)
ollama pull llama3.2
ollama pull nomic-embed-text

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Default config uses Ollama — no API key needed!
# Edit .env to switch to OpenAI or Azure OpenAI if desired

# Ingest documents
python -m app.core.ingest

# Start the server
uvicorn app.main:app --reload --port 8000
```

### Docker

```bash
docker-compose up --build
```

### Interactive API Docs (Swagger UI)

Once the server is running, open **http://localhost:8000/docs** for the interactive Swagger UI:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | RAG question answering (returns JSON with answer + sources) |
| `/api/v1/chat/stream` | POST | Streaming RAG with Server-Sent Events |
| `/api/v1/health` | GET | Health check |

### API Usage

```bash
# Basic query
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is MLflow?"}'

# Streaming response
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is MLflow?"}'
```

### Example Response

```json
{
  "answer": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides four main components: MLflow Tracking, MLflow Models, MLflow Model Registry, and MLflow Projects.",
  "sources": [
    {
      "content": "# MLflow Overview\n\nMLflow is an open-source platform for managing the end-to-end...",
      "source": "docs/mlflow-overview.md"
    }
  ]
}
```

## Project Structure

```
rag-qa-service/
├── app/
│   ├── main.py              # FastAPI application entry
│   ├── config.py             # Configuration management
│   ├── core/
│   │   ├── embeddings.py     # Embedding model setup
│   │   ├── vectorstore.py    # ChromaDB vector store
│   │   ├── retriever.py      # Document retrieval
│   │   ├── chain.py          # RAG chain composition
│   │   └── ingest.py         # Document ingestion pipeline
│   └── api/
│       ├── routes.py         # API route definitions
│       └── models.py         # Pydantic request/response models
├── tests/
│   ├── test_retriever.py     # Retriever unit tests
│   ├── test_chain.py         # Chain unit tests
│   └── test_api.py           # API integration tests
├── docs/                     # Knowledge base documents
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
