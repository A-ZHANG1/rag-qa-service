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
| LLM | OpenAI GPT-4o / Azure OpenAI |
| Embedding | text-embedding-3-small |
| Vector Store | ChromaDB |
| Observability | OpenTelemetry |
| Containerization | Docker |
| Testing | pytest |

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Local Development

```bash
# Clone the repo
git clone https://github.com/A-ZHANG1/rag-qa-service.git
cd rag-qa-service

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Ingest documents
python -m app.core.ingest

# Start the server
uvicorn app.main:app --reload --port 8000
```

### Docker

```bash
docker-compose up --build
```

### API Usage

```bash
# Basic query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is MLflow?"}'

# Streaming response
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is MLflow?"}'
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
