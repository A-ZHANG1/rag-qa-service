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

---

## 技术深度解析

### 纯 RAG vs Agent 模式

本项目目前实现了纯 RAG 模式，后续将加入 Agent 模式。两者的核心区别：

| | 纯 RAG | Agent |
|---|---|---|
| **流程** | 检索 → 生成（单轮，固定管道） | 思考 → 选工具 → 执行 → 再思考（多轮循环） |
| **能力边界** | 只能回答知识库里有的内容 | 可以联网搜索、调 API、做计算，突破知识库限制 |
| **复杂问题** | 无法拆解，一次检索定成败 | 自动拆解为多步，逐步收集信息再综合回答 |
| **速度** | 快（1-3秒） | 慢（可能 5-15秒，多轮推理） |
| **可控性** | 高，行为可预测 | 低，LLM 自主决策，可能走弯路 |
| **适用场景** | FAQ、文档问答、客服 | 复杂分析、多源信息整合、需要推理的任务 |

**具体示例**——用户问："MLflow 3.0 比 2.0 有什么改进？我该怎么迁移？"

- **纯 RAG**：检索知识库，如果文档里只有 MLflow 概述没有版本对比，就无法准确回答
- **Agent**：第1步用 `rag_search` 查本地知识库 → 发现信息不够 → 第2步用 `web_search` 搜索 MLflow 3.0 changelog → 第3步综合两个来源生成完整的对比分析和迁移建议

### 为什么用 OpenTelemetry 而不是 MLflow 做可观测性？

两者解决的是**完全不同的问题**：

| | MLflow | OpenTelemetry |
|---|---|---|
| **追踪什么** | ML 实验（超参数、accuracy、loss、模型文件） | 服务请求（每一步的耗时、输入输出、错误） |
| **时间粒度** | 一次实验运行（分钟到小时级） | 一次 API 请求内部（毫秒级） |
| **核心问题** | "哪组参数训练出的模型效果最好？" | "这次请求为什么慢？瓶颈在哪一步？" |
| **类比** | 实验室的实验记录本 | 医院的心电监护仪 |

**在本项目中**：
- 我们**不训练模型**，所以 MLflow 的实验追踪不适用
- 我们需要观察的是：一次 RAG/Agent 请求里，检索耗时多少、LLM 推理耗时多少、Agent 做了几步决策
- 这正是 OpenTelemetry 的强项——它能生成这样的调用链：

```
[POST /api/v1/chat]  总耗时 2.3s
  ├── [Embed Query]       120ms   input: "What is MLflow?"
  ├── [ChromaDB Search]    45ms   top_k=4, results=4
  ├── [Format Context]      2ms   context_length=2800 chars
  └── [LLM Generate]     2.1s    model=llama3.2, tokens_in=850, tokens_out=230

[POST /api/v1/agent]  总耗时 5.1s   (Agent 模式，多步决策)
  ├── [LLM Reason #1]    1.2s    decision: "use rag_search"
  ├── [RAG Search]        0.3s    query: "MLflow components", results=4
  ├── [LLM Reason #2]    1.5s    decision: "use web_search"
  ├── [Web Search]        0.8s    query: "MLflow 3.0 new features"
  └── [LLM Final]        1.3s    decision: "answer ready", tokens_out=450
```

基于这些数据，我们可以做出优化决策：
- 如果 LLM 推理占 90% 耗时 → 考虑用更小的模型或加缓存
- 如果检索结果质量差 → 调整 chunking 策略或加 reranker
- 如果 Agent 循环过多 → 优化 prompt 让 LLM 更高效地选择工具

## License

MIT
