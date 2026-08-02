# rag-qa-service

A production-style RAG (Retrieval-Augmented Generation) question answering service built with FastAPI, LangChain, and ChromaDB.

## Motivation / 为什么做这个

**问题**：团队/个人的知识散落在 Markdown 文档、wiki、笔记里，找一个具体答案往往要翻好几个文件、花几分钟；而通用 ChatGPT 既不知道你的私有文档、又可能"一本正经地编"。

**这个服务解决什么**：把你自己的文档喂进去，用 RAG 让 LLM **只基于你的文档**回答、并**给出出处**——把"翻文档找答案"从几分钟压到几秒，答案可溯源、可控。

**一套核心，多个数据域（组件复用）**：RAG 管道（chunk → embed → 检索 → 生成 → 评估 → 监控）与领域无关；领域差异被隔离到可插拔的**数据源连接器**（`app/sources/`）。内置：
- `sec` — SEC EDGAR 财报（10-K/10-Q…）问答，做"财报研究助手"
- `arxiv` — arXiv 论文问答，做"论文速读助手"
- `local` — 本地 Markdown/txt 文档

加一个新域 = 加一个连接器，其余零改动（见 [ADR-0004](docs/adr/0004-pluggable-data-source-connectors.md)）。

**目标用户**：需要对某个域的公开资料做**可溯源问答**的人（投资研究、论文速读、文档助手）。

**为什么做成 production 级**：RAG 是当下 LLM 落地最主流的形态；把「数据→检索→生成→评估→上线→监控」端到端跑通，比堆 SOTA 更能体现工程能力。演进计划见 [ROADMAP.md](ROADMAP.md)，关键取舍见 [docs/adr/](docs/adr/)。

**非目标（Non-goals）**：不追求超大规模（见 ADR-0001）；不做模型训练（用开箱/可微调组件）；不做多租户/权限（后续扩展）；**不做"预测涨跌/投资建议"这类玩具**——只做可溯源的信息检索问答。

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
- **Agent Mode**: Supervisor + specialist LangGraph workflow that falls back to live web search when local docs are insufficient, degrading gracefully instead of failing (see [ADR-0005](docs/adr/0005-rag-agent-supervisor-pattern.md))
- **REST API**: FastAPI with streaming (SSE) support
- **Observability**: OpenTelemetry tracing for full request lifecycle (zero-config console exporter by default)
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

# Ingest documents — 选一个数据源（可插拔连接器）
python -m app.core.ingest --source local                                                              # 本地 docs/
python -m app.core.ingest --source arxiv --query "retrieval augmented generation" --max-results 30    # arXiv 论文
python -m app.core.ingest --source sec --ticker AAPL --form 10-K --max-results 3                       # SEC 财报（先在 .env 配 SEC_USER_AGENT）

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
| `/api/v1/chat/agent` | POST | Agent-mode RAG: supervisor routes between local retrieval and live web search when local docs are insufficient, degrading gracefully if web search fails (returns answer + sources + `degraded` + `agent_trace`) |
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
│   ├── main.py              # FastAPI application entry (calls setup_telemetry() at startup)
│   ├── config.py             # Configuration management
│   ├── core/
│   │   ├── embeddings.py     # Embedding model setup
│   │   ├── vectorstore.py    # ChromaDB vector store
│   │   ├── retriever.py      # Document retrieval
│   │   ├── chain.py          # RAG chain composition
│   │   ├── ingest.py         # Document ingestion pipeline
│   │   └── telemetry.py      # OpenTelemetry tracing setup (console exporter by default)
│   ├── agents/
│   │   ├── rag_tools.py       # rag_search / web_search tools (never raise, see ADR-0005/0006)
│   │   └── rag_workflow.py   # Supervisor + specialist LangGraph workflow (Agent mode)
│   └── api/
│       ├── routes.py         # API route definitions
│       └── models.py         # Pydantic request/response models
├── tests/
│   ├── test_retriever.py     # Retriever unit tests
│   ├── test_chain.py         # Chain unit tests
│   ├── test_api.py           # API integration tests
│   └── test_rag_workflow.py  # Agent workflow tests (3 key paths + edge cases)
├── docs/                     # Knowledge base documents
│   └── adr/                  # Architecture decision records
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

本项目实现了两种模式：纯 RAG（`/api/v1/chat`）和 Agent 模式（`/api/v1/chat/agent`，见 [ADR-0005](docs/adr/0005-rag-agent-supervisor-pattern.md)）。两者的核心区别：

| | 纯 RAG | Agent |
|---|---|---|
| **流程** | 检索 → 生成（单轮，固定管道） | 督导者（supervisor）路由到专职子 Agent，按需综合多个来源 |
| **能力边界** | 只能回答知识库里有的内容 | 本地知识库不够时自动触发网络搜索，突破知识库限制 |
| **复杂问题** | 无法拆解，一次检索定成败 | RAG 专职 Agent 先查本地，不够再交给 Web 专职 Agent |
| **速度** | 快（1-3秒） | 慢（可能 5-15秒，取决于是否触发 web_search） |
| **可控性** | 高，行为可预测 | 中——路由决策基于 ChromaDB 检索距离阈值（`agent_rag_distance_threshold`），而非 LLM 自由决策，可预测性介于纯 RAG 和自由 ReAct 循环之间 |
| **适用场景** | FAQ、文档问答、客服 | 复杂分析、多源信息整合、需要联网补充最新信息的任务 |
| **失败处理** | 检索失败即返回空结果 | web_search 失败会优雅降级为纯 RAG 答案（`degraded: true`），而非报错——见 [ADR-0005](docs/adr/0005-rag-agent-supervisor-pattern.md) |

**具体示例**——用户问："MLflow 3.0 比 2.0 有什么改进？我该怎么迁移？"

- **纯 RAG**：检索知识库，如果文档里只有 MLflow 概述没有版本对比，就无法准确回答
- **Agent**：`rag_specialist` 查本地知识库 → 检索距离超过阈值，判定信息不够 → `web_specialist` 搜索 MLflow 3.0 changelog → `synthesis` 综合两个来源生成完整的对比分析和迁移建议，并标注每条结论来自本地文档还是网络搜索

**为什么不是单一 ReAct 循环，而是督导者 + 专职子 Agent**：`rag_specialist` 和 `web_specialist` 各自独立、互不影响——`web_specialist` 超时或搜索失败不会导致整个请求失败，`synthesis` 节点始终会基于已有信息给出回答（并诚实告知用户信息可能不完整），这是本项目对"多 Agent 协作如何保证一定有结果返回"这一设计问题的具体实践，详见 ADR-0005。

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
- `app/core/telemetry.py` 提供开箱即用的 OpenTelemetry 集成：默认用 `ConsoleSpanExporter` 零配置打印 span（无需额外部署 collector），设置 `OTEL_EXPORTER_OTLP_ENDPOINT` 环境变量即可切换到真实的 collector（Jaeger/Tempo 等）
- Agent 模式（`app/agents/rag_workflow.py`）的每个节点都包一层 span，调用链的**节点名称和属性字段**已通过测试验证真实生成（见 `tests/test_rag_workflow.py`），具体耗时取决于实际使用的模型和网络状况：

```
[chat_agent]
  ├── [rag_specialist]      attrs: success=true, min_distance=0.42, num_results=2
  ├── [supervisor_route]    attrs: decision="trigger_web_search"
  ├── [web_specialist]      attrs: success=true, num_results=3
  └── [synthesis]           attrs: degraded=false, num_sources=5
```

基于这些数据，我们可以做出优化决策：
- 如果 `synthesis` 占大部分耗时 → 考虑用更小的模型或加缓存（见 [CS229S 学习笔记](https://github.com/A-ZHANG1/interview-prep) 里 KV cache / prompt caching 相关内容）
- 如果 `rag_specialist` 的 `min_distance` 长期偏高（检索质量差）→ 调整 chunking 策略或加 reranker（见 ROADMAP Phase 2）
- 如果 `web_specialist` 频繁触发 `degraded=true` → 检查 DuckDuckGo 限流情况，评估是否切换到 Tavily（见 [ADR-0006](docs/adr/0006-web-search-provider-choice.md)）

## License

MIT
