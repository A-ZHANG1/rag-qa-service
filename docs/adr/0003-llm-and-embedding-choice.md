# ADR-0003: LLM 与 Embedding 选型——默认本地 Ollama，可切换 OpenAI/Azure

- **状态**：已采纳
- **日期**：2026-06-14

## 背景 / 问题
RAG 需要两个模型：生成用的 LLM 和检索用的 embedding 模型。选型要平衡：开发门槛（是否需要 API key）、成本、隐私（数据是否出域）、质量、可复现性。

## 候选方案
| 维度 | 本地 Ollama（默认） | OpenAI / Azure OpenAI（可选） |
|------|--------------------|-------------------------------|
| 成本 | 免费 | 按量付费 |
| API key | 不需要 | 需要 |
| 隐私 | 数据不出本机 | 数据发到云端 |
| 质量 | 中（llama3.2 等小模型） | 高（GPT-4 类） |
| 速度 | 取决于本地硬件 | 稳定、通常更快 |

- LLM：默认 `llama3.2`（Ollama），可切 `gpt-*`（OpenAI）或 Azure 部署。
- Embedding：默认 `nomic-embed-text`（Ollama），可切 `text-embedding-3-small`（OpenAI）。
- 两者都设 `temperature=0`，保证回答可复现（便于评估）。

## 决策
**默认走本地 Ollama**：让任何人 `git clone` 后**零成本、零 key、离线**就能跑通端到端；通过 `settings.use_ollama / use_azure` 一键切换到云端模型换取质量。

## 权衡 / 代价
- 本地小模型质量/速度不如 GPT-4——可接受，因为默认场景是**开发/演示**；生产可切云端。
- 切换成本低：`_get_llm()` / `get_embedding_model()` 已按配置分支，替换只改 `.env`。
- **注意**：切换 embedding 模型会改变向量空间，**必须重新 ingest** 整个知识库。

## 结果 / 回顾
待补：用 `eval/` 对比 Ollama vs OpenAI 在同一批问答上的 faithfulness/answer relevancy，量化"质量 vs 成本"的差距。
