# ROADMAP — 把 rag-qa-service 做成"业务逻辑完整 + production 级"的精品

> 目标：不是堆 SOTA，而是把「数据→检索→生成→评估→上线→监控」端到端跑通，且每个决策都有**业务理由和权衡**。面试时能讲成一个完整的故事。

图例：`[ ]` 待做 · `[~]` 进行中 · `[x]` 完成

---

## Phase 0 · 业务逻辑（最高优先）

> 这一阶段几乎不写代码，但对面试价值最高——"完整业务逻辑 > SOTA"。

- [x] README 加 **Motivation / 问题陈述**（谁用、解决什么问题、为什么值得做）
- [x] `docs/adr/` 建 **架构决策记录（ADR）**：向量库、chunking、LLM/embedding 选型的取舍
- [x] **可插拔数据源连接器**（`app/sources/`：local / sec / arxiv）——一套 RAG 核心、多域复用，见 ADR-0004
- [x] **RAG Agent 模式**（`app/agents/`：`rag_search` + `web_search` 督导者+专职子Agent架构，本地知识库不够时自动补充网络搜索，且 web 失败会优雅降级而非报错），见 [ADR-0005](docs/adr/0005-rag-agent-supervisor-pattern.md)、[ADR-0006](docs/adr/0006-web-search-provider-choice.md)
- [ ] 补充 ADR：为什么用 LangChain（vs 直接调 API / LlamaIndex）
- [ ] 在 README 写清**非目标（Non-goals）**：不做什么、边界在哪

## Phase 1 · 端到端 production

- [~] **Eval 骨架**（`eval/`，基于 RAGAS：faithfulness / answer relevancy / context precision & recall）
- [ ] 扩充 eval 数据集到 ≥20 条有代表性的问答（含边界 case：知识库里没有的问题、需要联网才能回答的问题）
- [x] **基础可观测性**：`app/core/telemetry.py` 提供 OpenTelemetry span 埋点（零配置 console exporter，设置 `OTEL_EXPORTER_OTLP_ENDPOINT` 可切换到真实 collector），Agent workflow 每个节点（`rag_specialist`/`supervisor_route`/`web_specialist`/`synthesis`）都有对应 span
- [ ] **监控**：把 span 数据聚合成指标——p50/p99 延迟、token 成本、检索命中率、`degraded` 触发频率
- [ ] 把线上 query/answer 落库，供离线评估（形成"评估闭环"）
- [ ] **CI**：GitHub Actions 跑 lint + pytest；可选跑一次 mini-eval
- [ ] 部署故事：README 写清如何部署（Docker）+ 环境变量 + 扩缩容思路

## Phase 2 · 系统 / 性能（配 Stanford CS229s）

- [ ] 用 **vLLM** 自托管一个开源模型作为 LLM 后端（现有 Ollama 是本地方案，vLLM 面向吞吐）
- [ ] 开 **continuous batching**，benchmark **前后延迟/吞吐**，数字写进 README
- [ ] 量化 **KV cache** 对多并发的影响
- [ ] 检索侧优化：批量 embedding、结果缓存（对照 §性能三件套）
- [ ] 加 **reranker**（cross-encoder）提升检索精度，评估 quality/latency 权衡

## Phase 3 · 训练组件（可选，配 Stanford CS336）

- [ ] 在领域文档上**微调一个小 reranker 或 embedding 模型**，对比开箱模型的检索质量
- [ ] （进阶）SFT 一个小模型统一回答风格；记录 cost/质量权衡与 motivation

---

## 学习资源映射

| 课程 | 优先级 | 对应 Phase |
|------|--------|-----------|
| **CS229s**（ML 系统：KV cache / vLLM / continuous batching / speculative decoding） | 🥇 先学 | Phase 2 |
| **CS336**（从零实现 LLM：预训练 / 分布式 / SFT / RLHF / scaling laws） | 🥈 次之 | Phase 3 |
| **CS149**（并行计算：SIMD / GPU / CUDA / FlashAttention 原理） | 🥉 打底 | 贯穿 Phase 2 的底层理解 |

> 建议：**CS229s 边学边改 Phase 2**，性价比最高；不要三门课齐头并进（各 100 小时级）。

---

## 面试怎么讲这个项目（一条主线）

> **Motivation**（解决什么问题）→ **架构取舍**（为什么 A 不 B，见 ADR）→ **端到端**（数据怎么来 / 模型怎么选 / 怎么评估 / 上线怎么监控）→ **性能优化**（带数字）→ **学到什么 / 下一步**。
