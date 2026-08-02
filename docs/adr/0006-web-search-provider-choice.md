# ADR-0006: web_search 工具的 Provider 选型——默认免费、可插拔

- **状态**：已采纳
- **日期**：2026-07-16

## 背景 / 问题

RAG Agent 模式（见 ADR-0005）需要一个 `web_search` 工具，在本地知识库检索不够时补充实时网络信息。需要选一个具体的搜索后端。约束：项目一贯坚持"默认免费、零 API Key"的风格（`llm_provider` 默认 `ollama`，见 `.env.example`），且已有"可插拔连接器"的架构先例（`app/sources/base.py` 的 `DataSource` 抽象，见 ADR-0004）。

## 候选方案

| 方案 | 优点 | 缺点 |
|------|------|------|
| **DuckDuckGo（`ddgs` 包，选中为默认）** | 完全免费、无需 API Key、无需注册账号，和项目"默认免费"的一贯风格一致 | 搜索质量/覆盖面不如商业搜索 API，有轻微限流风险 |
| Tavily | 专为 LLM Agent 设计，搜索质量更好，返回结构更贴合 RAG 场景 | 需要付费 API Key，引入外部依赖账号管理 |
| Serper（Google 搜索包装） | 搜索质量接近原生 Google | 同样需要付费 API Key |
| 不做可插拔，直接硬编码一个 provider | 实现最简单 | 违背 ADR-0004 已确立的"可插拔连接器"架构一致性；后续换 provider 要改核心代码 |

## 决策

默认使用 **DuckDuckGo**（`ddgs` 包），做成**可插拔**的 `WebSearchProvider` 抽象基类（镜像 `app.sources.base.DataSource` 的设计），`TavilyProvider` 作为文档化的付费可选项，通过 `WEB_SEARCH_PROVIDER` 环境变量切换，不用改代码。

## 权衡 / 代价

- 放弃了商业搜索 API 更好的搜索质量，换取零成本、零账号门槛的默认体验——和项目"演示端到端工程能力"的定位一致（面试项目不需要为搜索质量单独付费）。
- `web_search` 工具本身有超时（`web_search_timeout_s`，默认10秒）+ 有界重试（`tenacity`，2次），失败时返回结构化 `{"success": False, "error": ...}` 而不是抛出异常——这是保证"多 Agent 协作不会得不到结果"的关键实现细节，独立于选哪个 provider。
- 实现过程中发现并修复了一个真实 bug：最初用 `with ThreadPoolExecutor(...)` 包裹超时调用，其 `__exit__` 默认 `shutdown(wait=True)`，导致哪怕捕获了 `TimeoutError`，仍会阻塞等待挂起的线程跑完——恰好是这条 ADR 要防止的"降级不生效"问题本身。改为手动管理 executor 生命周期 + `shutdown(wait=False, cancel_futures=True)`，用"模拟30秒卡死的provider + 2秒超时"验证修复后精确在 2.0s 返回。
- 迁移路径：如果未来发现 DuckDuckGo 搜索质量成为实际瓶颈（比如 RAGAS 评估里 web-required 类问题得分明显偏低），切到 Tavily 只需要改 `.env` 里的 `WEB_SEARCH_PROVIDER=tavily` + 配置 `TAVILY_API_KEY`，`rag_workflow.py` 的调用方代码不用改。

## 结果 / 回顾

待补：接入真实使用场景后，记录 DuckDuckGo 的实际限流频率、搜索结果相关性，以及是否触发切换到 Tavily 的评估阈值。
