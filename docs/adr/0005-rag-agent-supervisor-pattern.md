# ADR-0005: RAG Agent 模式采用"督导者 + 专职子Agent"而非单一 ReAct 循环

- **状态**：已采纳
- **日期**：2026-07-16

## 背景 / 问题

README 里描述了"Agent 模式"的设想：用户问一个本地知识库信息不够的问题时，先查本地文档（`rag_search`），不够再查网络（`web_search`），综合两者生成答案。最直接的实现方式是让一个 Agent 自己决定要不要调用 `web_search`（单一 ReAct 循环，一个 Agent 身兼检索+搜索+综合三职）。

需要决定：用单一 ReAct 循环，还是拆成多个各司其职的子 Agent？约束：这是一个面试/学习项目，复杂度要有业务理由，不能为了炫技堆多 Agent；同时希望具备演示"多 Agent 协作如何保证一定有结果返回"这一较有深度的设计能力。

## 候选方案

| 方案 | 优点 | 缺点 |
|------|------|------|
| 单一 ReAct 循环（一个 Agent 自己决定调不调 web_search） | 实现简单，符合 README 最初的设想 | 检索失败和搜索失败耦合在同一个 Agent 里，任一环节异常都可能让整个循环挂掉；无法独立复用/替换某个能力 |
| **督导者 + 专职子Agent（选中）** | 关注点分离——RAG 专职、Web 专职、Supervisor 路由综合各自独立；某个子环节失败不拖累另一个；子环节可以独立测试/替换 | 比单一循环多几个节点，代码量略增 |
| 完整 Polly 式并行编码 + 跨厂商审查 | 面试话题更"炫" | 和 QA 服务的实际形状不匹配（那是解决并行代码生成问题的），过度设计 |

## 决策

选择**督导者（Supervisor）+ 专职子Agent（RAG Specialist / Web Specialist）+ Synthesis** 架构，用 LangGraph `StateGraph` 实现：`rag_specialist` 永远先跑，`_route_after_rag` 根据 ChromaDB 检索到的最小 L2 距离是否超过 `agent_rag_distance_threshold` 阈值，决定是否触发 `web_specialist`，最终统一在 `synthesis` 节点综合。

设计参考了对 OmniGent（`omnigent-ai/omnigent`）多 Agent 协作模式的分析：Polly 模式"督导者委派专职子Agent、而非一个 Agent 身兼多职"的核心思路，以及 `subagent_block_notifier.py` 体现的"子Agent失败不能让整个请求得不到结果"这一可靠性原则。

## 权衡 / 代价

- 放弃了单一循环的实现简洁性，换来了故障隔离：`web_specialist` 超时/失败时（见 ADR-0006 及 `rag_tools.web_search` 的 never-raise 设计），`synthesis` 节点能优雅降级为纯 RAG 答案并诚实告知用户，而不是让整个请求跟着挂掉——这是当初决定不做单一 ReAct 循环的直接原因。
- 硬性步数上限（`agent_max_steps`，默认4）在当前这个"最多两级分支"的简单图结构里暂时是防御性设计（当前拓扑不可能真正无限循环），是为未来可能加入"supervisor 用改写后的 query 重试 web_search"这类回环边做的预留保护。
- 明确不采纳的部分：Polly 完整并行编码版、Debby 式多模型辩论、Hindsight 跨会话记忆、MCTS 搜索——均与当前项目"不追求超大规模"的非目标（见 ADR-0001）冲突，属于过度设计。

## 结果 / 回顾

上线前用 3 条关键路径做了验证（本地文档足够不触发web / 不够触发web且成功 / web失败仍优雅降级返回真实答案），全部通过。待补：接入真实 Ollama/OpenAI 模型和真实 `web_search` provider 后，观察 `agent_rag_distance_threshold=0.35` 这个默认阈值在实际检索质量下是否需要调整。
