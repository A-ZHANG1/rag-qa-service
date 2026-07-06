# ADR-0004: 可插拔数据源连接器（组件复用）

- **状态**：已采纳
- **日期**：2026-07-06

## 背景 / 问题
最初知识库只从本地 `docs/*.md` 读文档。但"内部文档"这类数据**拿不到**（权限/合规），而 RAG 的价值恰恰取决于喂什么数据。我们希望这个服务能覆盖**多个公开数据域**（SEC 财报、arXiv 论文……），同时**不为每个域重写一套管道**。

## 候选方案
| 方案 | 优点 | 缺点 |
|------|------|------|
| **可插拔 DataSource 连接器（选中）** | 一套 RAG 核心 + 每域一个连接器；加域=加一个类，其余零改动；易测试、易演示多域 | 需要先抽象出统一接口 |
| 每个域一个独立项目/仓库 | 隔离清晰 | 大量重复代码（chunk/embed/retrieve/eval 全复制）；维护噩梦 |
| 在 ingest 里写 if/else 分支 | 起步快 | 很快变成上帝函数，耦合、难测 |

## 决策
引入 `app/sources/` 下的 **`DataSource` 抽象**：每个连接器实现 `fetch(**kwargs) -> list[Document]`，把"数据从哪来、怎么解析"这一唯一的域相关部分隔离出去。下游（chunk → embed → store → retrieve → generate → eval）**完全复用**。`ingest.py` 通过注册表按名字取连接器，自身与域无关。

```
app/sources/
├── base.py         # DataSource ABC（统一接口）
├── local_docs.py   # 本地 md/txt
├── arxiv.py        # arXiv 论文（公开 API）
├── sec_edgar.py    # SEC 财报（公开 API）
└── __init__.py     # 注册表 get_source(name)
```

## 权衡 / 代价
- 抽象带来一点前期设计成本，但换来**线性扩展**（O(1) 加新域）和可测试性。
- 各连接器返回统一的 `Document` + 标准 metadata（`source`、`domain`），因此**引用溯源**在所有域一致。
- **有意不引入新重依赖**：arXiv 用 `httpx` + stdlib XML，SEC 用 `httpx` + stdlib `html.parser`，复用现有依赖。

## 结果 / 回顾
待补：记录接入第三个域（如法规/政策）时实际改动量——预期只新增一个连接器文件 + 注册一行，验证"加域=加连接器"这一设计目标。
