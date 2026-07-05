# ADR-0002: 分块（Chunking）策略——Markdown 感知的递归切分

- **状态**：已采纳
- **日期**：2026-06-14

## 背景 / 问题
知识库主要是 Markdown 文档。chunking 直接影响检索质量：切太大→召回噪声多、上下文浪费 token；切太小→语义被切碎、答案缺上下文。

## 候选方案
| 方案 | 优点 | 缺点 |
|------|------|------|
| **RecursiveCharacterTextSplitter + Markdown 分隔符（选中）** | 优先在 `## / ###` 标题和段落边界切，尽量保持语义单元完整；实现简单 | 基于**字符数**而非 token 数，可能与模型 token 上限有偏差 |
| 固定长度切分 | 最简单 | 会从句子中间切断，语义破碎 |
| 语义切分（embedding 相似度找边界） | 语义最连贯 | 慢、成本高、实现复杂 |
| 按标题层级整段切 | 结构清晰 | 长 section 会超上下文，短的又太碎 |

当前实现（`app/core/ingest.py`）：
```python
RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
)
```

## 决策
选 **Markdown 感知的递归切分**：按"标题→段落→行→词"的优先级递归切，既保持语义单元、又保证块不超限；`chunk_size`/`chunk_overlap` 可配置，便于调参。

## 权衡 / 代价
- 字符数≠token 数——**待办**：如需精确控制，改用 token 级切分（`tiktoken`）。
- overlap 增加冗余存储换取跨块上下文连续性。

## 结果 / 回顾
待补：用 `eval/` 对比不同 `chunk_size`/`overlap` 下的 context precision/recall，选出最优参数并记录数字。
