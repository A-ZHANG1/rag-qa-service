# RAG 评估（Eval）

回答"**你怎么知道这个 RAG 好不好？**"——这是把项目从 demo 升级到 production 级的关键一环，也是面试高频追问。

## 指标（基于 RAGAS）

| 指标 | 衡量什么 | 差了说明 |
|------|----------|----------|
| **Faithfulness** | 答案是否**忠于检索到的上下文**（不编造） | LLM 在幻觉/加戏 |
| **Answer Relevancy** | 答案是否**切题** | 答非所问 |
| **Context Precision** | 检索到的上下文**有多少是相关的** | 检索噪声大 → 调 chunking/top_k |
| **Context Recall** | 该召回的相关内容**是否都召回了**（需 ground_truth） | 漏检 → 调 embedding/chunking |

> 经验：**Faithfulness/Context Precision 低 → 先修检索**（chunking、top_k、加 reranker），别急着换更大模型。

## 怎么跑

```bash
# 在仓库根目录
pip install -r eval/requirements.txt
python -m app.core.ingest        # 确保知识库已建
python -m eval.run_eval          # 跑评估
```

- 默认复用 `.env` 里配置的模型（Ollama / OpenAI / Azure）当"裁判 LLM"，零额外配置。
- 结果打印到终端，并逐条存到 `eval/results.json`。

## 数据集

`eval/dataset.jsonl`，每行一个用例：
```json
{"question": "What is MLflow?", "ground_truth": "MLflow is an open-source platform ..."}
```
- `ground_truth` 用于 context recall；没有也能跑（其余指标仍可算）。
- **建议扩到 ≥20 条**，覆盖：常规问答、多跳问题、**知识库里没有的问题**（测系统会不会老实说"我不知道"）。

## 下一步（见 ROADMAP Phase 1）

- [ ] 扩充数据集到 ≥20 条（含边界 case）
- [ ] 把 eval 接进 CI，PR 时跑 mini-eval，防止改动让质量退化
- [ ] 对比不同 `chunk_size` / `top_k` / 模型的指标，用数字驱动调参
