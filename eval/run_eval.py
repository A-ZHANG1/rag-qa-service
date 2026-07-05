"""
RAG 评估 harness（基于 RAGAS）。

对 rag-qa-service 跑一组问答，计算检索 + 生成质量指标：
- faithfulness        答案是否忠于检索到的上下文（不编造）
- answer relevancy    答案是否切题
- context precision   检索到的上下文有多少是真正相关的
- context recall      该召回的相关内容是否都召回了（需 reference/ground_truth）

用法（务必在**仓库根目录**运行，保证能 import app）：
    pip install -r eval/requirements.txt
    # 先 ingest 文档、并在 .env 配好模型（默认 Ollama）
    python -m eval.run_eval

说明：RAGAS 需要一个"评估用 LLM + embedding"作为裁判。本脚本默认**复用项目
配置的模型**（app.config），即 .env 配了什么就用什么，保证一致、零额外配置。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "dataset.jsonl"


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_samples(rows: list[dict]) -> list[dict]:
    """跑真实 RAG 管道，得到每个问题的 answer + 检索到的 contexts。"""
    from app.core.chain import ask
    from app.core.retriever import retrieve

    samples = []
    for row in rows:
        q = row["question"]
        contexts = [d.page_content for d in retrieve(q)]
        answer = ask(q)["answer"]
        samples.append(
            {
                "user_input": q,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": row.get("ground_truth", ""),
            }
        )
        print(f"  ✓ {q[:60]}")
    return samples


def build_eval_models():
    """把项目配置的 LLM + embedding 包成 RAGAS 的裁判模型。"""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from app.core.chain import _get_llm
    from app.core.embeddings import get_embedding_model

    return (
        LangchainLLMWrapper(_get_llm()),
        LangchainEmbeddingsWrapper(get_embedding_model()),
    )


def main() -> int:
    if not DATASET.exists():
        print(f"dataset not found: {DATASET}")
        return 1

    rows = load_dataset(DATASET)
    print(f"Loaded {len(rows)} eval case(s). Running RAG pipeline...")

    try:
        samples = collect_samples(rows)
    except Exception as exc:  # noqa: BLE001
        print(f"跑 RAG 管道失败（是否已 ingest 文档 / 配好 .env？）: {exc}")
        return 1

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError:
        print("未安装 RAGAS。请运行: pip install -r eval/requirements.txt")
        return 1

    eval_llm, eval_emb = build_eval_models()
    dataset = EvaluationDataset.from_list(samples)
    metrics = [
        Faithfulness(llm=eval_llm),
        ResponseRelevancy(llm=eval_llm, embeddings=eval_emb),
        LLMContextPrecisionWithReference(llm=eval_llm),
        LLMContextRecall(llm=eval_llm),
    ]

    result = evaluate(dataset=dataset, metrics=metrics)
    print("\n=== RAGAS results ===")
    print(result)

    out = EVAL_DIR / "results.json"
    try:
        result.to_pandas().to_json(out, orient="records", force_ascii=False, indent=2)
        print(f"\n已保存逐条结果到 {out}")
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
