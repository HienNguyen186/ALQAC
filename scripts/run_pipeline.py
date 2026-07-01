"""
Pipeline ALQAC 2026:
  BM25 (top-k luật) → BGE-M3 (top-20 articles) → Qwen2.5-3B (yes/no) → Qwen3-8B (predict)

Chạy:
  # Test nhanh, không GPU:
  python scripts/run_pipeline.py --rerank-mode mock --llm-mode mock

  # Full GPU:
  python scripts/run_pipeline.py --rerank-mode local --llm-mode local

  # Debug 5 cases:
  python scripts/run_pipeline.py --rerank-mode mock --llm-mode mock --limit 5
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from src.retrieval.bm25_retriever import LawRetriever
from src.reranking.dense_reranker import DenseRetriever
from src.reranking.llm_reranker   import LLMReranker
from src.prediction.llm_predictor import LLMPredictor


def run_pipeline(
    test_path:    str,
    corpus_path:  str,
    output_dir:   str,
    rerank_mode:  str = "mock",   # "mock" | "local"
    llm_mode:     str = "mock",   # "mock" | "local"
    dense_model:  str = "BAAI/bge-m3",
    llm1_model:   str = "Qwen/Qwen2.5-3B-Instruct",
    llm2_model:   str = "Qwen/Qwen3-8B",
    top_k_laws:   int = 3,    # BM25: lấy top bao nhiêu bộ luật
    top_k_dense:  int = 20,   # BGE-M3: lấy top bao nhiêu articles
    final_min_k:  int = 2,    # LLM reranker: giữ tối thiểu
    final_max_k:  int = 5,    # LLM reranker: giữ tối đa → vào Qwen3-8B
    limit:        int = None,
) -> str:

    # ── Load ──────────────────────────────────────────────────────────────────
    print("[1/4] Loading corpus (BM25)...")
    retriever = LawRetriever(corpus_path)
    print(f"      → {len(retriever._articles):,} articles | {len(retriever._law_to_indices)} laws")

    print(f"[2/4] Loading BGE-M3 (mode={rerank_mode})...")
    dense = DenseRetriever(mode=rerank_mode, model_name=dense_model)

    print(f"[3/4] Loading LLM Reranker Qwen2.5-3B (mode={rerank_mode})...")
    llm_reranker = LLMReranker(mode=rerank_mode, model_name=llm1_model)

    print(f"[4/4] Loading Qwen3-8B (mode={llm_mode})...")
    predictor = LLMPredictor(mode=llm_mode, model_name=llm2_model)

    with open(test_path, encoding="utf-8") as f:
        cases = json.load(f)
    if limit:
        cases = cases[:limit]
    print(f"      → {len(cases)} cases\n")

    # ── Pipeline loop ─────────────────────────────────────────────────────────
    submissions = []

    for case in tqdm(cases, desc="Pipeline"):
        case_id    = case["case_id"]
        case_query = case["case_query"]

        # Stage 1: BM25 → top-3 bộ luật → lấy tất cả articles của 3 luật đó
        law_ids         = retriever.retrieve_laws(case_query, top_k=top_k_laws)
        candidate_pool  = retriever.get_articles_by_laws(law_ids)

        # Stage 2: BGE-M3 → chấm cosine score → top-20
        top_articles = dense.retrieve(case_query, candidate_pool, top_k=top_k_dense)

        # Stage 3: Qwen2.5-3B → yes/no filter → 2~5 articles
        final_articles = llm_reranker.rerank(
            case_query, top_articles,
            min_keep=final_min_k,
            max_keep=final_max_k,
        )

        # Stage 4: Qwen3-8B → predict
        result = predictor.predict(case_query, final_articles)

        submissions.append({
            "case_id":      case_id,
            "prediction":   result.label,
            "law_evidence": [
                {"law_id": a["law_id"], "aid": a["aid"]}
                for a in final_articles
            ],
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(output_dir) / f"submission_{rerank_mode}_{llm_mode}_{ts}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submissions, f, ensure_ascii=False, indent=2)

    dist = Counter(s["prediction"] for s in submissions)
    print(f"\n{'='*55}")
    print(f"  Output : {out_path}")
    print(f"  Cases  : {len(submissions)}")
    print(f"  Labels : " + " | ".join(f"{k}:{v}" for k, v in dist.most_common()))
    print(f"{'='*55}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rerank-mode",  default="mock", choices=["mock", "local"])
    p.add_argument("--llm-mode",     default="mock", choices=["mock", "local"])
    p.add_argument("--dense-model",  default="BAAI/bge-m3")
    p.add_argument("--llm1-model",   default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--llm2-model",   default="Qwen/Qwen3-8B")
    p.add_argument("--top-k-laws",   type=int, default=3)
    p.add_argument("--top-k-dense",  type=int, default=20)
    p.add_argument("--final-min-k",  type=int, default=2)
    p.add_argument("--final-max-k",  type=int, default=5)
    p.add_argument("--limit",        type=int, default=None)
    p.add_argument("--test",         default="data/raw/ALQAC2026_public_test.json")
    p.add_argument("--corpus",       default="data/raw/corpus_law_pub.json")
    p.add_argument("--output",       default="outputs/submissions")
    args = p.parse_args()

    run_pipeline(
        test_path   = args.test,
        corpus_path = args.corpus,
        output_dir  = args.output,
        rerank_mode = args.rerank_mode,
        llm_mode    = args.llm_mode,
        dense_model = args.dense_model,
        llm1_model  = args.llm1_model,
        llm2_model  = args.llm2_model,
        top_k_laws  = args.top_k_laws,
        top_k_dense = args.top_k_dense,
        final_min_k = args.final_min_k,
        final_max_k = args.final_max_k,
        limit       = args.limit,
    )


if __name__ == "__main__":
    main()
