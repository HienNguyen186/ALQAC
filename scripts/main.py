"""
scripts/main.py
Run ALQAC 2026 pipeline on Public Test.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_cache_dir = PROJECT_ROOT / "models"
if _cache_dir.exists():
    os.environ.setdefault("HF_HOME", str(_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_cache_dir))

from src.prediction.llm_predictor import LLMPredictor
from src.reranking.dense_reranker import DenseRetriever
from src.reranking.llm_reranker import LLMReranker
from src.retrieval.bm25_retriever import LawRetriever
from src.utils.config import load_config
from src.utils.io import FriendlyFileError, load_json_file, resolve_path, setup_logging, validate_case_records

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ALQAC 2026 pipeline on Public Test.")
    parser.add_argument("--rerank-mode", default="mock", choices=["mock", "local"])
    parser.add_argument("--llm-mode",    default="mock", choices=["mock", "local"])
    parser.add_argument("--log-level",   default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg          = load_config()
    retrieval_cfg = cfg.get("retrieval", {})
    prediction_cfg = cfg.get("prediction", {})

    try:
        cases = validate_case_records(
            load_json_file(
                resolve_path(cfg["data"]["public_test"], PROJECT_ROOT),
                "ALQAC Public Test",
            )
        )
        LOGGER.info("Loaded %s cases", len(cases))

        LOGGER.info("Loading BM25 index ...")
        retriever = LawRetriever(resolve_path(cfg["data"]["law_corpus"], PROJECT_ROOT))

        LOGGER.info("Loading DenseRetriever (mode=%s) ...", args.rerank_mode)
        dense = DenseRetriever(
            mode=args.rerank_mode,
            model_name=retrieval_cfg.get("dense_model", "BAAI/bge-m3"),
            batch_size=int(retrieval_cfg.get("dense_batch_size", 32)),
            weight_dense=float(retrieval_cfg.get("weight_dense", 0.6)),
            weight_sparse=float(retrieval_cfg.get("weight_sparse", 0.4)),
            use_colbert=bool(retrieval_cfg.get("use_colbert", False)),
        )

        LOGGER.info("Loading LLMReranker (mode=%s) ...", args.rerank_mode)
        reranker = LLMReranker(
            mode=args.rerank_mode,
            model_name=prediction_cfg.get("reranker_model", "Qwen/Qwen2.5-3B-Instruct"),
        )

        LOGGER.info("Loading LLMPredictor (mode=%s) ...", args.llm_mode)
        predictor = LLMPredictor(
            mode=args.llm_mode,
            model_name=prediction_cfg.get("predictor_model", "Qwen/Qwen3-8B"),
        )

        results = []
        for case in tqdm(cases, desc="Pipeline"):
            query = str(case.get("case_query", "")).strip()

            candidate_pool = retriever.retrieve_candidate_pool(
                query=query,
                top_k_articles=int(retrieval_cfg.get("bm25_top_k_articles", 200)),
                top_k_laws=int(retrieval_cfg.get("bm25_top_k_laws", 5)),
                strategy=retrieval_cfg.get("candidate_strategy", "hybrid"),
            )
            dense_results = dense.retrieve(
                query=query,
                articles=candidate_pool,
                top_k=int(retrieval_cfg.get("dense_top_k", 30)),
            )
            final_articles = reranker.rerank(
                query=query,
                articles=dense_results,
                min_keep=int(retrieval_cfg.get("final_min_k", 3)),
                max_keep=int(retrieval_cfg.get("final_max_k", 15)),
            )
            prediction = predictor.predict(
                case_query=query,
                law_articles=final_articles,
            )

            results.append({
                "case_id":    case["case_id"],
                "prediction": prediction.label,
                "confidence": round(float(prediction.confidence), 4),
                "reasoning":  prediction.reasoning,
                "law_evidence": [
                    {"law_id": x["law_id"], "aid": x["aid"], "dense_score": x.get("dense_score")}
                    for x in final_articles
                ],
            })

        output_dir = PROJECT_ROOT / cfg.get("output", {}).get("submissions_dir", "outputs/submissions")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"submission_{args.rerank_mode}_{args.llm_mode}_{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)

        LOGGER.info("Done. Saved to: %s", output_path)

    except FriendlyFileError as exc:
        LOGGER.error("\n%s", exc)
        return 2
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
