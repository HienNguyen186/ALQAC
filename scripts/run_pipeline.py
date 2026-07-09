"""End-to-end ALQAC 2026 pipeline runner."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_cache_dir = PROJECT_ROOT / "models"
if _cache_dir.exists():
    os.environ.setdefault("HF_HOME", str(_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_cache_dir))

from tqdm import tqdm

from src.api.case_api import CaseAPIClient
from src.prediction.llm_predictor import LLMPredictor
from src.reranking.dense_reranker import DenseRetriever
from src.reranking.llm_reranker import LLMReranker
from src.retrieval.bm25_retriever import LawRetriever
from src.utils.config import load_config
from src.utils.io import FriendlyFileError, load_json_file, resolve_path, setup_logging, validate_case_records

LOGGER = logging.getLogger(__name__)

def get_case_context(case: dict[str, Any], case_api: CaseAPIClient) -> tuple[str, list[dict]]:
    """
    Public:
        dùng case_fact nếu có.

    Private:
        lấy qua Case API.
    """

    # Public test
    if case.get("case_fact"):
        return case["case_fact"], []

    # Private test
    chunks = case_api.retrieve_multi(
        case_id=str(case["case_id"]),
        case_query=str(case["case_query"]),
    )

    if not chunks:
        return str(case["case_query"]), []

    case_context = "\n\n".join(
        chunk["text"] for chunk in chunks
    )

    return case_context, chunks

def run_pipeline(
    test_path: str | Path,
    corpus_path: str | Path,
    output_dir: str | Path,
    rerank_mode: str = "mock",
    llm_mode: str = "mock",
    dense_model: str = "BAAI/bge-m3",
    llm1_model: str = "Qwen/Qwen2.5-3B-Instruct",
    llm2_model: str = "Qwen/Qwen3-8B",
    top_k_laws: int = 5,
    top_k_bm25_articles: int = 200,
    top_k_dense: int = 30,
    final_min_k: int = 3,
    final_max_k: int = 15,
    limit: int | None = None,
    dense_batch_size: int = 32,
    candidate_strategy: str = "hybrid",
    weight_dense: float = 0.6,
    weight_sparse: float = 0.4,
    use_colbert: bool = False,
) -> str:
    """Run retrieval, reranking, and prediction for ALQAC cases."""

    LOGGER.info("[1/4] Loading corpus and BM25 index")
    retriever = LawRetriever(resolve_path(corpus_path, PROJECT_ROOT))
    LOGGER.info("Indexed %s articles across %s laws", len(retriever._articles), len(retriever._law_to_indices))

    LOGGER.info("[2/4] Loading dense retriever mode=%s", rerank_mode)
    dense = DenseRetriever(
        mode=rerank_mode,
        model_name=dense_model,
        batch_size=dense_batch_size,
        weight_dense=weight_dense,
        weight_sparse=weight_sparse,
        use_colbert=use_colbert,
    )

    LOGGER.info("[3/4] Loading LLM reranker mode=%s", rerank_mode)
    llm_reranker = LLMReranker(mode=rerank_mode, model_name=llm1_model)

    LOGGER.info("[4/4] Loading predictor mode=%s", llm_mode)
    predictor = LLMPredictor(mode=llm_mode, model_name=llm2_model)

    LOGGER.info("[5/5] Initializing Case API")
    case_api = CaseAPIClient(
        mode="real",
    )

    LOGGER.info("Downloading private cases...")

    cases = case_api.get_private_cases()

    LOGGER.info("Received %d cases", len(cases))

    if limit is not None:
        cases = cases[:max(0, limit)]
    LOGGER.info("Loaded %s cases", len(cases))

    submissions: list[dict[str, Any]] = []
    for case in tqdm(cases, desc="Pipeline"):

        case_id = str(case["case_id"])
        case_query = str(case["case_query"])

        case_context, chunks = get_case_context(
            case,
            case_api,
        )

        # ==========================================
        # BM25
        # ==========================================
        candidate_pool = retriever.retrieve_candidate_pool(
            case_context,
            top_k_articles=top_k_bm25_articles,
            top_k_laws=top_k_laws,
            strategy=candidate_strategy,
        )

        # ==========================================
        # Dense Retrieval
        # ==========================================
        top_articles = dense.retrieve(
            case_context,
            candidate_pool,
            top_k=top_k_dense,
        )

        # ==========================================
        # LLM Reranker
        # ==========================================
        final_articles = llm_reranker.rerank(
            case_context,
            top_articles,
            min_keep=final_min_k,
            max_keep=final_max_k,
        )

        # ==========================================
        # Predictor
        # ==========================================
        result = predictor.predict(
            case_context,
            final_articles,
        )

        submissions.append(
            {
                "case_id": case_id,
                "prediction": result.label,
                "case_evidence": [
                    c["chunk_id"]
                    for c in chunks
                ],
                "law_evidence": [
                    {
                        "law_id": a["law_id"],
                        "aid": a["aid"],
                    }
                    for a in final_articles
                ],
            }
        )

    resolved_output_dir = resolve_path(output_dir, PROJECT_ROOT)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = resolved_output_dir / f"submission_{rerank_mode}_{llm_mode}_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(submissions, fh, ensure_ascii=False, indent=2)

    distribution = Counter(item["prediction"] for item in submissions)
    LOGGER.info("Output: %s", out_path)
    LOGGER.info("Cases: %s", len(submissions))
    LOGGER.info("Labels: %s", dict(distribution))
    return str(out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    config = load_config()
    rc = config.get("retrieval", {})
    pc = config.get("prediction", {})
    parser = argparse.ArgumentParser(description="Run the ALQAC 2026 Legal AI pipeline.")
    parser.add_argument("--rerank-mode", default="local", choices=["mock", "local"])
    parser.add_argument("--llm-mode",    default="local", choices=["mock", "local"])
    parser.add_argument("--dense-model", default=rc.get("dense_model", "BAAI/bge-m3"))
    parser.add_argument("--llm1-model",  default=pc.get("reranker_model", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--llm2-model",  default=pc.get("predictor_model", "Qwen/Qwen3-8B"))
    parser.add_argument("--top-k-laws",          type=int,   default=int(rc.get("bm25_top_k_laws", 5)))
    parser.add_argument("--top-k-bm25-articles", type=int,   default=int(rc.get("bm25_top_k_articles", 200)))
    parser.add_argument("--top-k-dense",         type=int,   default=int(rc.get("dense_top_k", 30)))
    parser.add_argument("--final-min-k",         type=int,   default=int(rc.get("final_min_k", 3)))
    parser.add_argument("--final-max-k",         type=int,   default=int(rc.get("final_max_k", 15)))
    parser.add_argument("--dense-batch-size",    type=int,   default=int(rc.get("dense_batch_size", 32)))
    parser.add_argument("--weight-dense",        type=float, default=float(rc.get("weight_dense", 0.6)))
    parser.add_argument("--weight-sparse",       type=float, default=float(rc.get("weight_sparse", 0.4)))
    parser.add_argument("--use-colbert", action="store_true", default=bool(rc.get("use_colbert", False)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--corpus", default=config.get("data", {}).get("law_corpus", "data/raw/corpus_law_pub.json"))
    parser.add_argument("--output", default=config.get("output", {}).get("submissions_dir", "outputs/submissions"))
    parser.add_argument("--candidate-strategy", default=rc.get("candidate_strategy", "hybrid"),
                        choices=["article", "law", "hybrid"])
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args   = parser.parse_args()
    setup_logging(args.log_level)
    try:
        run_pipeline(
            corpus_path=args.corpus,
            output_dir=args.output,
            rerank_mode=args.rerank_mode,
            llm_mode=args.llm_mode,
            dense_model=args.dense_model,
            llm1_model=args.llm1_model,
            llm2_model=args.llm2_model,
            top_k_laws=args.top_k_laws,
            top_k_bm25_articles=args.top_k_bm25_articles,
            top_k_dense=args.top_k_dense,
            final_min_k=args.final_min_k,
            final_max_k=args.final_max_k,
            limit=args.limit,
            dense_batch_size=args.dense_batch_size,
            candidate_strategy=args.candidate_strategy,
            weight_dense=args.weight_dense,
            weight_sparse=args.weight_sparse,
            use_colbert=args.use_colbert,
        )
    except FriendlyFileError as exc:
        LOGGER.error("\n%s", exc)
        return 2
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
