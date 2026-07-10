"""End-to-end ALQAC 2026 pipeline runner — Multi-Retrieval Architecture."""

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

# ── Setup HF cache offline ───────────────────────────────────────────
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

_cache_dir = PROJECT_ROOT / "models"
if _cache_dir.exists():
    os.environ.setdefault("HF_HOME", str(_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_cache_dir))

from tqdm import tqdm

from src.prediction.llm_predictor import LLMPredictor
from src.retrieval.multi_retriever import MultiRetriever
from src.reranking.llm_reranker import LLMReranker
from src.utils.config import load_config
from src.utils.io import FriendlyFileError, load_json_file, resolve_path, setup_logging, validate_case_records

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    test_path: str | Path,
    corpus_path: str | Path,
    output_dir: str | Path,
    rerank_mode: str = "local",
    llm_mode: str = "local",
    # BGE-M3
    bgem3_model: str = "BAAI/bge-m3",
    weight_dense: float = 0.6,
    weight_sparse: float = 0.4,
    # Vietnamese embedding
    vn_model: str = "dangvantuan/vietnamese-embedding",
    # BM25 params
    bm25_top_k_articles: int = 200,
    bm25_top_k_laws: int = 5,
    bm25_strategy: str = "hybrid",
    # Per-retriever top_k
    bgem3_top_k: int = 100,
    vn_top_k: int = 100,
    chunked_top_k: int = 100,
    # Final fusion top_k (input cho LLM reranker)
    fusion_top_k: int = 500,
    # LLM params
    llm1_model: str = "Qwen/Qwen2.5-3B-Instruct",
    llm2_model: str = "Qwen/Qwen3-8B",
    final_min_k: int = 3,
    final_max_k: int = 15,
    # Misc
    batch_size: int = 32,
    use_parallel: bool = True,
    limit: int | None = None,
) -> str:
    """Run Multi-Retrieval pipeline for ALQAC cases."""

    corpus_resolved = resolve_path(corpus_path, PROJECT_ROOT)

    # ── Tầng 1: Multi-Retrieval ──────────────────────────────────────
    LOGGER.info("[Pipeline] Initializing Multi-Retriever (mode=%s) ...", rerank_mode)
    multi_retriever = MultiRetriever(
        mode=rerank_mode,
        corpus_path=corpus_resolved,
        bm25_top_k_articles=bm25_top_k_articles,
        bm25_top_k_laws=bm25_top_k_laws,
        bm25_strategy=bm25_strategy,
        bgem3_top_k=bgem3_top_k,
        vn_top_k=vn_top_k,
        chunked_top_k=chunked_top_k,
        vn_model=vn_model,
        bgem3_model=bgem3_model,
        batch_size=batch_size,
        weight_dense=weight_dense,
        weight_sparse=weight_sparse,
        use_parallel=use_parallel,
    )

    # ── Tầng 2: LLM Reranker ─────────────────────────────────────────
    LOGGER.info("[Pipeline] Loading LLM Reranker (mode=%s) ...", rerank_mode)
    llm_reranker = LLMReranker(mode=rerank_mode, model_name=llm1_model)

    # ── Tầng 3: Predictor ────────────────────────────────────────────
    LOGGER.info("[Pipeline] Loading Predictor (mode=%s) ...", llm_mode)
    predictor = LLMPredictor(mode=llm_mode, model_name=llm2_model)

    # ── Load test data ────────────────────────────────────────────────
    cases = validate_case_records(
        load_json_file(resolve_path(test_path, PROJECT_ROOT), "ALQAC test set")
    )
    if limit is not None:
        cases = cases[:max(0, limit)]
    LOGGER.info("[Pipeline] Loaded %d cases", len(cases))

    # ── Inference loop ────────────────────────────────────────────────
    submissions: list[dict[str, Any]] = []
    for case in tqdm(cases, desc="Pipeline"):
        case_id    = str(case["case_id"])
        case_query = str(case["case_query"])

        # Tầng 1: Multi-retrieval → fused candidate pool
        candidates = multi_retriever.retrieve(case_query, final_top_k=fusion_top_k)

        # Tầng 2: LLM reranker → top-15 relevant articles
        final_articles = llm_reranker.rerank(
            case_query, candidates,
            min_keep=final_min_k,
            max_keep=final_max_k,
        )

        # Tầng 3: Predict verdict
        result = predictor.predict(case_query, final_articles)

        submissions.append({
            "case_id":    case_id,
            "prediction": result.label,
            "confidence": round(float(result.confidence), 4),
            "law_evidence": [
                {"law_id": a.get("law_id"), "aid": a.get("aid")}
                for a in final_articles
            ],
        })

    # ── Save output ───────────────────────────────────────────────────
    out_dir = resolve_path(output_dir, PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = out_dir / f"submission_{rerank_mode}_{llm_mode}_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(submissions, fh, ensure_ascii=False, indent=2)

    dist = Counter(item["prediction"] for item in submissions)
    LOGGER.info("[Pipeline] Output: %s", out_path)
    LOGGER.info("[Pipeline] Cases: %d | Labels: %s", len(submissions), dict(dist))
    return str(out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    cfg = load_config()
    rc  = cfg.get("retrieval", {})
    pc  = cfg.get("prediction", {})

    p = argparse.ArgumentParser(description="ALQAC 2026 Multi-Retrieval Pipeline")

    # Mode
    p.add_argument("--rerank-mode", default="local",  choices=["mock", "local"])
    p.add_argument("--llm-mode",    default="local",  choices=["mock", "local"])

    # Models
    p.add_argument("--bgem3-model", default=rc.get("bgem3_model",    "BAAI/bge-m3"))
    p.add_argument("--vn-model",    default=rc.get("vn_model",       "dangvantuan/vietnamese-embedding"))
    p.add_argument("--llm1-model",  default=pc.get("reranker_model", "Qwen/Qwen2.5-3B-Instruct"))
    p.add_argument("--llm2-model",  default=pc.get("predictor_model","Qwen/Qwen3-8B"))

    # BM25
    p.add_argument("--bm25-top-k-articles", type=int, default=int(rc.get("bm25_top_k_articles", 200)))
    p.add_argument("--bm25-top-k-laws",     type=int, default=int(rc.get("bm25_top_k_laws", 5)))
    p.add_argument("--bm25-strategy",       default=rc.get("bm25_strategy", "hybrid"),
                   choices=["article","law","hybrid"])

    # Per-retriever top_k
    p.add_argument("--bgem3-top-k",   type=int, default=int(rc.get("bgem3_top_k",   100)))
    p.add_argument("--vn-top-k",      type=int, default=int(rc.get("vn_top_k",      100)))
    p.add_argument("--chunked-top-k", type=int, default=int(rc.get("chunked_top_k", 100)))
    p.add_argument("--fusion-top-k",  type=int, default=int(rc.get("fusion_top_k",  500)))

    # BGE-M3 weights
    p.add_argument("--weight-dense",  type=float, default=float(rc.get("weight_dense",  0.6)))
    p.add_argument("--weight-sparse", type=float, default=float(rc.get("weight_sparse", 0.4)))

    # LLM reranker
    p.add_argument("--final-min-k", type=int, default=int(rc.get("final_min_k", 3)))
    p.add_argument("--final-max-k", type=int, default=int(rc.get("final_max_k", 15)))

    # Misc
    p.add_argument("--batch-size",    type=int,  default=int(rc.get("batch_size", 32)))
    p.add_argument("--no-parallel",   action="store_true")
    p.add_argument("--limit",         type=int,  default=None)
    p.add_argument("--test",   default=cfg.get("data",{}).get("public_test",  "data/raw/ALQAC2026_public_test.json"))
    p.add_argument("--corpus", default=cfg.get("data",{}).get("law_corpus",   "data/raw/corpus_law_pub.json"))
    p.add_argument("--output", default=cfg.get("output",{}).get("submissions_dir", "outputs/submissions"))
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> int:
    parser = build_arg_parser()
    args   = parser.parse_args()
    setup_logging(args.log_level)
    try:
        run_pipeline(
            test_path=args.test,
            corpus_path=args.corpus,
            output_dir=args.output,
            rerank_mode=args.rerank_mode,
            llm_mode=args.llm_mode,
            bgem3_model=args.bgem3_model,
            vn_model=args.vn_model,
            llm1_model=args.llm1_model,
            llm2_model=args.llm2_model,
            bm25_top_k_articles=args.bm25_top_k_articles,
            bm25_top_k_laws=args.bm25_top_k_laws,
            bm25_strategy=args.bm25_strategy,
            bgem3_top_k=args.bgem3_top_k,
            vn_top_k=args.vn_top_k,
            chunked_top_k=args.chunked_top_k,
            fusion_top_k=args.fusion_top_k,
            weight_dense=args.weight_dense,
            weight_sparse=args.weight_sparse,
            final_min_k=args.final_min_k,
            final_max_k=args.final_max_k,
            batch_size=args.batch_size,
            use_parallel=not args.no_parallel,
            limit=args.limit,
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
