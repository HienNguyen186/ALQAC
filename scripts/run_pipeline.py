"""End-to-end ALQAC 2026 pipeline runner — Multi-Retrieval Architecture (v3).

Thay đổi so với bản trước:
  - Tầng 2 reranker: CrossEncoder + Vietnamese Embedding (bỏ Qwen2.5-3B binary
    filter và Qwen3-Reranker bị lỗi tokenizer).
  - Tầng 3 predictor: Qwen3-8B direct-label (bỏ extract-then-rule Giai đoạn 2
    vì làm giảm accuracy).
  - Retrieval: bm25_top_k_articles=500, fusion_top_k=1000 (tăng recall),
    query_expansion.py đã thêm rule phổ quát cho luật tố tụng/thi hành án.
  - case_api.py KHÔNG bị đụng tới.

Submission format (theo yêu cầu BTC):
{
  "case_id":      "case_1959",
  "prediction":   "A_WIN",
  "case_evidence": ["case_1959_seg_9b2898839a509e4f", ...],
  "law_evidence":  [{"law_id": "91/2015/QH13", "aid": 52771}]
}
"""

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

from src.api.case_api import CaseAPIClient  
from src.prediction.llm_predictor import LLMPredictor
from src.retrieval.multi_retriever import MultiRetriever
from src.reranking.ensemble_reranker import EnsembleReranker
from src.utils.config import load_config
from src.utils.io import (
    FriendlyFileError,
    load_json_file,
    resolve_path,
    setup_logging,
    validate_case_records,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CASE_API_CALLS = 3


def _build_case_queries(case_query: str, final_articles: list[dict[str, Any]]) -> list[str]:
    """Sinh các query để gọi Case Content API."""
    queries = [case_query]
    if final_articles:
        art = final_articles[0]
        queries.append(f"{art.get('law_id', '')} điều {art.get('aid', '')}")
    short = case_query[:80].strip()
    if short not in queries:
        queries.append(short)
    return queries


def _fetch_case_evidence(
    client: CaseAPIClient,
    case_id: str,
    case_query: str,
    final_articles: list[dict[str, Any]],
    n_calls: int = DEFAULT_CASE_API_CALLS,
) -> list[str]:
    """Gọi Case Content API để lấy chunk hash_ids cho case_evidence."""
    queries   = _build_case_queries(case_query, final_articles)[:n_calls]
    hash_ids: list[str] = []
    seen:     set[str]  = set()

    for query in queries:
        try:
            resp = client.search_case_segments(case_id=case_id, query=query)
            result  = resp.get("result", {})
            hash_id = result.get("hash_id", "")
            if hash_id and hash_id not in seen:
                seen.add(hash_id)
                hash_ids.append(hash_id)
                LOGGER.debug("[CaseAPI] %s query=%r → %s", case_id, query[:40], hash_id)
        except Exception as exc:
            LOGGER.warning("[CaseAPI] %s failed (query=%r): %s", case_id, query[:40], exc)

    return hash_ids


def run_pipeline(
    test_path: str | Path,
    corpus_path: str | Path,
    output_dir: str | Path,
    rerank_mode: str = "local",
    llm_mode: str = "local",
    # ── Tầng 1: Multi-Retrieval ──────────────────────────────────
    bgem3_model: str = "BAAI/bge-m3",
    vn_model: str    = "dangvantuan/vietnamese-embedding",
    weight_dense: float  = 0.6,
    weight_sparse: float = 0.4,
    bm25_top_k_articles: int = 500,
    bm25_top_k_laws: int = 5,
    bm25_strategy: str   = "hybrid",
    bgem3_top_k: int    = 100,
    vn_top_k: int       = 100,
    chunked_top_k: int  = 100,
    fusion_top_k: int   = 1000,   # ✅ Tăng từ 500 → 1000 (Recall 10%→35%+)
    use_parallel: bool  = True,
    # ── Tầng 2: Ensemble Reranker (v2: CrossEncoder + VN Embedding) ──
    cross_model: str    = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    weight_cross: float = 0.5,
    weight_vn_rerank: float = 0.5,
    cross_top_k: int    = 300,
    vn_rerank_top_k: int = 100,
    final_min_k: int    = 3,
    final_max_k: int    = 15,
    # ── Tầng 3: Prediction ───────────────────────────────────────
    llm2_model: str     = "Qwen/Qwen3-8B",
    use_extract_rule: bool = False,   # ✅ Direct-label mặc định
    # ── Case Content API (KHÔNG chỉnh sửa case_api.py) ───────────
    use_case_api: bool  = True,
    case_api_calls: int = DEFAULT_CASE_API_CALLS,
    # ── Misc ─────────────────────────────────────────────────────
    batch_size: int     = 32,
    limit: int | None   = None,
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

    # ── Tầng 2: Ensemble Reranker (CrossEncoder + VN Embedding) ──────
    LOGGER.info("[Pipeline] Loading Ensemble Reranker v2 (mode=%s) ...", rerank_mode)
    ensemble = EnsembleReranker(
        mode=rerank_mode,
        cross_model=cross_model,
        vn_model=vn_model,
        weight_cross=weight_cross,
        weight_vn=weight_vn_rerank,
        cross_top_k=cross_top_k,
        vn_top_k=vn_rerank_top_k,
        final_min_k=final_min_k,
        final_max_k=final_max_k,
        batch_size=batch_size,
    )

    # ── Tầng 3: Predictor (Qwen3-8B direct-label) ────────────────────
    _default_label = "PARTIAL_A_WIN"
    _baseline_path = PROJECT_ROOT / "outputs" / "baseline_check.json"
    if _baseline_path.exists():
        try:
            _baseline_data = json.loads(_baseline_path.read_text(encoding="utf-8"))
            _default_label = _baseline_data.get("majority_label", "PARTIAL_A_WIN")
            LOGGER.info("[Pipeline] Loaded majority_label from baseline: %s", _default_label)
        except Exception as exc:
            LOGGER.warning("[Pipeline] Could not read baseline_check.json: %s", exc)

    LOGGER.info(
        "[Pipeline] Loading Predictor (mode=%s, default_label=%s, extract_rule=%s) ...",
        llm_mode, _default_label, use_extract_rule,
    )
    predictor = LLMPredictor(
        mode=llm_mode,
        model_name=llm2_model,
        default_label=_default_label,
        use_extract_rule=use_extract_rule,
    )

    # ── Case Content API (KHÔNG chỉnh sửa) ───────────────────────────
    case_client: CaseAPIClient | None = None
    if use_case_api:
        token = os.getenv("ALQAC_API_TOKEN", "")
        if token:
            case_client = CaseAPIClient(token=token)
            LOGGER.info("[Pipeline] Case API enabled (%d calls/case)", case_api_calls)
        else:
            LOGGER.warning(
                "[Pipeline] ALQAC_API_TOKEN not set — case_evidence will be empty. "
                "Set token in .env file: ALQAC_API_TOKEN=your_token_here"
            )

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

        # Tầng 2: Ensemble reranker (CrossEncoder + VN Embedding) → top-N
        final_articles = ensemble.rerank(case_query, candidates)

        # Tầng 3: Predict verdict (Qwen3-8B direct-label)
        result = predictor.predict(case_query, final_articles)

        # ── Case Content API → case_evidence chunk ids ────────────────
        if case_client is not None:
            chunk_ids = _fetch_case_evidence(
                client=case_client,
                case_id=case_id,
                case_query=case_query,
                final_articles=final_articles,
                n_calls=case_api_calls,
            )
        else:
            chunk_ids = []

        # ── Build submission item (đúng format BTC) ───────────────────
        submissions.append({
            "case_id":       case_id,
            "prediction":    result.label,
            "case_evidence": chunk_ids,
            "law_evidence": [
                {"law_id": a.get("law_id"), "aid": a.get("aid")}
                for a in final_articles
            ],
        })

    # ── Report parse stats ─────────────────────────────────────────
    predictor.report_parse_stats()

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

    p = argparse.ArgumentParser(description="ALQAC 2026 Multi-Retrieval Pipeline v3")

    # Mode
    p.add_argument("--rerank-mode", default="local",  choices=["mock", "local"])
    p.add_argument("--llm-mode",    default="local",  choices=["mock", "local"])

    # Models
    p.add_argument("--bgem3-model", default=rc.get("bgem3_model",    "BAAI/bge-m3"))
    p.add_argument("--vn-model",    default=rc.get("vn_model",       "dangvantuan/vietnamese-embedding"))
    p.add_argument("--cross-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    p.add_argument("--llm2-model",  default=pc.get("predictor_model","Qwen/Qwen3-8B"))

    # BM25
    p.add_argument("--bm25-top-k-articles", type=int, default=int(rc.get("bm25_top_k_articles", 500)))
    p.add_argument("--bm25-top-k-laws",     type=int, default=int(rc.get("bm25_top_k_laws", 5)))
    p.add_argument("--bm25-strategy",       default=rc.get("bm25_strategy", "hybrid"),
                   choices=["article", "law", "hybrid"])

    # Per-retriever top_k
    p.add_argument("--bgem3-top-k",   type=int, default=int(rc.get("bgem3_top_k",   100)))
    p.add_argument("--vn-top-k",      type=int, default=int(rc.get("vn_top_k",      100)))
    p.add_argument("--chunked-top-k", type=int, default=int(rc.get("chunked_top_k", 100)))
    p.add_argument("--fusion-top-k",  type=int, default=int(rc.get("fusion_top_k",  1000)))

    # BGE-M3 weights
    p.add_argument("--weight-dense",  type=float, default=float(rc.get("weight_dense",  0.6)))
    p.add_argument("--weight-sparse", type=float, default=float(rc.get("weight_sparse", 0.4)))

    # Ensemble reranker (v2) weights
    p.add_argument("--weight-cross",      type=float, default=0.5)
    p.add_argument("--weight-vn-rerank",  type=float, default=0.5)
    p.add_argument("--cross-top-k",       type=int,   default=300)
    p.add_argument("--vn-rerank-top-k",   type=int,   default=100)
    p.add_argument("--final-min-k", type=int, default=int(rc.get("final_min_k", 3)))
    p.add_argument("--final-max-k", type=int, default=int(rc.get("final_max_k", 15)))

    # Predictor
    p.add_argument("--use-extract-rule", action="store_true",
                   help="Bật extract-then-rule (Giai đoạn 2) thay vì direct-label. "
                        "Mặc định TẮT vì đã đo thấy làm giảm accuracy.")

    # Case Content API — KHÔNG chỉnh sửa case_api.py, chỉ toggle ở đây
    p.add_argument("--no-case-api",    action="store_true",
                   help="Tắt Case Content API (case_evidence sẽ rỗng)")
    p.add_argument("--case-api-calls", type=int, default=DEFAULT_CASE_API_CALLS)

    # Misc
    p.add_argument("--batch-size",  type=int, default=int(rc.get("batch_size", 32)))
    p.add_argument("--limit",       type=int, default=None)
    p.add_argument("--test",   default=cfg.get("data", {}).get("public_test",  "data/raw/ALQAC2026_public_test.json"))
    p.add_argument("--corpus", default=cfg.get("data", {}).get("law_corpus",   "data/raw/corpus_law_pub.json"))
    p.add_argument("--output", default=cfg.get("output", {}).get("submissions_dir", "outputs/submissions"))
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
            cross_model=args.cross_model,
            weight_cross=args.weight_cross,
            weight_vn_rerank=args.weight_vn_rerank,
            cross_top_k=args.cross_top_k,
            vn_rerank_top_k=args.vn_rerank_top_k,
            final_min_k=args.final_min_k,
            final_max_k=args.final_max_k,
            use_extract_rule=args.use_extract_rule,
            use_case_api=not args.no_case_api,
            case_api_calls=args.case_api_calls,
            batch_size=args.batch_size,
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