"""
fix_all.py — Chạy file này 1 lần để cập nhật toàn bộ project về đúng phiên bản.

Đặt file này vào thư mục gốc ALQAC/, rồi chạy:
    python fix_all.py

Script sẽ:
  1. Ghi đè tất cả các file đã thay đổi về đúng phiên bản mới nhất
  2. Xóa toàn bộ __pycache__
  3. Báo kết quả từng file
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
print("=" * 70)
print("ALQAC 2026 — Fix All Files")
print(f"Project root: {ROOT}")
print("=" * 70)
print()

FILES: dict[str, str] = {}

# ──────────────────────────────────────────────────────────────────────
FILES["src/reranking/dense_reranker.py"] = '''\
"""Dense retriever/reranker — BGE-M3 hybrid (dense + sparse) hoặc mock fallback.

Chế độ local dùng FlagEmbedding BGEM3FlagModel để kết hợp:
  - Dense embedding  (cosine similarity, dim=1024)
  - Sparse embedding (SPLADE-style lexical weights)

ColBERT bị tắt mặc định: mỗi article cần ~100 token × 1024 dim × float32 ≈ 400 KB,
tổng 3352 articles × 400 KB = 1.3 GB RAM — quá lớn cho CPU.
Bật ColBERT chỉ khi có GPU và set use_colbert=True.

Score cuối: w_dense * dense + w_sparse * sparse  (mặc định 0.6 / 0.4)
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import hashlib
import pickle
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import find_project_root, unique_by_key
from src.utils.text import normalize_text, tokenize

LOGGER = logging.getLogger(__name__)


class DenseRetriever:
    """Re-rank BM25 candidate articles bằng BGE-M3 hybrid scoring.

    Parameters
    ----------
    mode : "mock" | "local"
        mock  — token-overlap + RRF, không cần model.
        local — BGE-M3 qua FlagEmbedding; tự chọn CUDA nếu có, fallback CPU.
    model_name : str
        Hugging Face model ID.  Mặc định "BAAI/bge-m3".
    batch_size : int
        Số article encode mỗi batch.
    weight_dense : float
        Trọng số dense score trong hybrid (0–1).
    weight_sparse : float
        Trọng số sparse score trong hybrid (0–1).
    use_colbert : bool
        Bật ColBERT. Yêu cầu GPU và nhiều RAM.
    cache_dir : Path | None
        Thư mục lưu embedding cache.
    device : str | None
        "cuda" / "cpu" / None (tự detect).
    """

    def __init__(
        self,
        mode: str = "mock",
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        weight_dense: float = 0.6,
        weight_sparse: float = 0.4,
        use_colbert: bool = False,
        cache_dir: str | Path | None = None,
        device: str | None = None,
    ):
        self.mode = mode
        self.model_name = model_name
        self.batch_size = batch_size
        self.weight_dense = weight_dense
        self.weight_sparse = weight_sparse
        self.use_colbert = use_colbert
        self.cache_dir = (
            Path(cache_dir) if cache_dir
            else find_project_root() / "outputs" / "cache" / "embeddings"
        )
        self.device = device
        self._model = None

        if mode == "local":
            self._load_model()

    def _load_model(self) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "DenseRetriever local mode yêu cầu FlagEmbedding.\\n"
                "Cài đặt: pip install FlagEmbedding\\n"
                "Hoặc chạy với --rerank-mode mock để bỏ qua."
            ) from exc

        try:
            import torch
            use_fp16 = torch.cuda.is_available()
            chosen_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            use_fp16 = False
            chosen_device = "cpu"

        LOGGER.info("[DenseRetriever] Loading %s on %s (fp16=%s) ...", self.model_name, chosen_device, use_fp16)
        self._model = BGEM3FlagModel(
            self.model_name,
            use_fp16=use_fp16,
            devices=[chosen_device],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=self.use_colbert,
        )
        LOGGER.info("[DenseRetriever] Ready")

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def _cache_path(self, key_suffix: str) -> Path:
        safe_model = self.model_name.replace("/", "__")
        return self.cache_dir / f"{safe_model}_{key_suffix}.pkl"

    def _pool_signature(self, articles: list[dict[str, Any]]) -> str:
        parts = [
            f"{a.get(\'law_id\')}:{a.get(\'aid\')}:{self._content_hash(str(a.get(\'content\', \'\')))}"
            for a in articles
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    def _load_cache(self, path: Path) -> Any | None:
        if path.exists():
            try:
                with path.open("rb") as fh:
                    return pickle.load(fh)
            except Exception:
                path.unlink(missing_ok=True)
        return None

    def _save_cache(self, path: Path, obj: Any) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(obj, fh)

    def _encode_articles(self, articles: list[dict[str, Any]]) -> dict[str, Any]:
        sig = self._pool_signature(articles)
        cache_path = self._cache_path(f"articles_{sig}")
        cached = self._load_cache(cache_path)
        if cached is not None:
            LOGGER.debug("[DenseRetriever] Article cache hit (%d articles)", len(articles))
            return cached

        assert self._model is not None
        texts = [str(a.get("content", "")) for a in articles]
        LOGGER.info("[DenseRetriever] Encoding %d articles ...", len(texts))
        output = self._model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        result = {
            "dense":  np.asarray(output["dense_vecs"], dtype=np.float32),
            "sparse": output["lexical_weights"],
        }
        self._save_cache(cache_path, result)
        LOGGER.info("[DenseRetriever] Encoded and cached.")
        return result

    def _encode_query(self, query: str) -> dict[str, Any]:
        assert self._model is not None
        output = self._model.encode(
            [query],
            batch_size=1,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense":  np.asarray(output["dense_vecs"][0], dtype=np.float32),
            "sparse": output["lexical_weights"][0],
        }

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-9:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    @staticmethod
    def _sparse_scores(
        query_weights: dict[str, float],
        article_weights_list: list[dict[str, float]],
    ) -> np.ndarray:
        scores = np.zeros(len(article_weights_list), dtype=np.float32)
        for token, q_w in query_weights.items():
            for i, art_w in enumerate(article_weights_list):
                if token in art_w:
                    scores[i] += q_w * art_w[token]
        return scores

    def retrieve(
        self,
        query: str,
        articles: list[dict[str, Any]],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Re-rank articles theo query, trả về top_k với score fields."""
        candidates = unique_by_key(articles, ("law_id", "aid"))
        if not candidates or top_k <= 0:
            return []
        if self.mode == "mock":
            return self._mock_retrieve(query, candidates, top_k)
        if self._model is None:
            self._load_model()
        return self._bgem3_retrieve(query, candidates, top_k)

    def _mock_retrieve(self, query: str, candidates: list, top_k: int) -> list:
        query_terms = set(tokenize(query))
        overlap_scores: list[float] = []
        for art in candidates:
            art_terms = set(tokenize(art.get("content", "")))
            overlap = len(query_terms & art_terms) / max(len(query_terms), 1)
            overlap_scores.append(overlap)

        dense_order = sorted(range(len(candidates)), key=lambda i: (overlap_scores[i], -i), reverse=True)
        dense_rank = {idx: rank for rank, idx in enumerate(dense_order, 1)}

        scored = []
        for idx, art in enumerate(candidates):
            bm25_rank = int(art.get("bm25_rank", idx + 1))
            rrf = 1.0 / (60 + dense_rank[idx]) + 1.0 / (60 + bm25_rank)
            scored.append((rrf, overlap_scores[idx], -idx, art))

        ranked = sorted(scored, key=lambda x: (x[0], x[2]), reverse=True)[:top_k]
        result = []
        for rank, (final_score, ov_score, _, art) in enumerate(ranked, 1):
            entry = dict(art)
            entry["dense_score"]     = round(float(ov_score), 6)
            entry["sparse_score"]    = 0.0
            entry["retrieval_score"] = round(float(final_score), 6)
            entry["rank"]            = rank
            result.append(entry)
        return result

    def _bgem3_retrieve(self, query: str, candidates: list, top_k: int) -> list:
        article_enc = self._encode_articles(candidates)
        query_enc   = self._encode_query(query)

        art_dense  = article_enc["dense"]
        art_sparse = article_enc["sparse"]
        q_dense    = query_enc["dense"]
        q_sparse   = query_enc["sparse"]

        dense_scores  = np.dot(art_dense, q_dense).astype(np.float32)
        sparse_scores = self._sparse_scores(q_sparse, art_sparse)
        hybrid = (
            self.weight_dense  * self._minmax(dense_scores)
            + self.weight_sparse * self._minmax(sparse_scores)
        )

        top_indices = np.argsort(hybrid)[::-1][:top_k]
        result = []
        for rank, idx in enumerate(top_indices, 1):
            entry = dict(candidates[int(idx)])
            entry["dense_score"]     = float(dense_scores[idx])
            entry["sparse_score"]    = float(sparse_scores[idx])
            entry["retrieval_score"] = float(hybrid[idx])
            entry["rank"]            = rank
            result.append(entry)
        return result
'''

# ──────────────────────────────────────────────────────────────────────
FILES["scripts/run_pipeline.py"] = '''\
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

from src.prediction.llm_predictor import LLMPredictor
from src.reranking.dense_reranker import DenseRetriever
from src.reranking.llm_reranker import LLMReranker
from src.retrieval.bm25_retriever import LawRetriever
from src.utils.config import load_config
from src.utils.io import FriendlyFileError, load_json_file, resolve_path, setup_logging, validate_case_records

LOGGER = logging.getLogger(__name__)


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

    cases = validate_case_records(load_json_file(resolve_path(test_path, PROJECT_ROOT), "ALQAC public/private test set"))
    if limit is not None:
        cases = cases[:max(0, limit)]
    LOGGER.info("Loaded %s cases", len(cases))

    submissions: list[dict[str, Any]] = []
    for case in tqdm(cases, desc="Pipeline"):
        case_id    = str(case["case_id"])
        case_query = str(case["case_query"])

        candidate_pool = retriever.retrieve_candidate_pool(
            case_query,
            top_k_articles=top_k_bm25_articles,
            top_k_laws=top_k_laws,
            strategy=candidate_strategy,
        )
        top_articles   = dense.retrieve(case_query, candidate_pool, top_k=top_k_dense)
        final_articles = llm_reranker.rerank(case_query, top_articles, min_keep=final_min_k, max_keep=final_max_k)
        result         = predictor.predict(case_query, final_articles)

        submissions.append({
            "case_id":    case_id,
            "prediction": result.label,
            "confidence": round(float(result.confidence), 4),
            "law_evidence": [
                {"law_id": a.get("law_id"), "aid": a.get("aid")}
                for a in final_articles
            ],
        })

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
    parser.add_argument("--rerank-mode", default="mock", choices=["mock", "local"])
    parser.add_argument("--llm-mode",    default="mock", choices=["mock", "local"])
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
    parser.add_argument("--test",   default=config.get("data", {}).get("public_test", "data/raw/ALQAC2026_public_test.json"))
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
            test_path=args.test,
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
        LOGGER.error("\\n%s", exc)
        return 2
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

# ──────────────────────────────────────────────────────────────────────
FILES["scripts/main.py"] = '''\
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
        LOGGER.error("\\n%s", exc)
        return 2
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

# ──────────────────────────────────────────────────────────────────────
FILES["scripts/evaluate.py"] = '''\
"""Evaluate retrieval and optional prediction outputs for ALQAC public data."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
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

from src.retrieval.bm25_retriever import LawRetriever
from src.retrieval.law_name_map import parse_law_provisions
from src.reranking.dense_reranker import DenseRetriever
from src.reranking.llm_reranker import LLMReranker
from src.utils.io import FriendlyFileError, load_json_file, resolve_path, setup_logging, validate_case_records

LOGGER = logging.getLogger(__name__)


def law_f1(predicted: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[str, float] | None:
    pred_set = {(str(i["law_id"]), int(i["aid"])) for i in predicted if i.get("aid") is not None}
    gt_set   = {(str(i["law_id"]), int(i["aid"])) for i in ground_truth if i.get("aid") is not None}
    if not gt_set:
        return None
    tp        = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(gt_set)
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "pred": len(pred_set), "gt": len(gt_set),
            "precision": precision, "recall": recall, "f1": f1}


def macro_f1(labels: list[str], predictions: list[str]) -> float:
    unique_labels = sorted(set(labels) | set(predictions))
    scores = []
    for label in unique_labels:
        tp = sum(1 for y, p in zip(labels, predictions) if y == label and p == label)
        fp = sum(1 for y, p in zip(labels, predictions) if y != label and p == label)
        fn = sum(1 for y, p in zip(labels, predictions) if y == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall    = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def summarize_metrics(name: str, results: list[dict[str, float]], skipped: int) -> None:
    if not results:
        LOGGER.warning("No cases had parseable ground-truth legal provisions.")
        return
    n      = len(results)
    avg_p  = sum(r["precision"] for r in results) / n
    avg_r  = sum(r["recall"]    for r in results) / n
    avg_f1 = sum(r["f1"]        for r in results) / n
    cov    = sum(1 for r in results if r["tp"] > 0) / n
    LOGGER.info("%s evaluated=%s skipped=%s", name, n, skipped)
    LOGGER.info("Evidence Precision=%.4f Recall@k=%.4f Evidence F1=%.4f Coverage=%.4f",
                avg_p, avg_r, avg_f1, cov)


def evaluate_retrieval(corpus: str | Path, test: str | Path, top_k: int, verbose: bool = False) -> None:
    retriever = LawRetriever(resolve_path(corpus, PROJECT_ROOT))
    cases     = validate_case_records(load_json_file(resolve_path(test, PROJECT_ROOT), "ALQAC public test set"))
    results: list[dict[str, float]] = []
    skipped = 0

    for case in cases:
        gt_list   = parse_law_provisions(case.get("related_law_provisions", ""))
        retrieved = retriever.retrieve(str(case["case_query"]), top_k=top_k)
        pred_list = [{"law_id": i["law_id"], "aid": i["aid"]} for i in retrieved]
        metrics   = law_f1(pred_list, gt_list)
        if metrics is None:
            skipped += 1
            continue
        results.append(metrics)
        if verbose:
            LOGGER.info("%s P=%.2f R=%.2f F1=%.2f", case["case_id"],
                        metrics["precision"], metrics["recall"], metrics["f1"])

    summarize_metrics(f"BM25 top_k={top_k}", results, skipped)


def evaluate_pipeline_retrieval(
    corpus: str | Path,
    test: str | Path,
    top_k_bm25_articles: int,
    top_k_laws: int,
    top_k_dense: int,
    final_k: int,
    candidate_strategy: str,
    rerank_mode: str,
    dense_model: str,
    use_llm_reranker: bool,
    error_analysis: str | Path | None,
    verbose: bool = False,
) -> None:
    retriever   = LawRetriever(resolve_path(corpus, PROJECT_ROOT))
    dense       = DenseRetriever(
        mode=rerank_mode,
        model_name=dense_model,
        weight_dense=0.6,
        weight_sparse=0.4,
    )
    llm_reranker = LLMReranker(mode=rerank_mode) if use_llm_reranker else None
    cases        = validate_case_records(load_json_file(resolve_path(test, PROJECT_ROOT), "ALQAC public test set"))

    results: list[dict[str, float]] = []
    skipped = 0
    rows: list[dict[str, Any]] = []

    for case in cases:
        query   = str(case["case_query"])
        gt_list = parse_law_provisions(case.get("related_law_provisions", ""))
        candidates = retriever.retrieve_candidate_pool(
            query,
            top_k_articles=top_k_bm25_articles,
            top_k_laws=top_k_laws,
            strategy=candidate_strategy,
        )
        reranked = dense.retrieve(query, candidates, top_k=top_k_dense)
        if llm_reranker is not None:
            reranked = llm_reranker.rerank(query, reranked, min_keep=min(2, final_k), max_keep=final_k)
        else:
            reranked = reranked[:final_k]

        pred_list = [{"law_id": i["law_id"], "aid": i["aid"]} for i in reranked]
        metrics   = law_f1(pred_list, gt_list)
        if metrics is None:
            skipped += 1
            continue
        results.append(metrics)

        gt_set   = {(str(i["law_id"]), int(i["aid"])) for i in gt_list}
        pred_set = {(str(i["law_id"]), int(i["aid"])) for i in pred_list if i.get("aid") is not None}
        rows.append({
            "case_id":    case["case_id"],
            "hit":        int(bool(gt_set & pred_set)),
            "precision":  metrics["precision"],
            "recall":     metrics["recall"],
            "gt":         json.dumps(gt_list, ensure_ascii=False),
            "pred":       json.dumps(pred_list, ensure_ascii=False),
            "case_query": query,
        })
        if verbose:
            LOGGER.info("%s hit=%s P=%.2f R=%.2f", case["case_id"],
                        int(bool(gt_set & pred_set)), metrics["precision"], metrics["recall"])

    summarize_metrics(
        f"Pipeline strategy={candidate_strategy} bm25={top_k_bm25_articles} dense={top_k_dense} final={final_k}",
        results, skipped,
    )

    if error_analysis:
        out_path = resolve_path(error_analysis, PROJECT_ROOT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["case_id","hit","precision","recall","gt","pred","case_query"])
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Error analysis written to %s", out_path)


def evaluate_submission(test: str | Path, submission: str | Path) -> None:
    cases   = validate_case_records(load_json_file(resolve_path(test, PROJECT_ROOT), "ALQAC public test set"))
    outputs = load_json_file(resolve_path(submission, PROJECT_ROOT), "pipeline submission")
    by_id   = {str(item["case_id"]): item for item in outputs}
    gold: list[str] = []
    pred: list[str] = []
    for case in cases:
        label  = case.get("verdict_label")
        output = by_id.get(str(case["case_id"]))
        if label and output and output.get("prediction"):
            gold.append(str(label))
            pred.append(str(output["prediction"]))
    if not gold:
        LOGGER.warning("No overlapping labeled cases found.")
        return
    accuracy = sum(1 for y, p in zip(gold, pred) if y == p) / len(gold)
    LOGGER.info("Prediction Accuracy=%.4f MacroF1=%.4f Cases=%s",
                accuracy, macro_f1(gold, pred), len(gold))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ALQAC retrieval and predictions.")
    parser.add_argument("--mode", default="bm25", choices=["bm25", "pipeline"])
    parser.add_argument("--top-k",               type=int, default=20)
    parser.add_argument("--top-k-bm25-articles", type=int, default=200)
    parser.add_argument("--top-k-laws",          type=int, default=5)
    parser.add_argument("--top-k-dense",         type=int, default=30)
    parser.add_argument("--final-k",             type=int, default=15)
    parser.add_argument("--candidate-strategy",  default="hybrid", choices=["article","law","hybrid"])
    parser.add_argument("--rerank-mode",         default="mock",   choices=["mock","local"])
    parser.add_argument("--dense-model",         default="BAAI/bge-m3")
    parser.add_argument("--use-llm-reranker",    action="store_true")
    parser.add_argument("--error-analysis",      default=None)
    parser.add_argument("--verbose",             action="store_true")
    parser.add_argument("--corpus",     default="data/raw/corpus_law_pub.json")
    parser.add_argument("--test",       default="data/raw/ALQAC2026_public_test.json")
    parser.add_argument("--submission", default=None)
    parser.add_argument("--log-level",  default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    try:
        if args.mode == "pipeline":
            evaluate_pipeline_retrieval(
                corpus=args.corpus, test=args.test,
                top_k_bm25_articles=args.top_k_bm25_articles,
                top_k_laws=args.top_k_laws, top_k_dense=args.top_k_dense,
                final_k=args.final_k, candidate_strategy=args.candidate_strategy,
                rerank_mode=args.rerank_mode, dense_model=args.dense_model,
                use_llm_reranker=args.use_llm_reranker,
                error_analysis=args.error_analysis, verbose=args.verbose,
            )
        else:
            evaluate_retrieval(args.corpus, args.test, args.top_k, args.verbose)
        if args.submission:
            evaluate_submission(args.test, args.submission)
    except FriendlyFileError as exc:
        LOGGER.error("\\n%s", exc)
        return 2
    except Exception as exc:
        LOGGER.exception("Evaluation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

# ══════════════════════════════════════════════════════════════════════
# Write all files
# ══════════════════════════════════════════════════════════════════════
errors = []
for rel_path, content in FILES.items():
    target = ROOT / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(content, encoding="utf-8")
        print(f"✓  {rel_path}")
    except Exception as e:
        print(f"✗  {rel_path}  — {e}")
        errors.append(rel_path)

# ══════════════════════════════════════════════════════════════════════
# Clear pycache
# ══════════════════════════════════════════════════════════════════════
print()
count = 0
for d in ROOT.rglob("__pycache__"):
    shutil.rmtree(d, ignore_errors=True)
    count += 1
print(f"✓  Cleared {count} __pycache__ directories")

# ══════════════════════════════════════════════════════════════════════
print()
if errors:
    print(f"⚠  {len(errors)} file(s) failed: {errors}")
    sys.exit(1)
else:
    print("=" * 70)
    print("✅  All files updated successfully!")
    print("=" * 70)
    print()
    print("Chạy thử ngay:")
    print("  python scripts\\run_pipeline.py --rerank-mode mock --llm-mode mock --limit 5")
    print()
    print("Chạy local BGE-M3 (cần models/ đã download):")
    print("  python scripts\\run_pipeline.py --rerank-mode local --llm-mode mock")
