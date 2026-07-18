"""Ensemble Reranker v2 — Tầng 2 của pipeline.

THAY ĐỔI: Bỏ Qwen2.5-3B (LLMReranker binary CO/KHONG — chậm, dễ lỗi, hay
fallback=True khi crash) và Qwen3-Reranker (đã bị disable do tokenizer
incompatibility, luôn trả về 0.0). Thay bằng Vietnamese Embedding
(dangvantuan/vietnamese-embedding) — model dense được pretrain trên văn bản
tiếng Việt, cho relevance score liên tục [0,1] (có gradient, không mất
thông tin như binary), và không có vấn đề tương thích tokenizer.

Kiến trúc mới:

  ┌─ CrossEncoder (mmarco-MiniLM)         score → cross_score
  └─ Vietnamese Embedding (PhoBERT-based) score → vn_score
        ↓
  Ensemble: w_cross × cross_score + w_vn × vn_score
        ↓ top-k

Lợi ích so với bản cũ:
  - Không load Qwen2.5-3B (~7GB) và Qwen3-Reranker-0.6B → giải phóng VRAM
    cho Qwen3-8B predictor chạy full-precision hơn.
  - Nhanh hơn nhiều (không còn O(N) lần generate() của LLMReranker).
  - Vietnamese Embedding cho relevance liên tục, không bị "tất cả pass"
    khi model crash như binary filter cũ.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.vietnamese_retriever import VietnameseRetriever
from src.utils.io import unique_by_key

LOGGER = logging.getLogger(__name__)


class EnsembleReranker:
    """Ensemble CrossEncoder + Vietnamese Embedding → top-k articles.

    Parameters
    ----------
    mode : "mock" | "local"
    cross_model : str       CrossEncoder model
    vn_model : str          Vietnamese Embedding model (dangvantuan/vietnamese-embedding)
    weight_cross : float    Trọng số CrossEncoder (default 0.5)
    weight_vn : float       Trọng số Vietnamese Embedding (default 0.5)
    cross_top_k : int       CrossEncoder chỉ score top-N từ fusion pool
    vn_top_k : int          Vietnamese Embedding rerank top-N sau CrossEncoder
    final_min_k : int       Tối thiểu articles output
    final_max_k : int       Tối đa articles output
    batch_size : int
    model_cache_dir : Path | None
    device : str | None
    """

    def __init__(
        self,
        mode: str = "local",
        cross_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        vn_model: str = "dangvantuan/vietnamese-embedding",
        weight_cross: float = 0.5,
        weight_vn: float = 0.5,
        cross_top_k: int = 300,
        vn_top_k: int = 100,
        final_min_k: int = 3,
        final_max_k: int = 15,
        batch_size: int = 32,
        model_cache_dir: str | Path | None = None,
        device: str | None = None,
        # Tham số cũ được giữ lại (không dùng) để không phải sửa mọi call-site
        # nếu bạn còn code khác gọi EnsembleReranker với các kwargs cũ.
        qwen3_model: str | None = None,
        llm_model: str | None = None,
        weight_qwen3: float | None = None,
        weight_llm: float | None = None,
        qwen3_top_k: int | None = None,
        use_llm_filter: bool | None = None,
    ):
        self.mode         = mode
        self.weight_cross = weight_cross
        self.weight_vn    = weight_vn
        self.cross_top_k  = cross_top_k
        self.vn_top_k     = vn_top_k
        self.final_min_k  = final_min_k
        self.final_max_k  = final_max_k

        if qwen3_model or llm_model or use_llm_filter:
            LOGGER.info(
                "[EnsembleReranker] Ghi chú: qwen3_model/llm_model/use_llm_filter "
                "không còn dùng trong v2 (đã thay bằng Vietnamese Embedding)."
            )

        # Tìm model cache dir
        if model_cache_dir:
            _cache = Path(model_cache_dir)
        else:
            _here  = Path(__file__).resolve()
            _cache = next(
                (p / "models" for p in _here.parents if (p / "models").is_dir()),
                None,
            )

        LOGGER.info("[EnsembleReranker v2] Initializing (mode=%s) ...", mode)

        # ── CrossEncoder ──────────────────────────────────────────────
        LOGGER.info("[EnsembleReranker] [1/2] CrossEncoder ...")
        self.cross = CrossEncoderReranker(
            mode=mode,
            model_name=cross_model,
            batch_size=batch_size * 2,
            model_cache_dir=_cache,
            device=device,
        )

        # ── Vietnamese Embedding ──────────────────────────────────────
        LOGGER.info("[EnsembleReranker] [2/2] Vietnamese Embedding ...")
        self.vn = VietnameseRetriever(
            mode=mode,
            model_name=vn_model,
            batch_size=batch_size * 2,
            model_cache_dir=_cache,
            device=device,
        )

        LOGGER.info("[EnsembleReranker v2] Ready.")

    # ------------------------------------------------------------------
    # Normalize helper
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-9:
            return np.ones_like(arr, dtype=np.float32) * 0.5
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    # ------------------------------------------------------------------
    # Main rerank
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        articles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Ensemble rerank → top final_max_k articles.

        Pipeline:
          1. CrossEncoder score toàn bộ candidates → top cross_top_k
          2. Vietnamese Embedding rerank lại top cross_top_k → top vn_top_k
             (dùng để đo lường; điểm cuối vẫn tính trên union top cross_top_k)
          3. Weighted ensemble (CrossEncoder + VN Embedding) → final ranking
        """
        candidates = unique_by_key(articles, ("law_id", "aid"))
        if not candidates:
            return []

        if self.mode == "mock":
            return self._mock_rerank(query, candidates)

        # ── Bước 1: CrossEncoder score toàn bộ → top cross_top_k ─────
        LOGGER.info("[EnsembleReranker] Step 1: CrossEncoder (%d candidates) ...", len(candidates))
        cross_scores = self.cross.score(query, candidates)
        cross_order   = np.argsort(cross_scores)[::-1]
        top_idx       = cross_order[:self.cross_top_k]
        top_candidates = [candidates[i] for i in top_idx]
        top_cross_scores = cross_scores[top_idx]

        # ── Bước 2: Vietnamese Embedding score trên top_candidates ───
        LOGGER.info("[EnsembleReranker] Step 2: Vietnamese Embedding (%d candidates) ...",
                    len(top_candidates))
        vn_ranked = self.vn.retrieve(query, top_candidates, top_k=len(top_candidates))
        vn_score_map: dict[tuple, float] = {
            (str(a.get("law_id")), a.get("aid")): float(a.get("vn_score", 0.0))
            for a in vn_ranked
        }

        # ── Bước 3: Ensemble score ────────────────────────────────────
        LOGGER.info("[EnsembleReranker] Step 3: Ensemble scoring ...")
        norm_cross = self._minmax(top_cross_scores)
        raw_vn = np.array(
            [vn_score_map.get((str(a.get("law_id")), a.get("aid")), 0.0) for a in top_candidates],
            dtype=np.float32,
        )
        norm_vn = self._minmax(raw_vn)

        scored_entries = []
        for i, art in enumerate(top_candidates):
            ensemble = self.weight_cross * float(norm_cross[i]) + self.weight_vn * float(norm_vn[i])
            entry = dict(art)
            entry["cross_score"]     = float(top_cross_scores[i])
            entry["vn_score"]        = float(raw_vn[i])
            entry["ensemble_score"]  = round(ensemble, 6)
            entry["retrieval_score"] = round(ensemble, 6)
            scored_entries.append(entry)

        scored_entries.sort(key=lambda x: x["ensemble_score"], reverse=True)

        result = scored_entries[:self.final_max_k]
        if len(result) < self.final_min_k:
            result = scored_entries[:self.final_min_k]

        for rank, entry in enumerate(result, 1):
            entry["rank"] = rank

        LOGGER.info(
            "[EnsembleReranker] Final: %d articles (score range [%.3f, %.3f])",
            len(result),
            result[-1]["ensemble_score"] if result else 0,
            result[0]["ensemble_score"]  if result else 0,
        )
        return result

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _mock_rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mock: dùng rrf_score để rank, không cần model."""
        sorted_cands = sorted(
            candidates,
            key=lambda x: x.get("rrf_score", x.get("retrieval_score", 0.0)),
            reverse=True,
        )
        result = sorted_cands[:self.final_max_k]
        for rank, entry in enumerate(result, 1):
            entry["ensemble_score"] = entry.get("rrf_score", 0.0)
            entry["rank"] = rank
        return result