"""Ensemble Reranker — Tầng 2 của pipeline mới.

Kết hợp 3 reranker bằng weighted score:

  ┌─ CrossEncoder (mmarco-MiniLM)    score → cross_score
  ├─ Qwen3-Reranker-0.6B             score → qwen3_score
  └─ LLMReranker (Qwen2.5-3B binary) pass/fail → llm_relevant
        ↓
  Ensemble: w1×cross_score + w2×qwen3_score + w3×llm_boost
        ↓ top-15

Lý do ensemble:
  - CrossEncoder: nhanh, chạy toàn bộ 500 candidates, recall tốt
  - Qwen3Reranker: chính xác hơn cho tiếng Việt pháp lý, precision cao
  - LLMReranker: hard filter bổ sung (loại bỏ article rõ ràng không liên quan)
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
from src.reranking.qwen3_reranker import Qwen3Reranker
from src.reranking.llm_reranker import LLMReranker
from src.utils.io import unique_by_key

LOGGER = logging.getLogger(__name__)


class EnsembleReranker:
    """Ensemble 3 reranker → top-k articles với score tổng hợp.

    Parameters
    ----------
    mode : "mock" | "local"
    cross_model : str       CrossEncoder model
    qwen3_model : str       Qwen3-Reranker model
    llm_model : str         LLMReranker (Qwen2.5) model
    weight_cross : float    Trọng số CrossEncoder (default 0.35)
    weight_qwen3 : float    Trọng số Qwen3Reranker (default 0.50)
    weight_llm : float      Boost khi LLMReranker = CO (default 0.15)
    cross_top_k : int       CrossEncoder chỉ score top-N từ RRF để tiết kiệm thời gian
    qwen3_top_k : int       Qwen3Reranker score top-N sau CrossEncoder
    final_min_k : int       Tối thiểu articles output
    final_max_k : int       Tối đa articles output
    use_llm_filter : bool   Có dùng LLMReranker không (chậm nhưng tốt hơn)
    batch_size : int
    model_cache_dir : Path | None
    device : str | None
    """

    def __init__(
        self,
        mode: str = "local",
        cross_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        qwen3_model: str = "Qwen/Qwen3-Reranker-0.6B",
        llm_model: str   = "Qwen/Qwen2.5-3B-Instruct",
        weight_cross: float = 0.35,
        weight_qwen3: float = 0.50,
        weight_llm: float   = 0.15,
        cross_top_k: int  = 200,   # Score top-200 bằng CrossEncoder trước
        qwen3_top_k: int  = 50,    # Score top-50 bằng Qwen3 sau CrossEncoder
        final_min_k: int  = 3,
        final_max_k: int  = 15,
        use_llm_filter: bool = True,
        batch_size: int   = 32,
        model_cache_dir: str | Path | None = None,
        device: str | None = None,
    ):
        self.mode         = mode
        self.weight_cross = weight_cross
        self.weight_qwen3 = weight_qwen3
        self.weight_llm   = weight_llm
        self.cross_top_k  = cross_top_k
        self.qwen3_top_k  = qwen3_top_k
        self.final_min_k  = final_min_k
        self.final_max_k  = final_max_k
        self.use_llm_filter = use_llm_filter

        # Tìm model cache dir
        if model_cache_dir:
            _cache = Path(model_cache_dir)
        else:
            _here  = Path(__file__).resolve()
            _cache = next(
                (p / "models" for p in _here.parents if (p / "models").is_dir()),
                None,
            )

        LOGGER.info("[EnsembleReranker] Initializing (mode=%s) ...", mode)

        # ── CrossEncoder ──────────────────────────────────────────────
        LOGGER.info("[EnsembleReranker] [1/3] CrossEncoder ...")
        self.cross = CrossEncoderReranker(
            mode=mode,
            model_name=cross_model,
            batch_size=batch_size * 2,   # CrossEncoder nhỏ → batch lớn OK
            model_cache_dir=_cache,
            device=device,
        )

        # ── Qwen3-Reranker ────────────────────────────────────────────
        LOGGER.info("[EnsembleReranker] [2/3] Qwen3Reranker ...")
        self.qwen3 = Qwen3Reranker(
            mode=mode,
            model_name=qwen3_model,
            batch_size=8,
            model_cache_dir=_cache,
            device=device,
        )

        # ── LLMReranker (optional hard filter) ───────────────────────
        if use_llm_filter:
            LOGGER.info("[EnsembleReranker] [3/3] LLMReranker ...")
            self.llm = LLMReranker(
                mode=mode,
                model_name=llm_model,
                cache_dir=_cache,
            )
        else:
            self.llm = None
            LOGGER.info("[EnsembleReranker] [3/3] LLMReranker skipped (use_llm_filter=False)")

        LOGGER.info("[EnsembleReranker] All rerankers ready.")

    # ------------------------------------------------------------------
    # Normalize helper
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
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
          1. CrossEncoder score toàn bộ candidates (top cross_top_k)
          2. Qwen3Reranker score top qwen3_top_k sau CrossEncoder
          3. LLMReranker hard filter (optional)
          4. Weighted ensemble → final ranking
        """
        candidates = unique_by_key(articles, ("law_id", "aid"))
        if not candidates:
            return []

        if self.mode == "mock":
            return self._mock_rerank(query, candidates)

        # ── Bước 1: CrossEncoder score toàn bộ → top cross_top_k ─────
        LOGGER.info("[EnsembleReranker] Step 1: CrossEncoder (%d candidates) ...", len(candidates))
        cross_scores = self.cross.score(query, candidates)

        # Lấy top cross_top_k để đưa vào Qwen3
        cross_order  = np.argsort(cross_scores)[::-1]
        top_cross_idx = cross_order[:self.cross_top_k]
        top_candidates = [candidates[i] for i in top_cross_idx]
        top_cross_scores = cross_scores[top_cross_idx]

        # ── Bước 2: Qwen3Reranker score top qwen3_top_k ──────────────
        # Chỉ lấy top qwen3_top_k từ CrossEncoder để tiết kiệm VRAM
        qwen3_input   = top_candidates[:self.qwen3_top_k]
        qwen3_input_cross = top_cross_scores[:self.qwen3_top_k]

        LOGGER.info("[EnsembleReranker] Step 2: Qwen3Reranker (%d candidates) ...", len(qwen3_input))
        qwen3_scores = self.qwen3.score(query, qwen3_input)

        # Bản đồ (law_id, aid) → qwen3 score
        qwen3_map: dict[tuple, float] = {}
        for art, score in zip(qwen3_input, qwen3_scores):
            key = (str(art.get("law_id")), art.get("aid"))
            qwen3_map[key] = float(score)

        # ── Bước 3: LLM hard filter (optional) ───────────────────────
        llm_pass_set: set[tuple] = set()
        if self.llm is not None and self.use_llm_filter:
            LOGGER.info("[EnsembleReranker] Step 3: LLMReranker filter (%d candidates) ...", len(qwen3_input))
            # LLM filter chỉ trên qwen3_input (top-50)
            llm_result = self.llm.rerank(
                query, qwen3_input,
                min_keep=self.final_min_k,
                max_keep=len(qwen3_input),
            )
            for art in llm_result:
                if art.get("llm_relevant"):
                    llm_pass_set.add((str(art.get("law_id")), art.get("aid")))

        # ── Bước 4: Ensemble score ────────────────────────────────────
        LOGGER.info("[EnsembleReranker] Step 4: Ensemble scoring ...")

        # Normalize cross scores của top_candidates
        norm_cross = self._minmax(top_cross_scores)

        scored_entries = []
        for i, (art, c_score) in enumerate(zip(top_candidates, norm_cross)):
            key = (str(art.get("law_id")), art.get("aid"))

            # Qwen3 score (0 nếu không trong qwen3_input)
            q_score = qwen3_map.get(key, 0.0)

            # LLM boost
            llm_boost = 1.0 if key in llm_pass_set else 0.0

            # Weighted ensemble
            if self.llm is not None and self.use_llm_filter:
                ensemble = (
                    self.weight_cross * float(c_score)
                    + self.weight_qwen3 * q_score
                    + self.weight_llm   * llm_boost
                )
            else:
                # Không có LLM filter → phân phối lại weight
                w_c = self.weight_cross / (self.weight_cross + self.weight_qwen3)
                w_q = self.weight_qwen3 / (self.weight_cross + self.weight_qwen3)
                ensemble = w_c * float(c_score) + w_q * q_score

            entry = dict(art)
            entry["cross_score"]    = float(cross_scores[cross_order[i]])
            entry["qwen3_score"]    = q_score
            entry["llm_relevant"]   = key in llm_pass_set if self.use_llm_filter else None
            entry["ensemble_score"] = round(ensemble, 6)
            entry["retrieval_score"] = round(ensemble, 6)
            scored_entries.append(entry)

        # Sort by ensemble score
        scored_entries.sort(key=lambda x: x["ensemble_score"], reverse=True)

        # Giữ final_max_k, đảm bảo tối thiểu final_min_k
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
