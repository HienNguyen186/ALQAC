"""Ensemble Reranker Simplified — chỉ Vietnamese Embedding rerank.

Pipeline:
  BM25@1000 → Vietnamese Embedding score & sort → top 20 articles → Qwen3
  
Không dùng CrossEncoder (quá nặng), chỉ dùng Vietnamese embedding đã có sẵn.
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

from src.retrieval.vietnamese_retriever import VietnameseRetriever
from src.utils.io import unique_by_key

LOGGER = logging.getLogger(__name__)


class EnsembleReranker:
    """Reranker Simplified — chỉ Vietnamese Embedding, không CrossEncoder."""

    def __init__(
        self,
        mode: str = "local",
        vn_model: str = "dangvantuan/vietnamese-embedding",
        final_min_k: int = 3,
        final_max_k: int = 20,
        batch_size: int = 32,
        model_cache_dir: str | Path | None = None,
        device: str | None = None,
        # Tham số cũ (bỏ qua)
        cross_model: str | None = None,
        qwen3_model: str | None = None,
        weight_cross: float | None = None,
        weight_vn: float | None = None,
        cross_top_k: int | None = None,
        qwen3_top_k: int | None = None,
        use_llm_filter: bool | None = None,
    ):
        self.mode          = mode
        self.final_min_k   = final_min_k
        self.final_max_k   = final_max_k

        if cross_model or weight_cross or qwen3_model:
            LOGGER.info(
                "[EnsembleReranker] CrossEncoder, Qwen3-Reranker bỏ qua — "
                "chỉ dùng Vietnamese Embedding rerank."
            )

        LOGGER.info("[EnsembleReranker Simplified] Initializing (mode=%s) ...", mode)

        if model_cache_dir:
            _cache = Path(model_cache_dir)
        else:
            _here  = Path(__file__).resolve()
            _cache = next(
                (p / "models" for p in _here.parents if (p / "models").is_dir()),
                None,
            )

        self.vn = VietnameseRetriever(
            mode=mode,
            model_name=vn_model,
            batch_size=batch_size,
            model_cache_dir=_cache,
            device=device,
        )

        LOGGER.info("[EnsembleReranker Simplified] Ready (Vietnamese Embedding only).")

    def rerank(
        self,
        query: str,
        articles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rerank articles bằng Vietnamese Embedding → top final_max_k."""
        candidates = unique_by_key(articles, ("law_id", "aid"))
        if not candidates:
            return []

        if self.mode == "mock":
            return self._mock_rerank(query, candidates)

        LOGGER.info("[EnsembleReranker] Vietnamese Embedding rerank (%d candidates) ...",
                    len(candidates))
        
        # Score & sort bằng Vietnamese Embedding
        vn_ranked = self.vn.retrieve(query, candidates, top_k=len(candidates))
        
        result = vn_ranked[:self.final_max_k]
        if len(result) < self.final_min_k:
            result = vn_ranked[:self.final_min_k]

        for rank, entry in enumerate(result, 1):
            entry["rank"] = rank
            entry["retrieval_score"] = entry.get("vn_score", 0.0)

        LOGGER.info(
            "[EnsembleReranker] Final: %d articles (score range [%.3f, %.3f])",
            len(result),
            result[-1].get("vn_score", 0.0) if result else 0,
            result[0].get("vn_score", 0.0) if result else 0,
        )
        return result

    def _mock_rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mock: dùng retrieval_score, không score lại."""
        sorted_cands = sorted(
            candidates,
            key=lambda x: x.get("retrieval_score", 0.0),
            reverse=True,
        )
        result = sorted_cands[:self.final_max_k]
        for rank, entry in enumerate(result, 1):
            entry["rank"] = rank
        return result