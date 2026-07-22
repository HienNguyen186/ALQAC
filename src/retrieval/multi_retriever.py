"""Multi-Retriever v2 Simplified — BM25 + Vietnamese Embedding only.

Bỏ BGE-M3, Chunked, VietnameseRetriever dense search — chỉ dùng:
  1. BM25 (keyword) → top 500
  2. Vietnamese S-BERT/Embedding (dense) → top 200
  3. RRF Fusion → top 500 candidates

Lợi ích: nhẹ hơn, nhanh hơn, nhưng vẫn đạt Recall tốt.
"""

from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.bm25_retriever import LawRetriever
from src.retrieval.vietnamese_retriever import VietnameseRetriever
from src.utils.io import unique_by_key

LOGGER = logging.getLogger(__name__)


def reciprocal_rank_fusion(rankings: list[dict[tuple, float]]) -> dict[tuple, float]:
    """RRF — hợp gộp rankings từ nhiều retriever."""
    rrf_scores: dict[tuple, float] = {}
    for ranking in rankings:
        for rank, (key, score) in enumerate(sorted(ranking.items(), key=lambda x: x[1], reverse=True), 1):
            if key not in rrf_scores:
                rrf_scores[key] = 0.0
            rrf_scores[key] += 1.0 / (60 + rank)  # RRF formula: 1/(k + rank), k=60
    return rrf_scores


class MultiRetriever:
    """Multi-Retrieval v2 — BM25 + Vietnamese Embedding only."""

    def __init__(
        self,
        mode: str = "local",
        corpus_path: str | Path | None = None,
        bm25_top_k_articles: int = 500,
        bm25_top_k_laws: int = 5,
        bm25_strategy: str = "hybrid",
        vn_top_k: int = 200,
        vn_model: str = "dangvantuan/vietnamese-embedding",
        fusion_top_k: int = 500,
        batch_size: int = 32,
        model_cache_dir: str | Path | None = None,
        device: str | None = None,
        use_parallel: bool = False,
        # Tham số cũ (không dùng, giữ lại để không phá API cũ)
        bgem3_model: str | None = None,
        bgem3_top_k: int | None = None,
        weight_dense: float | None = None,
        weight_sparse: float | None = None,
        chunked_top_k: int | None = None,
    ):
        self.mode              = mode
        self.bm25_top_k_articles = bm25_top_k_articles
        self.bm25_top_k_laws   = bm25_top_k_laws
        self.bm25_strategy     = bm25_strategy
        self.vn_top_k          = vn_top_k
        self.fusion_top_k      = fusion_top_k
        self.batch_size        = batch_size

        if bgem3_model or weight_dense or chunked_top_k:
            LOGGER.info(
                "[MultiRetriever] Ghi chú: BGE-M3, weight_dense/sparse, chunked_top_k "
                "không còn dùng trong v2 Simplified. Chỉ BM25 + Vietnamese Embedding."
            )

        LOGGER.info("[MultiRetriever v2 Simplified] Initializing (mode=%s) ...", mode)

        # ── BM25 ──────────────────────────────────────────────────────
        LOGGER.info("[MultiRetriever] [1/2] BM25 Retriever ...")
        # strategy is a param of retrieve_candidate_pool(), not __init__()
        self.bm25 = LawRetriever(
            corpus_path=Path(corpus_path) if corpus_path else None,
        )

        # ── Vietnamese Embedding ──────────────────────────────────────
        LOGGER.info("[MultiRetriever] [2/2] Vietnamese Embedding Retriever ...")
        self.vn = VietnameseRetriever(
            mode=mode,
            model_name=vn_model,
            batch_size=batch_size,
            model_cache_dir=model_cache_dir,
            device=device,
        )

        LOGGER.info("[MultiRetriever v2 Simplified] Ready (BM25 + VN only).")

    # ------------------------------------------------------------------

    def retrieve(self, query: str, final_top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve articles: BM25 + Vietnamese Embedding → RRF → top-k.

        Parameters
        ----------
        query : str
        final_top_k : int, optional
            Override self.fusion_top_k

        Returns
        -------
        List of articles, sorted by RRF score, top final_top_k items.
        """
        top_k = final_top_k if final_top_k is not None else self.fusion_top_k

        if self.mode == "mock":
            return self._mock_retrieve(query, top_k)

        # ── BM25 ──────────────────────────────────────────────────────
        LOGGER.info("[MultiRetriever] BM25 (%d, strategy=%s) ...",
                    self.bm25_top_k_articles, self.bm25_strategy)
        bm25_results = self.bm25.retrieve_candidate_pool(
            query,
            top_k_articles=self.bm25_top_k_articles,
            top_k_laws=self.bm25_top_k_laws,
            strategy=self.bm25_strategy,
        )
        bm25_ranking: dict[tuple, float] = {
            (str(a.get("law_id")), a.get("aid")): float(a.get("bm25_score", 0.0))
            for a in bm25_results
        }

        # ── Vietnamese Embedding ──────────────────────────────────────
        LOGGER.info("[MultiRetriever] Vietnamese Embedding rerank (%d→%d) ...", len(bm25_results), self.vn_top_k)
        vn_results = self.vn.retrieve(
            query,
            articles=bm25_results,  # Rerank BM25 candidates
            top_k=self.vn_top_k,
        )
        vn_ranking: dict[tuple, float] = {
            (str(a.get("law_id")), a.get("aid")): float(a.get("vn_score", a.get("retrieval_score", 0.0)))
            for a in vn_results
        }

        # ── RRF Fusion ────────────────────────────────────────────────
        LOGGER.info("[MultiRetriever] RRF Fusion ...")
        rrf_scores = reciprocal_rank_fusion([bm25_ranking, vn_ranking])

        # ── Build output ──────────────────────────────────────────────
        merged_articles: dict[tuple, dict[str, Any]] = {}
        for article_list in [bm25_results, vn_results]:
            for a in article_list:
                key = (str(a.get("law_id")), a.get("aid"))
                if key not in merged_articles:
                    merged_articles[key] = dict(a)

        result = []
        for (law_id, aid), rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            article = merged_articles.get((law_id, aid), {})
            article["law_id"] = law_id
            article["aid"] = aid
            article["rrf_score"] = round(float(rrf_score), 6)
            article["retrieval_score"] = article["rrf_score"]
            result.append(article)

        LOGGER.info("[MultiRetriever] Retrieved: %d articles (RRF top-%d)", len(result), top_k)
        return result

    # ------------------------------------------------------------------

    def _mock_retrieve(self, query: str, final_top_k: int) -> list[dict[str, Any]]:
        """Mock mode — dùng BM25 thôi."""
        return self.bm25.retrieve(query, top_k=final_top_k)