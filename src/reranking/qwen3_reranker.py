"""Qwen3-Reranker — LLM-based relevance scorer dùng FlagLLMReranker.

Model: Qwen/Qwen3-Reranker-0.6B (nhẹ nhất, phù hợp T1200 4GB VRAM)

Khác với LLMReranker hiện tại (Qwen2.5-3B-Instruct):
  - LLMReranker: gọi generate() từng article → CO/KHONG (chậm, O(N) calls)
  - Qwen3Reranker: dùng FlagLLMReranker → score tất cả cùng lúc (nhanh hơn)
  - Output: relevance score liên tục [0, 1] thay vì binary CO/KHONG
  - Tốt hơn cho ensemble vì có gradient (không bị mất thông tin)
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

from src.utils.io import unique_by_key

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"


class Qwen3Reranker:
    """Relevance scorer dùng Qwen3-Reranker qua FlagEmbedding.

    Parameters
    ----------
    mode : "mock" | "local"
    model_name : str
        "Qwen/Qwen3-Reranker-0.6B" (default, nhẹ)
        "Qwen/Qwen3-Reranker-1.7B"  (tốt hơn, cần ~3.4 GB VRAM)
    batch_size : int
    max_length : int
        Max token length per pair. Qwen3-Reranker hỗ trợ 32K.
        Giới hạn 2048 để tiết kiệm VRAM trên T1200.
    model_cache_dir : Path | None
    device : str | None
    """

    def __init__(
        self,
        mode: str = "local",
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 8,       # Nhỏ hơn vì LLM nặng hơn CrossEncoder
        max_length: int = 2048,
        model_cache_dir: str | Path | None = None,
        device: str | None = None,
    ):
        self.mode       = mode
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device     = device

        if model_cache_dir:
            self.model_cache_dir = Path(model_cache_dir)
        else:
            _here = Path(__file__).resolve()
            self.model_cache_dir = next(
                (p / "models" for p in _here.parents if (p / "models").is_dir()),
                None,
            )

        self._model = None
        if mode == "local":
            self._load_model()

    def _load_model(self) -> None:
        try:
            from FlagEmbedding import FlagLLMReranker
        except ImportError as exc:
            raise ImportError(
                "Qwen3Reranker yêu cầu FlagEmbedding.\n"
                "Cài: pip install FlagEmbedding"
            ) from exc

        try:
            import torch
            use_fp16      = torch.cuda.is_available()
            chosen_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            use_fp16      = False
            chosen_device = "cpu"

        cache = str(self.model_cache_dir) if self.model_cache_dir else None
        LOGGER.info("[Qwen3Reranker] Loading %s on %s ...", self.model_name, chosen_device)

        self._model = FlagLLMReranker(
            self.model_name,
            use_fp16=use_fp16,
            devices=[chosen_device],
            cache_dir=cache,
            trust_remote_code=True,
            batch_size=self.batch_size,
            max_length=self.max_length,
            normalize=True,       # Output [0, 1] thay vì raw logits
        )
        LOGGER.info("[Qwen3Reranker] Ready")

    def score(
        self,
        query: str,
        articles: list[dict[str, Any]],
    ) -> np.ndarray:
        """Score tất cả articles, trả về array scores shape (N,) trong [0, 1]."""
        if not articles:
            return np.array([], dtype=np.float32)

        if self.mode == "mock":
            # Mock: dùng rrf_score nếu có, fallback về retrieval_score
            scores = np.array(
                [float(a.get("rrf_score", a.get("retrieval_score", 0.0))) for a in articles],
                dtype=np.float32,
            )
            mx = scores.max()
            return scores / mx if mx > 0 else scores

        if self._model is None:
            self._load_model()

        # FlagLLMReranker nhận list of [query, passage] pairs
        pairs = [
            [query[:1000], str(a.get("content", ""))[:1000]]
            for a in articles
        ]

        LOGGER.info("[Qwen3Reranker] Scoring %d pairs ...", len(pairs))
        raw = self._model.compute_score(pairs)
        scores = np.asarray(raw, dtype=np.float32)

        # normalize=True đã set → output trong [0, 1]
        LOGGER.info("[Qwen3Reranker] Done. Score range [%.3f, %.3f]", scores.min(), scores.max())
        return scores

    def rerank(
        self,
        query: str,
        articles: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank articles, thêm field qwen3_score."""
        candidates = unique_by_key(articles, ("law_id", "aid"))
        if not candidates:
            return []

        scores = self.score(query, candidates)

        result = []
        for art, score in zip(candidates, scores):
            entry = dict(art)
            entry["qwen3_score"] = float(score)
            result.append(entry)

        result.sort(key=lambda x: x["qwen3_score"], reverse=True)

        if top_k is not None:
            result = result[:top_k]

        for rank, entry in enumerate(result, 1):
            entry["rank"] = rank

        return result