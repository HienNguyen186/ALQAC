"""Cross-Encoder Reranker — monoT5-style relevance scoring.

Dùng sentence-transformers CrossEncoder để score từng cặp (query, article).
Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual, ~90 MB)

Khác với bi-encoder (BGE-M3, VN Embedding):
  - Bi-encoder: encode query và article riêng → dot product (nhanh nhưng kém chính xác)
  - Cross-encoder: nhận (query + article) cùng lúc → score trực tiếp (chậm nhưng chính xác hơn)

Dùng sau Multi-Retrieval để score top-500 candidates → chuẩn bị cho ensemble.
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

DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CrossEncoderReranker:
    """Score (query, article) pairs bằng cross-encoder.

    Parameters
    ----------
    mode : "mock" | "local"
    model_name : str
        HuggingFace cross-encoder model.
    batch_size : int
        Số cặp encode mỗi batch.
    max_length : int
        Max token length cho mỗi cặp (query + article).
    model_cache_dir : Path | None
    device : str | None
    """

    def __init__(
        self,
        mode: str = "local",
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        max_length: int = 512,
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
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker yêu cầu sentence-transformers.\n"
                "Cài: pip install sentence-transformers"
            ) from exc

        try:
            import torch
            chosen_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            chosen_device = "cpu"

        cache = str(self.model_cache_dir) if self.model_cache_dir else None
        LOGGER.info("[CrossEncoder] Loading %s on %s ...", self.model_name, chosen_device)

        self._model = CrossEncoder(
            self.model_name,
            device=chosen_device,
            cache_folder=cache,
            max_length=self.max_length,
            trust_remote_code=True,
        )
        LOGGER.info("[CrossEncoder] Ready")

    def score(
        self,
        query: str,
        articles: list[dict[str, Any]],
    ) -> np.ndarray:
        """Score tất cả articles, trả về array scores shape (N,).

        Không giới hạn top_k ở đây — để EnsembleReranker quyết định.
        """
        if not articles:
            return np.array([], dtype=np.float32)

        if self.mode == "mock":
            # Mock: score = bm25_score chuẩn hóa
            scores = np.array(
                [float(a.get("bm25_score", 0.0)) for a in articles],
                dtype=np.float32,
            )
            mx = scores.max()
            return scores / mx if mx > 0 else scores

        if self._model is None:
            self._load_model()

        # Tạo pairs (query, article_content)
        pairs = [
            (query[:500], str(a.get("content", ""))[:800])
            for a in articles
        ]

        LOGGER.info("[CrossEncoder] Scoring %d pairs ...", len(pairs))
        raw_scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores, dtype=np.float32)

        # Nếu output có 2 chiều (softmax): lấy cột positive (index 1)
        if scores.ndim == 2:
            scores = scores[:, 1]

        LOGGER.info("[CrossEncoder] Done. Score range [%.3f, %.3f]", scores.min(), scores.max())
        return scores

    def rerank(
        self,
        query: str,
        articles: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank articles, thêm field cross_score. Trả về top_k nếu chỉ định."""
        candidates = unique_by_key(articles, ("law_id", "aid"))
        if not candidates:
            return []

        scores = self.score(query, candidates)

        result = []
        for idx, (art, score) in enumerate(zip(candidates, scores)):
            entry = dict(art)
            entry["cross_score"] = float(score)
            result.append(entry)

        result.sort(key=lambda x: x["cross_score"], reverse=True)

        if top_k is not None:
            result = result[:top_k]

        for rank, entry in enumerate(result, 1):
            entry["rank"] = rank

        return result