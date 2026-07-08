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
        """
        Load BGE-M3 completely offline from local HuggingFace cache.
        """

        from src.utils.model_cache import (
            configure_hf_cache,
            get_model_path,
        )

        # Configure HF cache before importing FlagEmbedding
        configure_hf_cache()

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "DenseRetriever local mode yêu cầu FlagEmbedding.\n"
                "Cài đặt: pip install FlagEmbedding\n"
                "Hoặc chạy với --rerank-mode mock để bỏ qua."
            ) from exc

        try:
            import torch
            use_fp16 = torch.cuda.is_available()
            chosen_device = self.device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        except ImportError:
            use_fp16 = False
            chosen_device = "cpu"

        model_path = get_model_path(self.model_name)

        LOGGER.info(
            "[DenseRetriever] Loading local model:\n%s",
            model_path,
        )

        LOGGER.info(
            "[DenseRetriever] Device=%s fp16=%s",
            chosen_device,
            use_fp16,
        )

        self._model = BGEM3FlagModel(
            model_path,
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
            f"{a.get('law_id')}:{a.get('aid')}:{self._content_hash(str(a.get('content', '')))}"
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
