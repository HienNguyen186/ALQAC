"""Vietnamese dense retriever dùng dangvantuan/vietnamese-embedding (PhoBERT-based).

Đây là tầng thứ 3 trong Multi-Retrieval:
  BM25 → BGE-M3 → VietnameseRetriever → Chunked → Union

Lý do dùng riêng model này:
  - PhoBERT được pre-train trên văn bản pháp lý Việt Nam
  - Bắt được thuật ngữ chuyên ngành tiếng Việt tốt hơn BGE-M3
  - Embedding dim 768, nhỏ hơn BGE-M3 (1024) → nhanh hơn
  - Bổ sung recall cho những article BGE-M3 bỏ sót
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import find_project_root, unique_by_key

LOGGER = logging.getLogger(__name__)

# Model mặc định — PhoBERT fine-tuned cho Vietnamese semantic similarity
DEFAULT_MODEL = "dangvantuan/vietnamese-embedding"


class VietnameseRetriever:
    """Dense retriever dùng Vietnamese-specific embedding model.

    Parameters
    ----------
    mode : "mock" | "local"
    model_name : str
        HuggingFace model ID. Mặc định dangvantuan/vietnamese-embedding.
        Thay thế khác: "hiieu/halong_embedding"
    batch_size : int
    cache_dir : Path | None
        Thư mục lưu embedding cache.
    model_cache_dir : Path | None
        Thư mục chứa HuggingFace model đã tải.
    device : str | None
    """

    def __init__(
        self,
        mode: str = "local",
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 16,  # Giảm từ 64 → 16 để tránh scatter gather OOM
        cache_dir: str | Path | None = None,
        model_cache_dir: str | Path | None = None,
        device: str | None = None,
    ):
        self.mode       = mode
        self.model_name = model_name
        self.batch_size = 16  # ← Giảm từ 64 → 16 (GPU optimization)
        self.cache_dir  = (
            Path(cache_dir) if cache_dir
            else find_project_root() / "outputs" / "cache" / "vn_embeddings"
        )
        if model_cache_dir:
            self.model_cache_dir = Path(model_cache_dir)
        else:
            _here = Path(__file__).resolve()
            self.model_cache_dir = next(
                (p / "models" for p in _here.parents if (p / "models").is_dir()),
                None,
            )
        self.device  = device
        self._model  = None

        if mode == "local":
            self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "VietnameseRetriever yêu cầu sentence-transformers.\n"
                "Cài: pip install sentence-transformers"
            ) from exc

        try:
            import torch
            chosen_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            chosen_device = "cpu"

        cache = str(self.model_cache_dir) if self.model_cache_dir else None
        LOGGER.info("[VietnameseRetriever] Loading %s on %s ...", self.model_name, chosen_device)

        self._model = SentenceTransformer(
            self.model_name,
            device=chosen_device,
            cache_folder=cache,
            trust_remote_code=True,
        )

        self._model.max_seq_length = 256
        
        LOGGER.info(
            "Tokenizer vocab size: %d",
            self._model.tokenizer.vocab_size,
        )
        
        LOGGER.info(
            "Model vocab size: %d",
            self._model[0].auto_model.config.vocab_size,
        )

        LOGGER.info("[VietnameseRetriever] Ready (dim=%d)", self._model.get_sentence_embedding_dimension())

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def _cache_path(self, suffix: str) -> Path:
        safe = self.model_name.replace("/", "__")
        return self.cache_dir / f"{safe}_{suffix}.pkl"

    def _pool_sig(self, articles: list[dict[str, Any]]) -> str:
        parts = [f"{a.get('law_id')}:{a.get('aid')}:{self._sha(str(a.get('content', '')))}"
                 for a in articles]
        return hashlib.sha1("|".join(parts).encode()).hexdigest()

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

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_articles(self, articles: list[dict[str, Any]]) -> np.ndarray:
        """Encode articles → shape (N, D), với cache."""
        sig        = self._pool_sig(articles)
        cache_path = self._cache_path(f"articles_{sig}")
        cached     = self._load_cache(cache_path)
        if cached is not None:
            LOGGER.debug("[VietnameseRetriever] Article cache hit (%d)", len(articles))
            return cached

        assert self._model is not None
        # ← Truncate to 256 tokens to reduce VRAM (from full content)
        texts = []

        for idx, article in enumerate(articles):
            text = article.get("content")
        
            if text is None:
                LOGGER.warning("Article %d has None content", idx)
                text = ""
        
            elif not isinstance(text, str):
                LOGGER.warning(
                    "Article %d has invalid type: %s",
                    idx,
                    type(text),
                )
                text = str(text)
        
            texts.append(text)
            
        LOGGER.info("[VietnameseRetriever] Encoding %d articles ...", len(texts))
        
        try:
            embs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            LOGGER.exception("[VietnameseRetriever] Encode failed")
            raise
        
        result = np.asarray(embs, dtype=np.float32)
        
        self._save_cache(cache_path, result)
        LOGGER.info("[VietnameseRetriever] Encoded and cached.")
        return result

    def _encode_query(self, query: str) -> np.ndarray:
        assert self._model is not None
        emb = self._model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb[0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        articles: list[dict[str, Any]],
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        """Rank articles by Vietnamese embedding cosine similarity.

        Returns list of articles với thêm field:
          vn_score     — cosine similarity
          retrieval_score — alias cho downstream compatibility
          rank         — rank (1-based)
        """
        candidates = unique_by_key(articles, ("law_id", "aid"))
        if not candidates or top_k <= 0:
            return []

        # Mock mode: random-ish score dựa trên len overlap
        if self.mode == "mock":
            from src.utils.text import tokenize
            q_terms = set(tokenize(query))
            scored  = []
            for idx, art in enumerate(candidates):
                a_terms = set(tokenize(art.get("content", "")))
                score   = len(q_terms & a_terms) / max(len(q_terms), 1)
                scored.append((score, -idx, art))
            ranked = sorted(scored, reverse=True)[:top_k]
            result = []
            for rank, (score, _, art) in enumerate(ranked, 1):
                entry = dict(art)
                entry["vn_score"]        = round(float(score), 6)
                entry["retrieval_score"] = round(float(score), 6)
                entry["rank"]            = rank
                result.append(entry)
            return result

        if self._model is None:
            self._load_model()

        art_embs   = self._encode_articles(candidates)
        query_emb  = self._encode_query(query)
        scores     = np.dot(art_embs, query_emb).astype(np.float32)
        top_idx    = np.argsort(scores)[::-1][:top_k]

        result = []
        for rank, idx in enumerate(top_idx, 1):
            entry = dict(candidates[int(idx)])
            entry["vn_score"]        = float(scores[idx])
            entry["retrieval_score"] = float(scores[idx])
            entry["rank"]            = rank
            result.append(entry)
        return result