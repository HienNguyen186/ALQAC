"""Chunked BM25 Retriever — thay thế PageIndex (cloud API) bằng local sliding window.

Mục tiêu:
  - Điều luật dài (7.5% articles > 512 token) bị truncate khi encode dense
  - Giải pháp: chia mỗi article dài thành nhiều chunk nhỏ (256 token overlap 64)
  - Mỗi chunk được BM25 score riêng → lấy max score của article
  - Tăng recall cho article dài mà BM25 bỏ sót vì từ khóa nằm ở cuối

Pipeline:
  corpus articles → sliding window chunks → BM25 index
  query → BM25 score mỗi chunk → max-pool về article level → top-k articles
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
from rank_bm25 import BM25Okapi

from src.utils.io import unique_by_key, validate_corpus_records, load_json_file
from src.utils.text import tokenize

LOGGER = logging.getLogger(__name__)


class ChunkedRetriever:
    """BM25 trên sliding-window chunks của điều luật.

    Parameters
    ----------
    corpus_path : str | Path
        Đường dẫn file corpus JSON.
    chunk_size : int
        Số token mỗi chunk (mặc định 200 — phù hợp với avg 218 token/article).
    chunk_overlap : int
        Số token overlap giữa 2 chunk liên tiếp.
    min_chunk_tokens : int
        Bỏ qua chunk quá ngắn.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        chunk_size:    int = 200,
        chunk_overlap: int = 50,
        min_chunk_tokens: int = 20,
    ):
        self.chunk_size       = chunk_size
        self.chunk_overlap    = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens

        corpus = load_json_file(corpus_path, "Vietnamese legal corpus (chunked)")
        corpus = validate_corpus_records(corpus)

        # Mỗi chunk lưu: (law_id, aid, chunk_idx, tokens)
        self._chunks:   list[tuple[str, Any, int, list[str]]] = []
        # Map (law_id, aid) → list of chunk indices
        self._art_to_chunks: dict[tuple[str, Any], list[int]] = {}
        # Original article content
        self._articles: dict[tuple[str, Any], str]            = {}

        self._build_index(corpus)

    # ------------------------------------------------------------------
    # Sliding window chunker
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_tokens(tokens: list[str], size: int, overlap: int) -> list[list[str]]:
        """Chia list tokens thành các chunk với overlap."""
        if len(tokens) <= size:
            return [tokens]
        chunks = []
        step   = size - overlap
        start  = 0
        while start < len(tokens):
            chunks.append(tokens[start:start + size])
            start += step
        return chunks

    def _build_index(self, corpus: list[dict[str, Any]]) -> None:
        seen_arts: set[tuple[str, Any]] = set()

        for law in corpus:
            law_id = str(law["law_id"])
            for art in law.get("content", []):
                aid     = art.get("aid")
                content = str(
                    art.get("content_Article") or art.get("content") or ""
                ).strip()
                if not content:
                    continue

                key = (law_id, aid)
                if key in seen_arts:
                    continue
                seen_arts.add(key)
                self._articles[key] = content

                tokens = tokenize(content)
                if not tokens:
                    continue

                chunks = self._chunk_tokens(tokens, self.chunk_size, self.chunk_overlap)
                chunk_indices: list[int] = []
                for chunk_tokens in chunks:
                    if len(chunk_tokens) < self.min_chunk_tokens:
                        continue
                    idx = len(self._chunks)
                    self._chunks.append((law_id, aid, idx, chunk_tokens))
                    chunk_indices.append(idx)

                if chunk_indices:
                    self._art_to_chunks[key] = chunk_indices

        if not self._chunks:
            raise ValueError("ChunkedRetriever: không tìm thấy chunk nào trong corpus.")

        tokenized_chunks = [c[3] for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_chunks)

        total_arts   = len(self._art_to_chunks)
        total_chunks = len(self._chunks)
        avg_chunks   = total_chunks / max(total_arts, 1)
        LOGGER.info(
            "[ChunkedRetriever] Indexed %d articles → %d chunks (avg %.1f chunks/article)",
            total_arts, total_chunks, avg_chunks,
        )
        print(
            f"[ChunkedRetriever] {total_arts:,} articles → {total_chunks:,} chunks "
            f"(avg {avg_chunks:.1f}/article)"
        )

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k articles bằng chunk-level BM25 + max-pool.

        Returns list với fields:
          law_id, aid, content, chunked_bm25_score, rank
        """
        q_tokens   = tokenize(query)
        chunk_scores = np.asarray(self._bm25.get_scores(q_tokens), dtype=np.float32)

        # Max-pool chunk scores → article score
        art_scores: dict[tuple[str, Any], float] = {}
        for key, chunk_idxs in self._art_to_chunks.items():
            if chunk_idxs:
                art_scores[key] = float(np.max(chunk_scores[chunk_idxs]))

        if not art_scores:
            return []

        # Sắp xếp và lấy top-k
        ranked_arts = sorted(art_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result = []
        for rank, ((law_id, aid), score) in enumerate(ranked_arts, 1):
            result.append({
                "law_id":              law_id,
                "aid":                 aid,
                "content":             self._articles.get((law_id, aid), ""),
                "chunked_bm25_score":  score,
                "retrieval_score":     score,
                "rank":                rank,
            })
        return result
