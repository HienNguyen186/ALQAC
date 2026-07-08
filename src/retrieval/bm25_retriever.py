"""BM25 law retriever for the ALQAC 2026 legal outcome pipeline."""

from __future__ import annotations

# Auto-add project root when this file is executed directly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.query_expansion import expand_query
from src.utils.io import (
    load_json_file,
    unique_by_key,
    validate_corpus_records,
)
from src.utils.text import tokenize


# ======================================================================
# Data Model
# ======================================================================

@dataclass(frozen=True)
class LawArticle:
    """Internal representation of a legal article."""

    law_id: str
    aid: int | str
    content: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "law_id": self.law_id,
            "aid": self.aid,
            "content": self.content,
        }


# ======================================================================
# BM25 Retriever
# ======================================================================

class LawRetriever:
    """
    Sparse retriever using BM25.

    Pipeline
    --------
    Query
      → Query Expansion
      → Tokenization
      → BM25
      → Candidate Pool
    """

    def __init__(self, corpus_path: str | Path):
        corpus = load_json_file(corpus_path, "Vietnamese legal corpus")
        self.corpus = validate_corpus_records(corpus)

        self._articles: list[LawArticle] = []
        self._law_to_indices: dict[str, list[int]] = {}
        self._tokenized_articles: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

        # BM25 score cache (keyed by query string)
        self._last_query: str | None = None
        self._last_scores: np.ndarray | None = None

        self._build_index()

    # ------------------------------------------------------------------
    # Build BM25 index
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        seen: set[tuple[str, str, str]] = set()

        for law in self.corpus:
            law_id = str(law["law_id"])
            indices: list[int] = []

            for article in law.get("content", []):
                aid = article.get("aid")
                content = str(
                    article.get("content_Article")
                    or article.get("content")
                    or ""
                ).strip()

                if not content:
                    continue

                key = (law_id, str(aid), content)
                if key in seen:
                    continue
                seen.add(key)

                indices.append(len(self._articles))
                self._articles.append(LawArticle(law_id=law_id, aid=aid, content=content))

            if indices:
                self._law_to_indices[law_id] = indices

        if not self._articles:
            raise ValueError("No valid legal articles found in corpus.")

        print(
            f"[BM25] Indexed {len(self._articles):,} legal articles "
            f"from {len(self._law_to_indices):,} legal documents."
        )

        self._tokenized_articles = [tokenize(a.content) for a in self._articles]
        self.bm25 = BM25Okapi(self._tokenized_articles)

    # ------------------------------------------------------------------
    # BM25 score (cached)
    # ------------------------------------------------------------------

    def _score(self, query: str, *, expand: bool = True, use_cache: bool = True) -> np.ndarray:
        if use_cache and self._last_query == query and self._last_scores is not None:
            return self._last_scores

        expanded = expand_query(query) if expand else query
        tokens = tokenize(expanded)
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)

        self._last_query = query
        self._last_scores = scores
        return scores

    # ------------------------------------------------------------------
    # Fast top-k indices
    # ------------------------------------------------------------------

    @staticmethod
    def _top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
        if top_k <= 0:
            return np.asarray([], dtype=np.int64)
        k = min(top_k, len(scores))
        if k == len(scores):
            return np.argsort(scores)[::-1]
        idx = np.argpartition(scores, -k)[-k:]
        return idx[np.argsort(scores[idx])[::-1]]

    # ------------------------------------------------------------------
    # Retrieve top articles
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Return top-k legal articles ranked by BM25 score."""
        if top_k <= 0:
            return []

        scores = self._score(query)
        order = self._top_indices(scores, top_k)

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(order, start=1):
            article = self._articles[int(idx)].as_dict()
            article["bm25_score"] = float(scores[idx])
            article["bm25_rank"] = rank
            article["rank"] = rank
            results.append(article)

        return unique_by_key(results, ("law_id", "aid"))

    # ------------------------------------------------------------------
    # Retrieve top laws
    # ------------------------------------------------------------------

    def retrieve_laws(self, query: str, top_k: int = 5) -> list[str]:
        """
        Return top-k law_ids ranked by max-BM25 score across their articles.
        """
        if top_k <= 0:
            return []

        scores = self._score(query)
        law_scores: dict[str, float] = {}
        for law_id, indices in self._law_to_indices.items():
            if indices:
                law_scores[law_id] = float(np.max(scores[indices]))

        ranked = sorted(law_scores.items(), key=lambda x: x[1], reverse=True)
        return [law_id for law_id, _ in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Get all articles belonging to a set of laws
    # ------------------------------------------------------------------

    def get_articles_by_laws(self, law_ids: list[str]) -> list[dict[str, Any]]:
        selected = set(law_ids)
        articles = [
            a.as_dict()
            for a in self._articles
            if a.law_id in selected
        ]
        return unique_by_key(articles, ("law_id", "aid"))

    # ------------------------------------------------------------------
    # Candidate pool (hybrid strategy)
    # ------------------------------------------------------------------

    def retrieve_candidate_pool(
        self,
        query: str,
        top_k_articles: int = 100,
        top_k_laws: int = 5,
        strategy: str = "hybrid",
        max_candidates: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build a ranked candidate pool.

        strategy
        --------
        article : BM25 top-k articles only.
        law     : All articles from the top-k laws (by max BM25 score).
        hybrid  : Union of the above two strategies; law-branch articles
                  receive their real BM25 scores (not 0) so they compete
                  fairly in downstream re-ranking.
        """
        if strategy not in {"article", "law", "hybrid"}:
            raise ValueError("strategy must be 'article', 'law', or 'hybrid'")

        # Score the query ONCE; _score() caches the result.
        scores = self._score(query)

        merged: dict[tuple[str, str], dict[str, Any]] = {}

        # -------------------------------------------------------------- 
        # Strategy: article — BM25 top-k
        # -------------------------------------------------------------- 
        if strategy in {"article", "hybrid"}:
            article_hits = self.retrieve(query, top_k=top_k_articles)
            for item in article_hits:
                key = (str(item["law_id"]), str(item["aid"]))
                merged[key] = dict(item)

        # -------------------------------------------------------------- 
        # Strategy: law — all articles of top-k laws, scored by real BM25
        # -------------------------------------------------------------- 
        if strategy in {"law", "hybrid"}:
            top_laws = self.retrieve_laws(query, top_k=top_k_laws)
            law_articles = self.get_articles_by_laws(top_laws)

            for article in law_articles:
                key = (str(article["law_id"]), str(article["aid"]))
                if key in merged:
                    # Already present from article-branch — skip.
                    continue

                # Look up the real BM25 score from the cached scores array.
                law_id_str = str(article["law_id"])
                aid_val = article["aid"]
                article_idx: int | None = None
                for idx, art_obj in enumerate(self._articles):
                    if art_obj.law_id == law_id_str and art_obj.aid == aid_val:
                        article_idx = idx
                        break

                real_bm25 = float(scores[article_idx]) if article_idx is not None else 0.0
                # Assign a rank beyond the article-branch window so article-branch
                # articles are preferred when scores tie, but law-branch articles
                # still compete on their actual BM25 score.
                law_rank = top_k_articles + len(merged) + 1

                merged[key] = {
                    **article,
                    "bm25_score": real_bm25,
                    "bm25_rank": law_rank,
                }

        # Sort by real BM25 score (descending), break ties by rank (ascending).
        candidates = sorted(
            merged.values(),
            key=lambda x: (x["bm25_score"], -x["bm25_rank"]),
            reverse=True,
        )

        if max_candidates is not None:
            candidates = candidates[:max_candidates]

        return candidates

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def num_articles(self) -> int:
        return len(self._articles)

    @property
    def num_laws(self) -> int:
        return len(self._law_to_indices)

    def __len__(self) -> int:
        return len(self._articles)
