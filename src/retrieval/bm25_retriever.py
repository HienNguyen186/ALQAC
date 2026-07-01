"""
Module : src/retrieval/bm25_retriever.py
Engine : BM25 Law Retriever  .  Stage 1/4 trong pipeline
Dung boi: scripts/run_pipeline.py, scripts/evaluate.py

NHIEM VU
  Tim top-k bo luat lien quan bang keyword matching (BM25 Okapi).
  Khong can GPU - chay tren CPU < 1 giay.

INPUT
  LawRetriever(corpus_path: str)
    corpus_path - data/raw/corpus_law_pub.json  (18 luat, 3352 articles)

  .retrieve_laws(query: str, top_k: int = 3) -> list[str]
    query  - noi dung vu an (case_query)
    top_k  - so bo luat muon lay

  .get_articles_by_laws(law_ids: list[str]) -> list[dict]
    law_ids - VD: ["91/2015/QH13", "45/2013/QH13"]

OUTPUT
  retrieve_laws()        -> ["91/2015/QH13", "45/2013/QH13", "52/2014/QH13"]

  get_articles_by_laws() -> [
    {"law_id": "91/2015/QH13", "aid": 52771, "content": "Dieu 1. Pham vi..."},
    ...  # toan bo articles cua 3 luat (~vai tram entries)
  ]
  -> Pool nay duoc dua thang vao DenseRetriever (BGE-M3).

CACH TINH SCORE
  Score(luat) = MAX{ BM25_score(article) : article trong luat do }
"""

import json
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class LawRetriever:
    def __init__(self, corpus_path: str):
        with open(corpus_path, encoding="utf-8") as f:
            self.corpus = json.load(f)

        self._articles: list[tuple[str, str, str]] = []
        self._law_to_indices: dict[str, list[int]] = {}

        for law in self.corpus:
            law_id  = law["law_id"]
            indices = []
            for art in law["content"]:
                idx = len(self._articles)
                self._articles.append((law_id, art["aid"], art["content_Article"]))
                indices.append(idx)
            self._law_to_indices[law_id] = indices

        tokenized = [tokenize(text) for _, _, text in self._articles]
        self.bm25  = BM25Okapi(tokenized)

    def retrieve_laws(self, query: str, top_k: int = 3) -> list[str]:
        """Top-k law_id lien quan nhat. Score = MAX bm25 score cua articles trong luat."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        law_scores: dict[str, float] = {}
        for law_id, indices in self._law_to_indices.items():
            law_scores[law_id] = max(scores[i] for i in indices)

        ranked = sorted(law_scores.items(), key=lambda x: x[1], reverse=True)
        return [law_id for law_id, _ in ranked[:top_k]]

    def get_articles_by_laws(self, law_ids: list[str]) -> list[dict]:
        """Toan bo articles thuoc law_ids -> pool cho BGE-M3."""
        law_set = set(law_ids)
        return [
            {"law_id": law_id, "aid": aid, "content": content}
            for law_id, aid, content in self._articles
            if law_id in law_set
        ]

    def retrieve(self, query: str, top_k_laws: int = 3) -> list[dict]:
        """Shortcut: retrieve_laws -> get_articles_by_laws trong 1 call."""
        return self.get_articles_by_laws(self.retrieve_laws(query, top_k=top_k_laws))
