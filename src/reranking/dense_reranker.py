"""
Module : src/reranking/dense_reranker.py
Engine : Dense Retriever (BGE-M3 / SBert)  .  Stage 2/4 trong pipeline
Dung boi: scripts/run_pipeline.py

NHIEM VU
  Cham diem semantic giua query va tung article bang cosine similarity.
  Lay top-k articles co diem cao nhat. Can GPU ~2GB (BAAI/bge-m3).

INPUT
  DenseRetriever(mode, model_name)
    mode       - "mock" (test) | "local" (BGE-M3 that)
    model_name - "BAAI/bge-m3"

  .retrieve(query: str, articles: list[dict], top_k: int = 20)
    query    - noi dung vu an
    articles - pool tu BM25: [{"law_id", "aid", "content"}, ...]
    top_k    - so articles lay ra

OUTPUT
  list[dict] - top_k articles them 2 field moi:
  [
    {"law_id": "91/2015/QH13", "aid": 53354, "content": "...",
     "dense_score": 0.872, "rank": 1},
    ...
  ]
  -> Dua vao LLMReranker (Qwen2.5-3B).

CACH TINH SCORE
  query_vec   = encoder(query)         # normalize L2
  article_vec = encoder(article)       # normalize L2
  score       = dot(query_vec, article_vec)  in [-1, 1]  (= cosine vi da normalize)
"""


class DenseRetriever:
    def __init__(self, mode: str = "mock", model_name: str = "BAAI/bge-m3"):
        self.mode       = mode
        self.model_name = model_name
        self._model     = None

        if mode == "local":
            self._load_model()

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        print(f"  [DenseRetriever] Loading {self.model_name} ...")
        self._model = SentenceTransformer(self.model_name)
        print(f"  [DenseRetriever] Ready")

    def retrieve(self, query: str, articles: list[dict], top_k: int = 20) -> list[dict]:
        if not articles:
            return []

        if self.mode == "mock":
            result = []
            for i, art in enumerate(articles[:top_k]):
                entry = dict(art)
                entry["dense_score"] = round(1.0 - i * 0.01, 3)
                entry["rank"] = i + 1
                result.append(entry)
            return result

        import numpy as np

        texts = [art["content"] for art in articles]

        query_emb = self._model.encode(
            f"Represent this query for searching legal articles: {query}",
            normalize_embeddings=True,
        )
        article_embs = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        scores = np.dot(article_embs, query_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        result = []
        for rank, idx in enumerate(top_indices):
            entry = dict(articles[idx])
            entry["dense_score"] = float(scores[idx])
            entry["rank"] = rank + 1
            result.append(entry)
        return result
