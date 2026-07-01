**Logic flow toàn bộ:**

```
scripts/run_pipeline.py          ← ĐIỂM VÀO DUY NHẤT
│
├─ src/retrieval/bm25_retriever.py
│    LawRetriever.retrieve_laws()     → BM25 score 18 luật → top-3 law_id
│    LawRetriever.get_articles_by_laws() → lấy ~vài trăm articles của 3 luật
│
├─ src/reranking/dense_reranker.py
│    DenseRetriever.retrieve()        → BGE-M3 cosine score → top-20 articles
│
├─ src/reranking/llm_reranker.py
│    LLMReranker.rerank()             → Qwen2.5-3B hỏi yes/no từng article → 2~5
│
├─ src/prediction/llm_predictor.py
│    LLMPredictor.predict()           → Qwen3-8B CoT → A_WIN/B_WIN/...
│
└─ outputs/submissions/submission_*.json   ← KẾT QUẢ

scripts/evaluate.py              ← Đánh giá law retrieval trên public test
│    Dùng: bm25_retriever + law_name_map
│    Output: Precision / Recall / F1

src/retrieval/law_name_map.py   ← Bảng tra cứu (không chạy độc lập)
│    "Bộ luật Dân sự 2015" → "91/2015/QH13"
│    Điều 584 + offset → aid thật trong corpus

src/api/case_api.py             ← Chưa dùng (Step ⑤ sau)
│    Gọi API lấy nội dung vụ án cho private test
```

**Luồng data qua từng bước:**

```
case_query (text)
  → [BM25]          3 law_id
  → [get_articles]  ~300 articles {law_id, aid, content}
  → [BGE-M3]        20 articles + dense_score
  → [Qwen2.5-3B]    2~5 articles + llm_relevant=True
  → [Qwen3-8B]      label + reasoning text
  → submission.json {case_id, prediction, law_evidence}
```