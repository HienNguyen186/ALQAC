# ALQAC 2026 Legal Outcome Prediction

This repository implements an end-to-end Legal AI pipeline for predicting Vietnamese civil-case outcomes with supporting legal evidence.

Pipeline:

```text
case_query
  -> BM25 law retriever
  -> Dense retriever/reranker (BGE-M3)
  -> LLM relevance reranker (Qwen2.5-3B)
  -> LLM outcome predictor (Qwen3)
  -> prediction + legal evidence
```

The architecture is unchanged from the original design, but the code now validates inputs, resolves paths from the project root, supports deterministic mock runs, caches dense embeddings, and handles missing files with friendly messages.

## Environment Creation

Python 3.10+ is recommended.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux or Google Colab:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In Colab, install a CUDA-compatible PyTorch wheel first if needed, then run `pip install -r requirements.txt`.

## Dataset Preparation

Place the public test and law corpus here:

```text
data/raw/ALQAC2026_public_test.json
data/raw/corpus_law_pub.json
```

You can also pass explicit paths:

```bash
python scripts/run_pipeline.py --test /path/to/ALQAC2026_public_test.json --corpus /path/to/corpus_law_pub.json
```

If a required file is missing, the scripts print the expected location and the CLI argument to use instead of crashing with `FileNotFoundError`.

## Model Downloading

Mock mode does not download models and is intended for smoke tests, CI, and CPU-only debugging.

Local mode uses Hugging Face models:

```text
BAAI/bge-m3
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen3-8B
```

Download is automatic through `sentence-transformers` and `transformers` the first time local mode runs. Make sure you have enough disk space and a Hugging Face token configured if a selected model requires authentication.

## Running Mock Mode

This is the required smoke-test command and should run without GPU:

```bash
python scripts/run_pipeline.py --rerank-mode mock --llm-mode mock
```

For a quick check:

```bash
python scripts/run_pipeline.py --rerank-mode mock --llm-mode mock --limit 5
```

## Running Local Mode

Local mode enables BGE-M3 dense retrieval and Qwen reranking/prediction:

```bash
python scripts/run_pipeline.py --rerank-mode local --llm-mode local
```

Dense article embeddings are cached under:

```text
outputs/cache/embeddings/
```

The dense retriever automatically uses CUDA when available and falls back to CPU. The LLM stages are designed for GPU inference; on Windows, `bitsandbytes` is skipped by default in `requirements.txt`, so use mock mode or a Linux/Colab GPU environment for 4-bit local LLM inference.

## Evaluation

Evaluate BM25 evidence retrieval:

```bash
python scripts/evaluate.py --top-k 20
```

Evaluate a generated submission against public labels:

```bash
python scripts/evaluate.py --submission outputs/submissions/submission_mock_mock_YYYYMMDD_HHMMSS.json
```

Metrics reported include evidence precision, recall@k, evidence F1, coverage, accuracy, and macro F1.

## Expected Outputs

Pipeline outputs are written to:

```text
outputs/submissions/submission_<rerank-mode>_<llm-mode>_<timestamp>.json
```

Each item contains:

```json
{
  "case_id": "case_4101",
  "prediction": "PARTIAL_A_WIN",
  "confidence": 0.62,
  "law_evidence": [
    {"law_id": "91/2015/QH13", "aid": 53354}
  ]
}
```

## Engineering Notes

- `pathlib` is used for cross-platform path handling.
- Scripts detect the project root and validate data before execution.
- Mock prediction is deterministic for reproducible tests.
- BM25 retrieval normalizes Vietnamese text, removes duplicate articles, and supports article-level evaluation.
- Dense retrieval batches embeddings, supports GPU/CPU, and caches article vectors.
- LLM prompts require constrained output formats to reduce hallucination.

## Regenerating Requirements

```bash
python scripts/generate_requirements.py
```
