# ALQAC 2026 — Quick Start Guide

## Cách sử dụng

### 1️⃣ Lần đầu tiên — Tải models (khi có internet)

**Windows PowerShell:**
```powershell
cd path/to/ALQAC
python scripts/download_models.py
```

**Linux/Mac:**
```bash
cd path/to/ALQAC
python scripts/download_models.py
```

Models sẽ lưu vào thư mục `models/` (tổng ~25 GB).

---

### 2️⃣ Chạy pipeline (offline, không cần download lại)

**Windows — Click file batch:**
```
run_local.bat
```

**PowerShell / Linux / Mac:**
```bash
python scripts/run_pipeline.py --rerank-mode local --llm-mode mock
```

---

## Chế độ chạy

| Chế độ | BGE-M3 | LLM | Tốc độ | RAM | Dùng khi |
|---|---|---|---|---|---|
| `mock mock` | ❌ | ❌ | ⚡⚡⚡ | 1 GB | Test nhanh, CPU-only |
| `local mock` | ✅ | ❌ | ⚡⚡ | 2 GB | Retrieve tốt, không cần predict |
| `local local` | ✅ | ✅ | ⚡ | 4+ GB | Đầy đủ, cần GPU mạnh |

### Mock mode (test nhanh, không model):
```bash
python scripts/run_pipeline.py --rerank-mode mock --llm-mode mock
```

### Local BGE-M3 + Mock LLM (khuyến nghị T1200):
```bash
python scripts/run_pipeline.py --rerank-mode local --llm-mode mock
```

### Full local (cần GPU tốt):
```bash
python scripts/run_pipeline.py --rerank-mode local --llm-mode local
```

---

## Tuning nếu GPU hết memory

```bash
# Giảm batch size
python scripts/run_pipeline.py --rerank-mode local --dense-batch-size 8

# Giảm candidates
python scripts/run_pipeline.py --rerank-mode local --top-k-bm25-articles 100

# Chỉ test 5 cases
python scripts/run_pipeline.py --rerank-mode local --llm-mode mock --limit 5
```

---

## Output

Kết quả lưu vào `outputs/submissions/submission_<mode>_<timestamp>.json`

**Để evaluate:**
```bash
python scripts/evaluate.py --mode bm25 --top-k 20
python scripts/evaluate.py --submission outputs/submissions/submission_*.json
```

---

## Troubleshooting

**❌ Error: "Found no NVIDIA driver"**
→ Kiểm tra: `nvidia-smi`
→ Cài lại PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu130`

**❌ Error: "No such file or directory: models"**
→ Chạy: `python scripts/download_models.py`

**❌ Memory error on T1200**
→ Dùng: `python scripts/run_pipeline.py --rerank-mode local --llm-mode mock --dense-batch-size 8`

---

## Kiến trúc

```
case_query
    ↓
BM25 (top-200)         [fast, sparse]
    ↓
BGE-M3 (dense+sparse)  [semantic, GPU]
    ↓
LLM Reranker (top-15)  [relevance filter, CPU]
    ↓
LLM Predictor (label)  [outcome prediction, GPU]
```

