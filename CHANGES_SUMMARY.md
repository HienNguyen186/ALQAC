# Cập nhật — Offline Mode (Không download lại model)

## ✅ Những gì đã sửa

### 1. **Cache setup tự động trong 3 scripts chính**

**Thêm vào đầu:**
- `scripts/run_pipeline.py`
- `scripts/main.py`  
- `scripts/evaluate.py`

```python
import os
_cache_dir = PROJECT_ROOT / "models"
if _cache_dir.exists():
    os.environ.setdefault("HF_HOME", str(_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_cache_dir))
```

**Kết quả:** Code tự động tìm `./models/` nếu tồn tại, không download lại.

---

### 2. **Tạo `download_models.py`**

Script download tất cả model (chỉ chạy 1 lần):

```bash
python scripts/download_models.py
```

Download vào `./models/`:
- `BAAI/bge-m3` (~2.3 GB)
- `Qwen/Qwen2.5-3B-Instruct` (~7 GB)
- `Qwen/Qwen3-8B` (~15 GB)

---

### 3. **Tạo `run_local.bat` (Windows)**

Click file để chạy pipeline local mode:
```
run_local.bat
```

Tự động:
1. Kiểm tra `models/` directory
2. Verify GPU (`CUDA: True/False`)
3. Chạy pipeline 10 cases

---

### 4. **Tạo `QUICK_START.md`**

Hướng dẫn nhanh:
- Cách tải model
- Chế độ chạy (mock/local)
- Tuning GPU memory
- Troubleshooting

---

## 🚀 Cách sử dụng

### **Lần đầu (khi có internet):**
```bash
python scripts/download_models.py
```

### **Lần sau (offline, không download):**
```bash
# Windows
run_local.bat

# PowerShell / Linux / Mac
python scripts/run_pipeline.py --rerank-mode local --llm-mode mock
```

---

## 📦 Files thay đổi

| File | Thay đổi |
|---|---|
| `scripts/run_pipeline.py` | ✏️ Thêm cache setup |
| `scripts/main.py` | ✏️ Thêm cache setup |
| `scripts/evaluate.py` | ✏️ Thêm cache setup |
| `scripts/download_models.py` | ✨ **Mới** |
| `run_local.bat` | ✨ **Mới** |
| `QUICK_START.md` | ✨ **Mới** |

---

## ✔️ Tested

- ✅ Mock mode (không model)
- ✅ Config loading (bm25_top_k_articles=200, final_max_k=15)
- ✅ BGE-M3 FlagEmbedding API (dense + sparse)
- ✅ Cache tự động detect (nếu `./models/` tồn tại)

---

## 💾 Dung lượng

Total models: ~25 GB (chia nhỏ):
- BGE-M3: 2.3 GB
- Qwen2.5-3B: 7 GB
- Qwen3-8B: 15 GB

