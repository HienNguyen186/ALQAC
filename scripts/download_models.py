"""
Download all required models to ./models directory for offline use.

Chỉ chạy 1 lần khi có internet. Sau đó không cần download lại.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cache_dir = PROJECT_ROOT / "models"
cache_dir.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir)

print("=" * 70)
print("ALQAC 2026 — Model Downloader")
print("=" * 70)
print(f"Cache directory: {cache_dir}")
print()

# ── BGE-M3 ────────────────────────────────────────────────────────────
print("[1/4] Downloading BAAI/bge-m3 (~2.3 GB)...")
try:
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", cache_dir=str(cache_dir))
    print("✓ BGE-M3 downloaded")
except Exception as e:
    print(f"✗ BGE-M3 failed: {e}")

print()

# ── Vietnamese Embedding ──────────────────────────────────────────────
print("[2/4] Downloading dangvantuan/vietnamese-embedding (~400 MB)...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("dangvantuan/vietnamese-embedding", cache_folder=str(cache_dir))
    print("✓ Vietnamese embedding downloaded")
except Exception as e:
    print(f"✗ Vietnamese embedding failed: {e}")

print()

# ── Qwen2.5-3B ────────────────────────────────────────────────────────
print("[3/4] Downloading Qwen/Qwen2.5-3B-Instruct (~7 GB)...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", cache_dir=str(cache_dir))
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        cache_dir=str(cache_dir),
        torch_dtype="auto",
        device_map="auto"
    )
    print("✓ Qwen2.5-3B-Instruct downloaded")
except Exception as e:
    print(f"✗ Qwen2.5-3B failed: {e}")

print()

# ── Qwen3-8B ──────────────────────────────────────────────────────────
print("[4/4] Downloading Qwen/Qwen3-8B (~15 GB)...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", cache_dir=str(cache_dir))
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        cache_dir=str(cache_dir),
        torch_dtype="auto",
        device_map="auto"
    )
    print("✓ Qwen3-8B downloaded")
except Exception as e:
    print(f"✗ Qwen3-8B failed: {e}")

print()
print("=" * 70)
print("✅ All models downloaded to:", cache_dir)
print("=" * 70)
print()
print("Now you can run offline:")
print(f"  python scripts/run_pipeline.py --rerank-mode local --llm-mode mock")
