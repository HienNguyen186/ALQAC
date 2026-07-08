"""
Utility functions for loading local HuggingFace models completely offline.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Folder chứa models
MODELS_ROOT = PROJECT_ROOT / "models"

# Khai báo các model trong project
MODELS = {
    "bge_m3": MODELS_ROOT / "models--BAAI--bge-m3",
    "bge_reranker": MODELS_ROOT / "models--BAAI--bge-reranker-v2-m3",
    "qwen3": MODELS_ROOT / "models--Qwen--Qwen3-8B",
    "qwen25": MODELS_ROOT / "models--Qwen--Qwen2.5-3B-Instruct",
}


def get_snapshot(model_name: str) -> Path:
    """
    Return snapshot directory of a local HuggingFace model.

    Example:
        get_snapshot("bge_m3")
    """

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODELS.keys())}"
        )

    snapshots_dir = MODELS[model_name] / "snapshots"

    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"Cannot find snapshot folder:\n{snapshots_dir}"
        )

    snapshots = sorted(
        [p for p in snapshots_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if len(snapshots) == 0:
        raise FileNotFoundError(
            f"No snapshot inside:\n{snapshots_dir}"
        )

    return snapshots[0]


def get_model_path(model_name: str) -> str:
    """
    Return string path for HuggingFace loading.
    """
    return str(get_snapshot(model_name))


def print_models():
    """
    Print all detected local models.
    """
    print("=" * 60)

    for name in MODELS:

        try:
            print(f"{name:<15} -> {get_snapshot(name)}")

        except Exception as e:
            print(f"{name:<15} -> NOT FOUND ({e})")

    print("=" * 60)