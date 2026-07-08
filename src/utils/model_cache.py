"""
Utilities for managing local Hugging Face models and cache.

This module centralizes:
    - Hugging Face cache configuration
    - Local model lookup
    - Snapshot discovery
    - Model validation

All models are expected to be stored inside:

project_root/
└── models/
    ├── models--BAAI--bge-m3/
    ├── models--BAAI--bge-reranker-v2-m3/
    ├── models--Qwen--Qwen3-8B/
    └── ...

Each model follows the standard HuggingFace cache layout.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.io import find_project_root

_CONFIGURED_DIR: Path | None = None

###############################################################################
# Project folders
###############################################################################

PROJECT_ROOT = find_project_root()

MODELS_ROOT = PROJECT_ROOT / "models"

###############################################################################
# Mapping: HuggingFace repo -> local cache folder
###############################################################################

LOCAL_MODELS = {
    "BAAI/bge-m3": "models--BAAI--bge-m3",
    "BAAI/bge-reranker-v2-m3": "models--BAAI--bge-reranker-v2-m3",
    "Qwen/Qwen3-8B": "models--Qwen--Qwen3-8B",
    "Qwen/Qwen2.5-3B-Instruct": "models--Qwen--Qwen2.5-3B-Instruct",
}

###############################################################################
# Cache configuration
###############################################################################


def configure_hf_cache(cache_dir: str | Path | None = None) -> Path:
    """
    Configure HuggingFace cache directories.

    By default, use:

        project/models

    instead of ~/.cache/huggingface

    Safe to call multiple times.
    """

    global _CONFIGURED_DIR

    if _CONFIGURED_DIR is not None and cache_dir is None:
        return _CONFIGURED_DIR

    resolved = (
        Path(cache_dir).expanduser()
        if cache_dir
        else Path(
            os.getenv(
                "ALQAC_MODEL_CACHE_DIR",
                str(MODELS_ROOT),
            )
        )
    )

    resolved.mkdir(parents=True, exist_ok=True)

    hub_dir = resolved / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)

    st_dir = resolved / "sentence-transformers"
    st_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(resolved)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_dir)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(st_dir)

    _CONFIGURED_DIR = resolved

    return resolved


###############################################################################
# Local model utilities
###############################################################################


def get_snapshot(model_name: str) -> Path:
    """
    Return snapshot directory of a local model.

    Example:

        get_snapshot("Qwen/Qwen3-8B")

    ->
        models/models--Qwen--Qwen3-8B/snapshots/5617a9...
    """

    if model_name not in LOCAL_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'.\n"
            f"Available models:\n"
            + "\n".join(LOCAL_MODELS.keys())
        )

    model_dir = MODELS_ROOT / LOCAL_MODELS[model_name]

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Local model folder not found:\n{model_dir}"
        )

    snapshot_root = model_dir / "snapshots"

    if not snapshot_root.exists():
        raise FileNotFoundError(
            f"No snapshots folder:\n{snapshot_root}"
        )

    snapshots = sorted(
        [p for p in snapshot_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not snapshots:
        raise FileNotFoundError(
            f"No snapshot found in:\n{snapshot_root}"
        )

    return snapshots[0]


def get_model_path(model_name: str) -> str:
    """
    Return snapshot path as string.

    This path can be passed directly into

        AutoModel.from_pretrained()

        AutoTokenizer.from_pretrained()

        CrossEncoder()

        BGEM3FlagModel()
    """

    return str(get_snapshot(model_name))


###############################################################################
# Diagnostics
###############################################################################


def is_model_cached(model_name: str) -> bool:
    """
    Check whether a local snapshot exists.
    """

    try:
        get_snapshot(model_name)
        return True
    except Exception:
        return False


def print_available_models() -> None:
    """
    Print all local models and their snapshot paths.
    """

    print("=" * 80)
    print("LOCAL MODELS")
    print("=" * 80)

    for repo in LOCAL_MODELS:

        try:
            snapshot = get_snapshot(repo)
            print(f"✓ {repo}")
            print(f"    {snapshot}")

        except Exception as e:
            print(f"✗ {repo}")
            print(f"    {e}")

    print("=" * 80)