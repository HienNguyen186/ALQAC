"""Configuration loading for ALQAC scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import find_project_root, require_file

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "law_corpus": "data/raw/corpus_law_pub.json",
        "public_test": "data/raw/ALQAC2026_public_test.json",
    },
    "retrieval": {
        "bm25_top_k_laws": 5,
        "dense_top_k": 20,
        "final_min_k": 2,
        "final_max_k": 5,
        "dense_model": "BAAI/bge-m3",
        "dense_batch_size": 32,
        "rerank_threshold": 0.5,
    },
    "prediction": {
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "predictor_model": "Qwen/Qwen3-8B",
    },
    "case_api": {
        "mode": "mock",
        "base_url": "https://alqac2026-leaderboard.ngrok.app",
        "max_queries": 4,
        "only_when_missing_case_fact": True,
    },
    "output": {"submissions_dir": "outputs/submissions"},
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML configuration, falling back to DEFAULT_CONFIG if absent."""

    root = find_project_root()
    path = root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    if not path.exists():
        return DEFAULT_CONFIG
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Reading config.yaml requires pyyaml. Install requirements.txt.") from exc
    with require_file(path, "pipeline configuration").open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_update(DEFAULT_CONFIG, loaded)
