"""Common path, logging, and JSON helpers for the ALQAC pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_MARKERS = (".git", "configs", "src")


class FriendlyFileError(FileNotFoundError):
    """Raised when an expected project file is missing."""


def find_project_root(start: str | Path | None = None) -> Path:
    """Return the project root by walking upward from *start*.

    The function works on Windows, Linux, and Colab. It does not depend on the
    current shell directory, which prevents ModuleNotFoundError/FileNotFoundError
    when scripts are launched from another folder.
    """

    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return Path.cwd().resolve()


def ensure_project_on_path(root: Path | None = None) -> Path:
    """Insert the project root into sys.path and return it."""

    project_root = root or find_project_root()
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve *path* relative to the detected project root when needed."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return ((root or find_project_root()) / candidate).resolve()


def require_file(path: str | Path, description: str, root: Path | None = None) -> Path:
    """Validate that a required file exists, otherwise raise a friendly error."""

    resolved = resolve_path(path, root)
    if resolved.is_file():
        return resolved

    rel_hint = Path(path).as_posix()
    message = (
        f"Missing required file: {description}\n"
        f"Expected location: {resolved}\n"
        f"Suggestion: place the file at '{rel_hint}' under the project root, "
        "or pass an explicit path with the matching CLI argument."
    )
    raise FriendlyFileError(message)


def load_json_file(path: str | Path, description: str, root: Path | None = None) -> Any:
    """Load a JSON file after friendly validation."""

    resolved = require_file(path, description, root)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_case_records(cases: Any) -> list[dict[str, Any]]:
    """Validate public/private test records used by the pipeline."""

    if not isinstance(cases, list):
        raise ValueError("Test data must be a JSON list of case records.")
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Case #{idx} is not an object.")
        missing = [field for field in ("case_id", "case_query") if field not in case]
        if missing:
            raise ValueError(f"Case #{idx} is missing required field(s): {', '.join(missing)}")
    return cases


def validate_corpus_records(corpus: Any) -> list[dict[str, Any]]:
    """Validate law corpus records used by the retriever."""

    if not isinstance(corpus, list):
        raise ValueError("Law corpus must be a JSON list of law records.")
    for idx, law in enumerate(corpus):
        if not isinstance(law, dict):
            raise ValueError(f"Law #{idx} is not an object.")
        if "law_id" not in law or "content" not in law:
            raise ValueError(f"Law #{idx} must contain 'law_id' and 'content'.")
        if not isinstance(law["content"], list):
            raise ValueError(f"Law {law.get('law_id', idx)} content must be a list.")
    return corpus


def setup_logging(level: str = "INFO") -> None:
    """Configure compact console logging once."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
        force=True,
    )


def unique_by_key(items: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Return items deduplicated by a tuple of dictionary keys."""

    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        key = tuple(item.get(k) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
