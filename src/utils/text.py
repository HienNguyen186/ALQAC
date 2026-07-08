"""Vietnamese-oriented text normalization and tokenization."""

from __future__ import annotations

import re
import unicodedata

try:
    from ftfy import fix_text as _ftfy_fix_text
except Exception:  # pragma: no cover - optional dependency
    _ftfy_fix_text = None

_WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)
_STOPWORDS = {
    "va", "cua", "la", "co", "cac", "mot", "nhung", "duoc", "khong", "cho",
    "trong", "ve", "voi", "theo", "tai", "tu", "den", "nay", "do", "bi", "ben",
}


def normalize_text(text: object, *, strip_accents: bool = False) -> str:
    """Normalize legal Vietnamese text for matching and prompting."""

    value = "" if text is None else str(text)
    if _ftfy_fix_text is not None:
        value = _ftfy_fix_text(value)
    value = unicodedata.normalize("NFKC", value).lower()
    value = value.replace("đ", "d") if strip_accents else value
    if strip_accents:
        value = "".join(
            ch for ch in unicodedata.normalize("NFD", value)
            if unicodedata.category(ch) != "Mn"
        )
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: object) -> list[str]:
    """Tokenize Vietnamese legal text with normalization and light filtering."""

    normalized = normalize_text(text, strip_accents=True)
    tokens = _WORD_RE.findall(normalized)
    return [tok for tok in tokens if len(tok) > 1 and tok not in _STOPWORDS]


def stable_hash(text: object) -> int:
    """Small deterministic hash for mock-mode tie breaking."""

    value = normalize_text(text, strip_accents=True)
    total = 0
    for idx, char in enumerate(value, 1):
        total = (total + idx * ord(char)) % 1_000_003
    return total
