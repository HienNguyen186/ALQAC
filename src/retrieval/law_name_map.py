"""Map Vietnamese law names and article numbers to corpus law_id/aid pairs."""

from __future__ import annotations

# Auto-add project root when this file is run directly.
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


import re
from typing import Any

from src.utils.text import normalize_text

# ---------------------------------------------------------------------------
# Map normalised law name → corpus law_id
# Keys are already accent-stripped (produced by normalize_text with strip_accents=True).
# ---------------------------------------------------------------------------
LAW_NAME_TO_ID: dict[str, str] = {
    "bo luat dan su nam 2015": "91/2015/QH13",
    "bo luat dan su 2015":     "91/2015/QH13",
    "bo luat dan su":          "91/2015/QH13",
    "bo luat to tung dan su nam 2015": "92/2015/QH13",
    "bo luat to tung dan su 2015":     "92/2015/QH13",
    "bo luat to tung dan su":          "92/2015/QH13",
    "luat dat dai nam 2013": "45/2013/QH13",
    "luat dat dai 2013":     "45/2013/QH13",
    "luat dat dai":          "45/2013/QH13",
    "nghi quyet so 326/2016": "326/2016/UBTVQH14",
    "nghi quyet 326/2016":    "326/2016/UBTVQH14",
    "luat thi hanh an dan su": "26/2008/QH12",
    "luat hon nhan va gia dinh nam 2014": "52/2014/QH13",
    "luat hon nhan va gia dinh 2014":     "52/2014/QH13",
    "luat hon nhan va gia dinh":          "52/2014/QH13",
    "luat cac to chuc tin dung 2010": "47/2010/QH12",
    "luat cac to chuc tin dung":      "47/2010/QH12",
    "luat xay dung nam 2014": "50/2014/QH13",
    "luat xay dung 2014":     "50/2014/QH13",
    "luat ho tich": "60/2014/QH13",
    "nghi dinh so 37/2015": "37/2015/NĐ-CP",
    "nghi dinh 37/2015":    "37/2015/NĐ-CP",
    "luat khieu nai": "02/2011/QH13",
    "luat to tung hanh chinh": "93/2015/QH13",
    "luat kinh doanh bat dong san nam 2014": "66/2014/QH13",
    "luat kinh doanh bat dong san 2014":     "66/2014/QH13",
    "luat kinh doanh bat dong san":          "66/2014/QH13",
    "luat nuoi con nuoi": "52/2010/QH12",
    "luat nguoi cao tuoi": "39/2009/QH12",
}

# ---------------------------------------------------------------------------
# Stale-version markers: law names that reference obsolete statute editions
# not present in the corpus.  These must NOT be fuzzy-matched to current laws
# because article numbering is incompatible.
# ---------------------------------------------------------------------------
_STALE_YEAR_MARKERS = ("1987", "1995", "2000", "2001", "2003", "2004", "2005", "2006", "2008", "2009")

# ---------------------------------------------------------------------------
# LAW_AID_OFFSET: corpus law_id → base aid of article 1 (aid = offset + dieu - 1)
# IMPORTANT: keys must match the law_id strings stored in the corpus JSON exactly.
# The corpus uses "NĐ-CP" (Unicode Đ), not "ND-CP".
# ---------------------------------------------------------------------------
LAW_AID_OFFSET: dict[str, int] = {
    "47/2010/QH12":      270,
    "66/2014/QH13":      819,
    "24/2012/NĐ-CP":    1470,   # Unicode Đ — matches corpus
    "60/2014/QH13":     3448,
    "52/2010/QH12":     3525,
    "26/2008/QH12":     5266,
    "19/2011/NĐ-CP":   7613,   # Unicode Đ — matches corpus
    "326/2016/UBTVQH14": 13600,
    "02/2011/QH13":    13287,
    "93/2015/QH13":    14306,
    "37/2015/NĐ-CP":   4054,   # Unicode Đ — matches corpus
    "39/2009/QH12":    52197,
    "91/2015/QH13":    52771,
    "92/2015/QH13":    50666,
    "52/2014/QH13":    53873,
    "45/2013/QH13":    55951,
    "100/2015/QH13":   56445,
    "50/2014/QH13":    56963,
}


def normalize_law_name(name: str) -> str:
    return normalize_text(name, strip_accents=True)


def _is_stale_law(normalized_name: str) -> bool:
    """Return True when the name explicitly references an obsolete version."""
    return any(marker in normalized_name for marker in _STALE_YEAR_MARKERS)


def law_name_to_id(name: str) -> str | None:
    """
    Map a Vietnamese law name string to a corpus law_id.

    Returns None for:
    - exact-miss with no fuzzy match
    - stale-version names (e.g. "Bộ luật Dân sự 2005") that must not be
      silently mapped to the 2015 edition (different article numbering).
    """
    normalized = normalize_law_name(name)

    # Fast path: exact match.
    if normalized in LAW_NAME_TO_ID:
        return LAW_NAME_TO_ID[normalized]

    # Fuzzy substring match — only for non-stale names.
    if _is_stale_law(normalized):
        return None

    best_key = ""
    best_id: str | None = None
    for key, law_id in LAW_NAME_TO_ID.items():
        if key in normalized and len(key) > len(best_key):
            best_key = key
            best_id = law_id
    return best_id


def parse_article_number(text: str) -> int | None:
    normalized = normalize_text(text, strip_accents=True)
    match = re.search(r"\bdieu\s+(\d+)", normalized, re.IGNORECASE)
    return int(match.group(1)) if match else None


def dieu_to_aid(law_id: str, dieu_number: int) -> int | None:
    offset = LAW_AID_OFFSET.get(law_id)
    if offset is None:
        return None
    return offset + dieu_number - 1


def parse_law_provisions(raw: str) -> list[dict[str, Any]]:
    """Parse related_law_provisions into [{law_id, aid}] records."""

    results: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        if "|" not in line:
            continue
        law_name, article_text = [part.strip() for part in line.split("|", 1)]
        law_id = law_name_to_id(law_name)
        article_number = parse_article_number(article_text)
        if law_id is None or article_number is None:
            continue
        aid = dieu_to_aid(law_id, article_number)
        if aid is not None:
            results.append({"law_id": law_id, "aid": aid})

    seen: set[tuple[str, int]] = set()
    unique: list[dict[str, Any]] = []
    for item in results:
        key = (item["law_id"], int(item["aid"]))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
