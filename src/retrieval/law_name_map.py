"""
Mapping tên luật (từ related_law_provisions) → law_id trong corpus.

Tại sao cần file này:
- Ground truth dùng tên tiếng Việt tự nhiên, viết không thống nhất
  VD: "Bộ luật Dân sự năm 2015" / "Bộ luật Dân sự 2015" / "Bộ luật Dân sự"
- Corpus dùng mã văn bản: "91/2015/QH13"
- aid trong corpus KHÔNG phải số điều — là global row ID
  VD: BLDS 2015 Điều 584 → aid = 52771 + 584 - 1 = 53354
"""

import re

# ─── Tên luật → law_id ────────────────────────────────────────────────────────
LAW_NAME_TO_ID: dict[str, str] = {
    "bộ luật dân sự năm 2015":             "91/2015/QH13",
    "bộ luật dân sự 2015":                 "91/2015/QH13",
    "bộ luật tố tụng dân sự năm 2015":     "92/2015/QH13",
    "bộ luật tố tụng dân sự 2015":         "92/2015/QH13",
    "bộ luật tố tụng dân sự":              "92/2015/QH13",
    "bộ luật tố tụng":                     "92/2015/QH13",
    "luật đất đai năm 2013":               "45/2013/QH13",
    "luật đất đai 2013":                   "45/2013/QH13",
    "luật đất đai":                        "45/2013/QH13",
    "nghị quyết số 326/2016":              "326/2016/UBTVQH14",
    "nghị quyết 326/2016":                 "326/2016/UBTVQH14",
    "luật thi hành án dân sự":             "26/2008/QH12",
    "luật hôn nhân và gia đình năm 2014":  "52/2014/QH13",
    "luật hôn nhân và gia đình 2014":      "52/2014/QH13",
    "luật hôn nhân và gia đình":           "52/2014/QH13",
    "luật các tổ chức tín dụng 2010":      "47/2010/QH12",
    "luật các tổ chức tín dụng":           "47/2010/QH12",
    "luật xây dựng năm 2014":              "50/2014/QH13",
    "luật xây dựng 2014":                  "50/2014/QH13",
    "luật hộ tịch":                        "60/2014/QH13",
    "nghị định số 37/2015":                "37/2015/NĐ-CP",
    "nghị định 37/2015":                   "37/2015/NĐ-CP",
    "luật khiếu nại":                      "02/2011/QH13",
    "luật tố tụng hành chính":             "93/2015/QH13",
    "luật kinh doanh bất động sản năm 2014": "66/2014/QH13",
    "luật kinh doanh bất động sản 2014":   "66/2014/QH13",
    "luật nuôi con nuôi":                  "52/2010/QH12",
    "luật người cao tuổi":                 "39/2009/QH12",
}

# ─── Offset table: aid của Điều N = LAW_AID_OFFSET[law_id] + N - 1 ───────────
LAW_AID_OFFSET: dict[str, int] = {
    "47/2010/QH12":       270,
    "66/2014/QH13":       819,
    "24/2012/NĐ-CP":     1470,
    "60/2014/QH13":      3448,
    "52/2010/QH12":      3525,
    "26/2008/QH12":      5266,
    "19/2011/NĐ-CP":     7613,
    "326/2016/UBTVQH14": 13600,
    "02/2011/QH13":      13287,
    "93/2015/QH13":      14306,
    "37/2015/NĐ-CP":      4054,
    "39/2009/QH12":      52197,
    "91/2015/QH13":      52771,
    "92/2015/QH13":      50666,
    "52/2014/QH13":      53873,
    "45/2013/QH13":      55951,
    "100/2015/QH13":     56445,
    "50/2014/QH13":      56963,
}


def normalize_law_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def law_name_to_id(name: str) -> str | None:
    normalized = normalize_law_name(name)
    if normalized in LAW_NAME_TO_ID:
        return LAW_NAME_TO_ID[normalized]
    best_key, best_id = "", None
    for key, law_id in LAW_NAME_TO_ID.items():
        if key in normalized and len(key) > len(best_key):
            best_key, best_id = key, law_id
    return best_id


def parse_article_number(text: str) -> int | None:
    """
    Trích xuất số điều từ:
      "Điều 584"              → 584
      "Khoản 5 Điều 26"      → 26
      "Điểm a Khoản 1 Điều 37" → 37
    """
    match = re.search(r"điều\s+(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def dieu_to_aid(law_id: str, dieu_number: int) -> int | None:
    """
    (law_id, số điều) → aid thật trong corpus.
    VD: ("91/2015/QH13", 584) → 52771 + 584 - 1 = 53354
    """
    offset = LAW_AID_OFFSET.get(law_id)
    if offset is None:
        return None
    return offset + dieu_number - 1


def parse_law_provisions(raw: str) -> list[dict]:
    """
    Parse related_law_provisions → list of {"law_id": ..., "aid": <corpus aid>}
    Bỏ qua luật không có trong corpus (BLDS 2005, 1995...).
    """
    results = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        parts = line.split("|", 1)
        law_name    = parts[0].strip()
        article_text = parts[1].strip()

        law_id = law_name_to_id(law_name)
        if law_id is None:
            continue

        dieu = parse_article_number(article_text)
        if dieu is None:
            continue

        aid = dieu_to_aid(law_id, dieu)
        if aid is None:
            continue

        results.append({"law_id": law_id, "aid": aid})

    # Deduplicate
    seen, unique = set(), []
    for item in results:
        key = (item["law_id"], item["aid"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
