"""Rule-based query expansion for Vietnamese civil-law retrieval.

Nâng cấp: thêm rule PHỔ QUÁT cho các luật thủ tục/tố tụng gần như luôn xuất
hiện trong ground-truth của MỌI vụ án dân sự (thẩm quyền, án phí, thi hành án)
nhưng bị BM25 bỏ sót vì case_query hiếm khi nhắc trực tiếp các từ này.

Debug data cho thấy:
  92/2015/QH13 (Bộ luật Tố tụng Dân sự): 14 provisions trong GT, chỉ retrieve 7%
  26/2008/QH12 (Luật Thi hành án Dân sự): 5 provisions trong GT, retrieve 0%
"""

from __future__ import annotations

from src.utils.text import normalize_text


EXPANSION_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("boi thuong thiet hai ngoai hop dong", "thiet hai ngoai hop dong"),
        "boi thuong thiet hai can cu phat sinh trach nhiem dieu 584 dieu 585 dieu 589 dieu 590 bo luat dan su",
    ),
    (
        ("con cho", "suc vat", "vat nuoi", "tha rong", "can nguoi"),
        "suc vat gay thiet hai chu so huu boi thuong dieu 603 bo luat dan su",
    ),
    (
        ("hop dong vay", "vay tien", "no goc", "lai suat"),
        "hop dong vay tai san nghia vu tra no lai suat dieu 463 dieu 466 dieu 468 bo luat dan su",
    ),
    (
        ("hop dong chuyen nhuong", "mua ban", "dat dai", "quyen su dung dat"),
        "chuyen nhuong quyen su dung dat hop dong dat dai dieu kien hieu luc tranh chap dat",
    ),
    (
        ("hon nhan", "ly hon", "nuoi con", "cap duong"),
        "ly hon quyen nuoi con cap duong tai san chung luat hon nhan va gia dinh",
    ),
    (
        ("thua ke", "di chuc", "di san"),
        "thua ke di chuc di san hang thua ke chia di san bo luat dan su",
    ),
    (
        ("an phi", "le phi", "chi phi to tung"),
        "an phi le phi toa an nghi quyet 326 2016 ubtvqh14",
    ),
    (
        ("lai cham tra", "cham thanh toan", "cham tra"),
        "nghia vu cham tra lai cham tra dieu 357 dieu 468 bo luat dan su",
    ),
    # ------------------------------------------------------------------
    # ✅ MỚI: Rule phổ quát — kích hoạt cho HẦU HẾT mọi vụ án dân sự.
    # Trigger bằng các từ khóa gần như luôn xuất hiện trong case_query
    # ("kiện", "khởi kiện", "tranh chấp", "yêu cầu") để bù đắp việc
    # BM25 hiếm khi match trực tiếp "tố tụng" hay "thi hành án".
    # ------------------------------------------------------------------
    (
        ("khoi kien", "tranh chap", "yeu cau toa an", "nguyen don", "bi don"),
        "tham quyen giai quyet thu tuc to tung dan su khoi kien nghia vu chung minh "
        "bo luat to tung dan su nghia vu thi hanh an dan su",
    ),
    (
        ("thi hanh an", "cuong che", "ke bien tai san"),
        "thi hanh an dan su cuong che ke bien tai san luat thi hanh an dan su",
    ),
)


def expand_query(query: str, extra_text: str | None = None) -> str:
    """Append legal-domain hints when a query contains known dispute terms."""

    base = " ".join(part for part in (query, extra_text or "") if part)
    normalized = normalize_text(base, strip_accents=True)
    additions: list[str] = []
    for triggers, expansion in EXPANSION_RULES:
        if any(trigger in normalized for trigger in triggers):
            additions.append(expansion)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"