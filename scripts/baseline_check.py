"""Giai đoạn 0 — Baseline check cho ALQAC 2026.

Tính:
  1. Phân phối 4 nhãn verdict_label trong test set.
  2. Accuracy nếu luôn đoán majority class (sàn tối thiểu).
  3. Ghi ra majority class để dùng làm fallback ở Giai đoạn 1.

Chạy:
    python scripts/baseline_check.py --test data/raw/ALQAC2026_public_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_json_file, resolve_path, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline majority-class check.")
    parser.add_argument("--test", default="data/raw/ALQAC2026_public_test.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    cases = load_json_file(resolve_path(args.test, PROJECT_ROOT), "ALQAC public test set")

    labels = [str(c["verdict_label"]) for c in cases if c.get("verdict_label")]
    missing = len(cases) - len(labels)

    if not labels:
        print("Không tìm thấy field 'verdict_label' nào trong test set — "
              "không thể tính baseline. Kiểm tra lại file test.")
        return 1

    counts = Counter(labels)
    total = len(labels)
    majority_label, majority_count = counts.most_common(1)[0]
    majority_acc = majority_count / total

    print("=" * 60)
    print(f"Tổng số case có nhãn: {total}  (thiếu nhãn: {missing})")
    print("-" * 60)
    print(f"{'Nhãn':<18}{'Số lượng':>10}{'Tỷ lệ %':>12}")
    for label, cnt in counts.most_common():
        print(f"{label:<18}{cnt:>10}{cnt/total*100:>11.2f}%")
    print("-" * 60)
    print(f"Majority-class baseline : {majority_label} "
          f"→ accuracy sàn = {majority_acc:.4f} ({majority_acc*100:.2f}%)")
    print("=" * 60)

    if majority_acc >= 0.70:
        print("⚠️  Majority-class baseline đã >= 70%. Nghĩa là chỉ cần model "
              "không tệ hơn 'đoán bừa nhãn phổ biến nhất' cũng gần đạt mục tiêu. "
              "Ưu tiên kiểm tra vì sao pipeline hiện tại đang thấp hơn mức này.")
    else:
        print(f"Majority-class baseline < 70% — cần model thực sự học được tín hiệu "
              f"từ evidence, không thể 'ăn may' bằng cách đoán 1 nhãn.")

    print()
    print(f"→ Dùng '{majority_label}' làm giá trị fallback mặc định trong "
          f"llm_predictor.py (Giai đoạn 1), thay vì hard-code PARTIAL_A_WIN.")

    out = {
        "total_labeled": total,
        "missing_label": missing,
        "distribution": dict(counts),
        "majority_label": majority_label,
        "majority_baseline_accuracy": round(majority_acc, 4),
    }
    out_path = resolve_path("outputs/baseline_check.json", PROJECT_ROOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
