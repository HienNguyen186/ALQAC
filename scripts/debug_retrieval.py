"""Debug retrieval — tìm nguyên nhân Recall@20 chỉ ~10%.

Chạy:
    python scripts/debug_retrieval.py --test data/raw/ALQAC2026_public_test.json --corpus data/raw/corpus_law_pub.json --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import LawRetriever
from src.retrieval.law_name_map import parse_law_provisions
from src.utils.io import load_json_file, resolve_path, setup_logging, validate_case_records

LOGGER = __import__("logging").getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug retrieval Recall@k")
    parser.add_argument("--test", default="data/raw/ALQAC2026_public_test.json")
    parser.add_argument("--corpus", default="data/raw/corpus_law_pub.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    retriever = LawRetriever(resolve_path(args.corpus, PROJECT_ROOT))
    cases = validate_case_records(load_json_file(resolve_path(args.test, PROJECT_ROOT), "test"))
    cases = cases[:args.limit]

    LOGGER.info("=" * 70)
    LOGGER.info("DEBUG RETRIEVAL — kiểm tra tại sao Recall@k thấp")
    LOGGER.info("=" * 70)

    all_recalls = []
    law_coverage = Counter()  # Đếm tần suất law_id trong GT
    retrieved_law_coverage = Counter()  # Đếm tần suất lấy được law_id nào

    for case_idx, case in enumerate(cases, 1):
        case_id = str(case["case_id"])
        query = str(case["case_query"])
        gt_raw = case.get("related_law_provisions", "")

        # Parse ground truth
        gt_list = parse_law_provisions(gt_raw)
        gt_set = {(str(i["law_id"]), int(i["aid"])) for i in gt_list if i.get("aid") is not None}

        if not gt_set:
            LOGGER.info("[Case %d] %s — NO GROUND TRUTH, skip", case_idx, case_id)
            continue

        # Retrieve top-k
        retrieved = retriever.retrieve(query, top_k=args.top_k)
        pred_set = {(str(i["law_id"]), int(i["aid"])) for i in retrieved}

        # Metrics
        tp = len(pred_set & gt_set)
        recall = tp / len(gt_set) if gt_set else 0
        all_recalls.append(recall)

        # Track which laws are in GT vs retrieved
        for law_id, aid in gt_set:
            law_coverage[law_id] += 1
        for law_id, aid in (pred_set & gt_set):
            retrieved_law_coverage[law_id] += 1

        LOGGER.info("")
        LOGGER.info("[Case %d/%d] %s", case_idx, len(cases), case_id)
        LOGGER.info("  Query (first 100 chars): %s...", query[:100])
        LOGGER.info("  GT: %d provisions", len(gt_set))
        for law_id, aid in sorted(gt_set)[:5]:
            LOGGER.info("    - %s aid=%s", law_id, aid)
        if len(gt_set) > 5:
            LOGGER.info("    ... + %d more", len(gt_set) - 5)

        LOGGER.info("  Retrieved top-20: %d provisions", len(pred_set))
        for law_id, aid in sorted(pred_set)[:5]:
            LOGGER.info("    - %s aid=%s (BM25 rank %d)", law_id, aid,
                        next((i for i, r in enumerate(retrieved, 1) if r["law_id"] == law_id and r["aid"] == aid), -1))
        if len(pred_set) > 5:
            LOGGER.info("    ... + %d more", len(pred_set) - 5)

        LOGGER.info("  Recall@%d: %d/%d = %.2f%%", args.top_k, tp, len(gt_set), recall * 100)

        # Hiển thị những GT bị miss
        missed = gt_set - pred_set
        if missed:
            LOGGER.warning("  MISSED GT (bị BM25 bỏ): %d provisions", len(missed))
            for law_id, aid in sorted(missed)[:3]:
                LOGGER.warning("    - %s aid=%s", law_id, aid)
            if len(missed) > 3:
                LOGGER.warning("    ... + %d more", len(missed) - 3)

    LOGGER.info("")
    LOGGER.info("=" * 70)
    if all_recalls:
        avg_recall = sum(all_recalls) / len(all_recalls)
        LOGGER.info("Average Recall@%d across %d cases: %.4f (%.2f%%)",
                    args.top_k, len(all_recalls), avg_recall, avg_recall * 100)
    LOGGER.info("=" * 70)

    LOGGER.info("")
    LOGGER.info("LAW COVERAGE ANALYSIS:")
    LOGGER.info("  Laws in ground truth (tất cả case):")
    for law_id, cnt in law_coverage.most_common():
        retrieved_cnt = retrieved_law_coverage.get(law_id, 0)
        coverage_pct = retrieved_cnt / cnt * 100 if cnt else 0
        LOGGER.info("    %s: %d provisions (retrieved %d = %.1f%%)",
                    law_id, cnt, retrieved_cnt, coverage_pct)

    LOGGER.info("")
    LOGGER.info("DIAGNOSIS:")
    LOGGER.info("  Nếu Recall < 20%:")
    LOGGER.info("    → BM25 query không match được tốt (text mismatch, dấu phụ, etc)")
    LOGGER.info("    → Cần: query expansion, tokenization fix, hoặc tăng top_k")
    LOGGER.info("")
    LOGGER.info("  Nếu một law_id coverage = 0% (không bao giờ lấy được):")
    LOGGER.info("    → Law này missing từ corpus hoặc format khác")
    LOGGER.info("    → Kiểm tra corpus_law_pub.json xem có law_id đó không")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
