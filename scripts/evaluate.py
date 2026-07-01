"""
Đánh giá BM25 retrieval locally trên public test.

Chạy từ thư mục gốc:
    python scripts/evaluate.py [--top-k 5]

Metrics:
  - Micro Law F1   : precision/recall/F1 trên từng điều luật
  - Coverage       : % cases có ít nhất 1 điều luật đúng
"""

import json
import sys
import argparse
from pathlib import Path

# Để import src.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.bm25_retriever import LawRetriever
from src.retrieval.law_name_map import parse_law_provisions


# ─── Metrics ────────────────────────────────────────────────────────────────

def law_f1(predicted: list[dict], ground_truth: list[dict]) -> dict:
    """
    So sánh predicted vs ground_truth ở mức (law_id, aid).
    Cả hai đều là list of {"law_id": ..., "aid": ...}.
    """
    pred_set = {(d["law_id"], int(d["aid"])) for d in predicted}
    gt_set   = {(d["law_id"], int(d["aid"])) for d in ground_truth}

    if not gt_set:          # case không có ground truth → bỏ qua
        return None

    tp        = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(gt_set)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "pred": len(pred_set), "gt": len(gt_set),
        "precision": precision, "recall": recall, "f1": f1,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5,
                        help="Số articles lấy từ BM25 (default: 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="In chi tiết từng case")
    args = parser.parse_args()

    corpus_path = "data/raw/corpus_law_pub.json"
    test_path   = "data/raw/ALQAC2026_public_test.json"

    print(f"[1/3] Loading law corpus...")
    retriever = LawRetriever(corpus_path)
    print(f"      → {len(retriever._articles)} articles indexed")

    print(f"[2/3] Loading public test...")
    with open(test_path, encoding="utf-8") as f:
        cases = json.load(f)
    print(f"      → {len(cases)} cases")

    print(f"[3/3] Evaluating BM25 top_k={args.top_k}...\n")

    results, skipped = [], 0

    for case in cases:
        query    = case["case_query"]
        gt_raw   = case.get("related_law_provisions", "")
        gt_list  = parse_law_provisions(gt_raw)

        # Lấy top-k articles từ BM25
        retrieved = retriever.retrieve(query, top_k=args.top_k)
        pred_list = [{"law_id": d["law_id"], "aid": d["aid"]} for d in retrieved]

        metrics = law_f1(pred_list, gt_list)

        if metrics is None:
            skipped += 1
            continue

        results.append(metrics)

        if args.verbose:
            print(f"  [{case['case_id']}]  P={metrics['precision']:.2f}  "
                  f"R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}  "
                  f"(pred={metrics['pred']}, gt={metrics['gt']}, tp={metrics['tp']})")

    # ─── Aggregate ──────────────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("Không có case nào để evaluate.")
        return

    avg_p  = sum(r["precision"] for r in results) / n
    avg_r  = sum(r["recall"]    for r in results) / n
    avg_f1 = sum(r["f1"]        for r in results) / n
    coverage = sum(1 for r in results if r["tp"] > 0) / n

    print("=" * 50)
    print(f"  top_k          : {args.top_k}")
    print(f"  Cases evaluated: {n}  (skipped: {skipped})")
    print(f"  Precision      : {avg_p:.4f}")
    print(f"  Recall         : {avg_r:.4f}")
    print(f"  Micro F1       : {avg_f1:.4f}   ← số quan trọng nhất")
    print(f"  Coverage       : {coverage:.4f}  (có ít nhất 1 hit)")
    print("=" * 50)


if __name__ == "__main__":
    main()
