"""Evaluate retrieval and optional prediction outputs for ALQAC public data."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_cache_dir = PROJECT_ROOT / "models"
if _cache_dir.exists():
    os.environ.setdefault("HF_HOME", str(_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_cache_dir))

from src.retrieval.bm25_retriever import LawRetriever
from src.retrieval.law_name_map import parse_law_provisions
from src.reranking.dense_reranker import DenseRetriever
from src.reranking.llm_reranker import LLMReranker
from src.utils.io import FriendlyFileError, load_json_file, resolve_path, setup_logging, validate_case_records

LOGGER = logging.getLogger(__name__)


def law_f1(predicted: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[str, float] | None:
    pred_set = {(str(i["law_id"]), int(i["aid"])) for i in predicted if i.get("aid") is not None}
    gt_set   = {(str(i["law_id"]), int(i["aid"])) for i in ground_truth if i.get("aid") is not None}
    if not gt_set:
        return None
    tp        = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(gt_set)
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "pred": len(pred_set), "gt": len(gt_set),
            "precision": precision, "recall": recall, "f1": f1}


def macro_f1(labels: list[str], predictions: list[str]) -> float:
    unique_labels = sorted(set(labels) | set(predictions))
    scores = []
    for label in unique_labels:
        tp = sum(1 for y, p in zip(labels, predictions) if y == label and p == label)
        fp = sum(1 for y, p in zip(labels, predictions) if y != label and p == label)
        fn = sum(1 for y, p in zip(labels, predictions) if y == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall    = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def summarize_metrics(name: str, results: list[dict[str, float]], skipped: int) -> None:
    if not results:
        LOGGER.warning("No cases had parseable ground-truth legal provisions.")
        return
    n      = len(results)
    avg_p  = sum(r["precision"] for r in results) / n
    avg_r  = sum(r["recall"]    for r in results) / n
    avg_f1 = sum(r["f1"]        for r in results) / n
    cov    = sum(1 for r in results if r["tp"] > 0) / n
    LOGGER.info("%s evaluated=%s skipped=%s", name, n, skipped)
    LOGGER.info("Evidence Precision=%.4f Recall@k=%.4f Evidence F1=%.4f Coverage=%.4f",
                avg_p, avg_r, avg_f1, cov)


def evaluate_retrieval(corpus: str | Path, test: str | Path, top_k: int, verbose: bool = False) -> None:
    retriever = LawRetriever(resolve_path(corpus, PROJECT_ROOT))
    cases     = validate_case_records(load_json_file(resolve_path(test, PROJECT_ROOT), "ALQAC public test set"))
    results: list[dict[str, float]] = []
    skipped = 0

    for case in cases:
        gt_list   = parse_law_provisions(case.get("related_law_provisions", ""))
        retrieved = retriever.retrieve(str(case["case_query"]), top_k=top_k)
        pred_list = [{"law_id": i["law_id"], "aid": i["aid"]} for i in retrieved]
        metrics   = law_f1(pred_list, gt_list)
        if metrics is None:
            skipped += 1
            continue
        results.append(metrics)
        if verbose:
            LOGGER.info("%s P=%.2f R=%.2f F1=%.2f", case["case_id"],
                        metrics["precision"], metrics["recall"], metrics["f1"])

    summarize_metrics(f"BM25 top_k={top_k}", results, skipped)


def evaluate_pipeline_retrieval(
    corpus: str | Path,
    test: str | Path,
    top_k_bm25_articles: int,
    top_k_laws: int,
    top_k_dense: int,
    final_k: int,
    candidate_strategy: str,
    rerank_mode: str,
    dense_model: str,
    use_llm_reranker: bool,
    error_analysis: str | Path | None,
    verbose: bool = False,
) -> None:
    retriever   = LawRetriever(resolve_path(corpus, PROJECT_ROOT))
    dense       = DenseRetriever(
        mode=rerank_mode,
        model_name=dense_model,
        weight_dense=0.6,
        weight_sparse=0.4,
    )
    llm_reranker = LLMReranker(mode=rerank_mode) if use_llm_reranker else None
    cases        = validate_case_records(load_json_file(resolve_path(test, PROJECT_ROOT), "ALQAC public test set"))

    results: list[dict[str, float]] = []
    skipped = 0
    rows: list[dict[str, Any]] = []

    for case in cases:
        query   = str(case["case_query"])
        gt_list = parse_law_provisions(case.get("related_law_provisions", ""))
        candidates = retriever.retrieve_candidate_pool(
            query,
            top_k_articles=top_k_bm25_articles,
            top_k_laws=top_k_laws,
            strategy=candidate_strategy,
        )
        reranked = dense.retrieve(query, candidates, top_k=top_k_dense)
        if llm_reranker is not None:
            reranked = llm_reranker.rerank(query, reranked, min_keep=min(2, final_k), max_keep=final_k)
        else:
            reranked = reranked[:final_k]

        pred_list = [{"law_id": i["law_id"], "aid": i["aid"]} for i in reranked]
        metrics   = law_f1(pred_list, gt_list)
        if metrics is None:
            skipped += 1
            continue
        results.append(metrics)

        gt_set   = {(str(i["law_id"]), int(i["aid"])) for i in gt_list}
        pred_set = {(str(i["law_id"]), int(i["aid"])) for i in pred_list if i.get("aid") is not None}
        rows.append({
            "case_id":    case["case_id"],
            "hit":        int(bool(gt_set & pred_set)),
            "precision":  metrics["precision"],
            "recall":     metrics["recall"],
            "gt":         json.dumps(gt_list, ensure_ascii=False),
            "pred":       json.dumps(pred_list, ensure_ascii=False),
            "case_query": query,
        })
        if verbose:
            LOGGER.info("%s hit=%s P=%.2f R=%.2f", case["case_id"],
                        int(bool(gt_set & pred_set)), metrics["precision"], metrics["recall"])

    summarize_metrics(
        f"Pipeline strategy={candidate_strategy} bm25={top_k_bm25_articles} dense={top_k_dense} final={final_k}",
        results, skipped,
    )

    if error_analysis:
        out_path = resolve_path(error_analysis, PROJECT_ROOT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["case_id","hit","precision","recall","gt","pred","case_query"])
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Error analysis written to %s", out_path)


def evaluate_submission(test: str | Path, submission: str | Path) -> None:
    cases   = validate_case_records(load_json_file(resolve_path(test, PROJECT_ROOT), "ALQAC public test set"))
    outputs = load_json_file(resolve_path(submission, PROJECT_ROOT), "pipeline submission")
    by_id   = {str(item["case_id"]): item for item in outputs}
    gold: list[str] = []
    pred: list[str] = []
    for case in cases:
        label  = case.get("verdict_label")
        output = by_id.get(str(case["case_id"]))
        if label and output and output.get("prediction"):
            gold.append(str(label))
            pred.append(str(output["prediction"]))
    if not gold:
        LOGGER.warning("No overlapping labeled cases found.")
        return
    accuracy = sum(1 for y, p in zip(gold, pred) if y == p) / len(gold)
    LOGGER.info("Prediction Accuracy=%.4f MacroF1=%.4f Cases=%s",
                accuracy, macro_f1(gold, pred), len(gold))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ALQAC retrieval and predictions.")
    parser.add_argument("--mode", default="bm25", choices=["bm25", "pipeline"])
    parser.add_argument("--top-k",               type=int, default=20)
    parser.add_argument("--top-k-bm25-articles", type=int, default=200)
    parser.add_argument("--top-k-laws",          type=int, default=5)
    parser.add_argument("--top-k-dense",         type=int, default=30)
    parser.add_argument("--final-k",             type=int, default=15)
    parser.add_argument("--candidate-strategy",  default="hybrid", choices=["article","law","hybrid"])
    parser.add_argument("--rerank-mode",         default="mock",   choices=["mock","local"])
    parser.add_argument("--dense-model",         default="BAAI/bge-m3")
    parser.add_argument("--use-llm-reranker",    action="store_true")
    parser.add_argument("--error-analysis",      default=None)
    parser.add_argument("--verbose",             action="store_true")
    parser.add_argument("--corpus",     default="data/raw/corpus_law_pub.json")
    parser.add_argument("--test",       default="data/raw/ALQAC2026_public_test.json")
    parser.add_argument("--submission", default=None)
    parser.add_argument("--log-level",  default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    try:
        if args.mode == "pipeline":
            evaluate_pipeline_retrieval(
                corpus=args.corpus, test=args.test,
                top_k_bm25_articles=args.top_k_bm25_articles,
                top_k_laws=args.top_k_laws, top_k_dense=args.top_k_dense,
                final_k=args.final_k, candidate_strategy=args.candidate_strategy,
                rerank_mode=args.rerank_mode, dense_model=args.dense_model,
                use_llm_reranker=args.use_llm_reranker,
                error_analysis=args.error_analysis, verbose=args.verbose,
            )
        else:
            evaluate_retrieval(args.corpus, args.test, args.top_k, args.verbose)
        if args.submission:
            evaluate_submission(args.test, args.submission)
    except FriendlyFileError as exc:
        LOGGER.error("\n%s", exc)
        return 2
    except Exception as exc:
        LOGGER.exception("Evaluation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
