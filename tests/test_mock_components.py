"""Tests for deterministic mock components."""

from src.prediction.llm_predictor import LLMPredictor
from src.reranking.dense_reranker import DenseRetriever
from src.reranking.llm_reranker import _parse_yes_no


def test_dense_mock_prefers_overlap():
    dense = DenseRetriever(mode="mock")
    articles = [
        {"law_id": "A", "aid": 1, "content": "hop dong vay tien"},
        {"law_id": "B", "aid": 2, "content": "hon nhan gia dinh"},
    ]
    result = dense.retrieve("tranh chap hop dong vay tien", articles, top_k=1)
    assert result[0]["law_id"] == "A"


def test_yes_no_parser():
    assert _parse_yes_no("CO") is True
    assert _parse_yes_no("Không") is False


def test_mock_predictor_is_deterministic():
    predictor = LLMPredictor(mode="mock")
    first = predictor.predict("nguyen don yeu cau boi thuong", [])
    second = predictor.predict("nguyen don yeu cau boi thuong", [])
    assert first.label == second.label
    assert 0 <= first.confidence <= 1
