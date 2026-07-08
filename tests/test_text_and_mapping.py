"""Tests for text normalization and legal provision parsing."""

from src.retrieval.law_name_map import law_name_to_id, parse_article_number, parse_law_provisions
from src.utils.text import tokenize


def test_tokenize_normalizes_vietnamese_accents():
    tokens = tokenize("Bộ luật Dân sự năm 2015, Điều 584")
    assert "bo" in tokens
    assert "luat" in tokens
    assert "dan" in tokens
    assert "584" in tokens


def test_law_name_parser_accepts_common_names():
    assert law_name_to_id("Bộ luật Dân sự năm 2015") == "91/2015/QH13"
    assert parse_article_number("Khoản 1 Điều 584") == 584


def test_parse_law_provisions_deduplicates():
    raw = "Bộ luật Dân sự năm 2015 | Điều 584\nBộ luật Dân sự 2015 | Khoản 1 Điều 584"
    parsed = parse_law_provisions(raw)
    assert parsed == [{"law_id": "91/2015/QH13", "aid": 53354}]
