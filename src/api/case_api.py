"""
Module : src/api/case_api.py
Engine : Case Content API Client
Dùng bởi: scripts/run_pipeline.py (bước lấy case evidence, chạy trước BM25 law retrieval)

NHIỆM VỤ
  Gọi API của BTC để lấy nội dung vụ án theo từng chunk.
  BẮT BUỘC cho private test (chỉ có case_id + case_query, không có case_fact).

INPUT
  CaseAPIClient(token: str, base_url: str, mode: str = "real")
    token    — ALQAC_API_TOKEN trong file .env (bỏ qua nếu mode="mock")
    base_url — https://alqac2026-leaderboard.ngrok.app
    mode     — "real": gọi API thật.
               "mock": trả dữ liệu giả lập, không tốn rate-limit budget, không
               cần token — dùng để chạy/test toàn bộ pipeline offline.

  .search_case_segments(case_id: str, query: str) → dict
    case_id — VD: "0001"
    query   — từ khóa tìm kiếm trong nội dung vụ án
    → Mỗi lần gọi trả về 1 CHUNK văn bản (không phải toàn bộ vụ án).

  .retrieve_multi(case_id: str, case_query: str, extra_queries=None, max_queries=4) → list[dict]
    Gọi search_case_segments tối đa `max_queries` lần/case với các câu hỏi khác nhau
    (mặc định: case_query gốc + các câu hỏi phụ trong DEFAULT_SUB_QUERIES bao quát
    nguyên đơn / bị đơn / chứng cứ / phán quyết), gộp thành danh sách chunk evidence,
    loại trùng theo hash_id.

OUTPUT (search_case_segments)
  {
    "case_id": "0001",
    "result": {
      "hash_id": "hashsdjfvhlisduhfliudh",
      "text":    "Ngày 04/5/2018 nguyên đơn có nhận chuyển nhượng..."
    }
  }

OUTPUT (retrieve_multi)
  [
    {"chunk_id": "hashsdjfvhlisduhfliudh", "text": "...", "score": 1.0, "query": "..."},
    ...
  ]
  `score` là điểm heuristic theo thứ tự truy vấn (API của BTC không trả relevance
  score thật): case_query gốc được ưu tiên cao nhất, các câu hỏi phụ giảm dần.

RATE LIMIT
  1 request / 5 giây / team — đã tự xử lý trong _wait_rate_limit() (bỏ qua khi mode="mock").
  Gọi quá budget → bị penalty ở điểm Penalized Case Evidence Recall.
  => retrieve_multi() vì vậy mặc định giới hạn 4 query/case.
"""

from __future__ import annotations

# Auto-add project root when this file is run directly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

from src.utils.text import stable_hash

load_dotenv()

LOGGER = logging.getLogger(__name__)

# Default sub-queries used to cover a case from a few different angles when the
# caller does not pass its own `extra_queries`. Generic on purpose: these civil
# case aspects are usually present regardless of dispute type (contract, land,
# marriage, tort, ...), so they work as a reasonable default within the 4
# request/case rate-limit budget.
DEFAULT_SUB_QUERIES: tuple[str, ...] = (
    "yêu cầu khởi kiện của nguyên đơn",
    "trình bày và phản tố của bị đơn",
    "chứng cứ và tình tiết vụ án",
    "quyết định và nhận định của tòa án",
)


class CaseAPIClient:
    def __init__(self, token: str | None = None, base_url: str | None = None, mode: str = "real"):
        if mode not in {"real", "mock"}:
            raise ValueError("mode must be 'real' or 'mock'")
        self.mode = mode
        self.token = token or os.getenv("ALQAC_API_TOKEN")
        self.base_url = (base_url or os.getenv("ALQAC_API_BASE_URL",
                         "https://alqac2026-leaderboard.ngrok.app")).rstrip("/")
        self._last_call_time = 0.0
        self.rate_limit_sec = 5

    def _wait_rate_limit(self) -> None:
        if self.mode == "mock":
            return
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - elapsed)

    def search_case_segments(self, case_id: str, query: str) -> dict[str, Any]:
        """
        POST /v1/case_segments/search
        Trả về dict: {"case_id": ..., "result": {"hash_id": ..., "text": ...}}
        """

        if self.mode == "mock":
            hash_id = f"mock-{stable_hash(f'{case_id}:{query}')}"
            return {
                "case_id": case_id,
                "result": {
                    "hash_id": hash_id,
                    "text": f"[MOCK case segment] case_id={case_id} query='{query}'",
                },
            }

        self._wait_rate_limit()
        headers = { "X-API-Key": self.token}
        payload = {"case_id": case_id, "query": query}

        resp = requests.post(
            f"{self.base_url}/v1/case_segments/search",
            json=payload,
            headers=headers,
            timeout=30,
        )
        self._last_call_time = time.time()
        resp.raise_for_status()
        return resp.json()

    def retrieve_multi(
        self,
        case_id: str,
        case_query: str,
        extra_queries: list[str] | None = None,
        max_queries: int = 4,
    ) -> list[dict[str, Any]]:
        """Call search_case_segments up to `max_queries` times per case and merge results.

        The first query is always `case_query` itself; the remaining slots are
        filled from `extra_queries` (or DEFAULT_SUB_QUERIES if not provided).
        Duplicate chunks (same hash_id) are dropped. Errors on individual
        sub-queries are logged and skipped rather than aborting the whole case,
        since losing one chunk shouldn't lose the rest of the evidence.
        """

        queries = [case_query, *(extra_queries if extra_queries is not None else DEFAULT_SUB_QUERIES)]
        queries = queries[: max(0, max_queries)]

        chunks: list[dict[str, Any]] = []
        seen_hash: set[str] = set()

        for rank, query in enumerate(queries):
            try:
                response = self.search_case_segments(case_id, query)
            except Exception as exc:  # network/API errors must not abort the whole case
                LOGGER.warning("case_api query failed case_id=%s query=%r: %s", case_id, query, exc)
                continue

            result = response.get("result") or {}
            hash_id = result.get("hash_id")
            text = result.get("text")
            if not text or hash_id in seen_hash:
                continue
            seen_hash.add(hash_id)
            chunks.append(
                {
                    "chunk_id": hash_id,
                    "text": text,
                    # Heuristic order-based score (BTC endpoint has no native
                    # relevance score): case_query itself ranks highest, then
                    # sub-queries in the order they were asked.
                    "score": round(max(0.0, 1.0 - 0.15 * rank), 4),
                    "query": query,
                }
            )

        return chunks
