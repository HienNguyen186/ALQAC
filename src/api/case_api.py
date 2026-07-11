"""
Module : src/api/case_api.py
Engine : Case Content API Client
Dùng bởi: chưa tích hợp vào pipeline (Step ⑤ — sau khi có submission đầu tiên)

NHIỆM VỤ
  Gọi API của BTC để lấy nội dung vụ án theo từng chunk.
  BẮT BUỘC cho private test (chỉ có case_id + case_query, không có case_fact).

INPUT
  CaseAPIClient(token: str, base_url: str)
    token    — ALQAC_API_TOKEN trong file .env
    base_url — https://alqac2026-leaderboard.ngrok.app

  .search_case_segments(case_id: str, query: str) → dict
    case_id — VD: "0001"
    query   — từ khóa tìm kiếm trong nội dung vụ án

OUTPUT
  {
    "case_id": "0001",
    "result": {
      "hash_id": "hashsdjfvhlisduhfliudh",
      "text":    "Ngày 04/5/2018 nguyên đơn có nhận chuyển nhượng..."
    }
  }
  → Mỗi lần gọi trả về 1 CHUNK văn bản (không phải toàn bộ vụ án).
  → Gọi nhiều lần với query khác nhau để thu thập đủ thông tin.

RATE LIMIT
  1 request / 5 giây / team — đã tự xử lý trong _wait_rate_limit().
  Gọi quá budget → bị penalty ở điểm Penalized Case Evidence Recall.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()


class CaseAPIClient:
    def __init__(self, token: str = None, base_url: str = None):
        self.token = token or os.getenv("ALQAC_API_TOKEN")
        self.base_url = (base_url or os.getenv("ALQAC_API_BASE_URL",
                         "https://alqac2026-leaderboard.ngrok.app")).rstrip("/")
        self._last_call_time = 0
        self.rate_limit_sec = 5

    def _wait_rate_limit(self):
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - elapsed)

    def search_case_segments(self, case_id: str, query: str) -> dict:
        """
        POST /v1/case_segments/search
        Trả về dict: {"case_id": ..., "result": {"hash_id": ..., "text": ...}}
        """
        self._wait_rate_limit()
        headers = {"Authorization": f"Bearer {self.token}"}
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