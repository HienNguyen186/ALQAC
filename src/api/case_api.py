"""
Module : src/api/case_api.py
Engine : Case Content API Client

NHIỆM VỤ
  Gọi API của BTC để lấy nội dung vụ án theo từng chunk.
  BẮT BUỘC cho private test (chỉ có case_id + case_query, không có case_fact).

INPUT
  CaseAPIClient(token: str, base_url: str)
    token    — ALQAC_API_TOKEN trong file .env  (gửi qua header X-Api-Key)
    base_url — https://alqac-api.ngrok.pro

  .search_case_segments(case_id: str, query: str) → dict
    case_id — VD: "case_1007_0037"
    query   — từ khóa tìm kiếm trong nội dung vụ án

OUTPUT (theo docs BTC)
  {
    "results": [
      {
        "score":    0.886,
        "text":     "Ngày 04/5/2018 nguyên đơn có nhận chuyển nhượng...",
        "chunk_id": "case_1007_0037_chunk_2"
      }
    ]
  }
  → Mỗi lần gọi trả về 1 segment (top-1).
  → Gọi nhiều lần với query khác nhau để thu thập đủ chunk_id.
  → chunk_id dùng để điền vào case_evidence trong submission.

RATE LIMIT
  1 request / 5 giây / team — đã tự xử lý trong _wait_rate_limit().
  Gọi quá budget → bị penalty ở điểm Penalized Case Evidence Recall.

LƯU Ý AUTH
  Header phải là: X-Api-Key: <token>   (KHÔNG dùng Authorization: Bearer)
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()


class CaseAPIClient:
    def __init__(self, token: str = None, base_url: str = None):
        self.token = token or os.getenv("ALQAC_API_TOKEN")
        self.base_url = (
            base_url
            or os.getenv("ALQAC_API_BASE_URL", "https://alqac-api.ngrok.pro")
        ).rstrip("/")
        self._last_call_time: float = 0
        self.rate_limit_sec: int = 5

    # ── Rate limiting ────────────────────────────────────────────────────

    def _wait_rate_limit(self):
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - elapsed)

    # ── Public API ───────────────────────────────────────────────────────

    def search_case_segments(self, case_id: str, query: str) -> dict:
        """
        POST /retrieve
        Header : X-Api-Key: <token>
        Body   : {"query": str, "case_id": str}

        Trả về dict nguyên gốc từ server:
          {"results": [{"score": float, "text": str, "chunk_id": str}]}

        Raises:
          requests.HTTPError nếu server trả lỗi (401, 422, 429, 503…)
        """
        self._wait_rate_limit()
        try:
            resp = requests.post(
                f"{self.base_url}/retrieve",
                json={"query": query, "case_id": case_id},
                headers={
                    "X-Api-Key": self.token,
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            # Cập nhật thời gian dù thành công hay thất bại
            self._last_call_time = time.time()

    def get_chunk_id(self, case_id: str, query: str) -> str | None:
        """
        Helper: trả thẳng chunk_id của segment top-1, hoặc None nếu lỗi/rỗng.
        Dùng trong _fetch_case_evidence() thay cho .get("result", {}).get("hash_id").
        """
        try:
            data = self.search_case_segments(case_id=case_id, query=query)
            results = data.get("results", [])
            if results:
                return results[0].get("chunk_id")
        except Exception:
            raise
        return None
