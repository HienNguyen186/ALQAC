# ALQAC 2026 — Context (cập nhật 30/06/2026)

> Sinh viên Data Science, TPHCM. Claude là mentor/gia sư.
> File tổng hợp toàn bộ thông tin từ email BTC + website chính thức.
> **Luôn đọc file này khi bắt đầu buổi làm việc mới.**

---

## 1. Thông tin cuộc thi

| | |
|---|---|
| **Tên** | Automated Legal Question Answering Competition (ALQAC 2026) |
| **Sự kiện** | Associated with KSE 2026, Kanazawa, Nhật Bản, 11–14/11/2026 |
| **Tổ chức** | Nguyen Lab - JAIST |
| **Homepage** | https://sites.google.com/view/ALQAC2026 |
| **Leaderboard** | https://alqac2026-leaderboard.ngrok.app/ |
| **API docs** | https://alqac2026-leaderboard.ngrok.app/api-docs |
| **Secret token** | `alqac_ulXGWBIP32LwpxKFFVCGd0Cf53naKG1D` |

---

## 2. Bài toán

### Input — Public test (50 cases, đã có label)
```json
{
  "case_id": "case_4101",
  "case_query": "Chị Lê Thị T khởi kiện... Agent dự đoán ai thắng?"
}
```

> ⚠️ **Trên thực tế, public test còn có thêm `case_fact`, `verdict_label`, v.v.**
> **Nhưng private test sẽ CHỈ có `case_id` + `case_query` — không có gì khác.**

### Input — Private test (format thực tế)
```json
{
  "case_id": "0001",
  "case_query": "Ông Nguyễn Khắc Vũ H1 (nguyên đơn) và Chu Quang Nguyễn H2 (bị đơn) tranh chấp..."
}
```

### Output — submission.json
```json
[
  {
    "case_id": "0001",
    "prediction": "A_WIN",
    "law_evidence": [
      {"law_id": "47/2010/QH12", "aid": 270},
      {"law_id": "91/2015/QH13", "aid": 357}
    ]
  }
]
```

---

## 3. Nhãn dự đoán — 4 nhãn (⚠️ đã cập nhật từ email 22/6)

| Nhãn | Ý nghĩa chính xác theo website |
|---|---|
| `A_WIN` | Tòa chấp nhận **toàn bộ** yêu cầu nguyên đơn |
| `B_WIN` | Tòa **bác toàn bộ** yêu cầu nguyên đơn |
| `PARTIAL_A_WIN` | Tòa chấp nhận một phần, phần được chấp nhận **> 50%** |
| `PARTIAL_B_WIN` | Tòa chấp nhận một phần, phần được chấp nhận **≤ 50%** |

> ⚠️ **Sửa lỗi so với context cũ:** `PARTIAL_B_WIN` KHÔNG phải "bị đơn thắng một phần lập luận".
> Cả PARTIAL_A và PARTIAL_B đều là nguyên đơn thắng một phần — chỉ khác ở tỷ lệ (>50% vs ≤50%).

**Phân bố nhãn trong public test (50 cases):**
- PARTIAL_A_WIN: 19
- A_WIN: 16
- B_WIN: 10
- PARTIAL_B_WIN: 5

---

## 4. Nguồn dữ liệu

### 4a. Law Corpus (đã có sẵn)
- File: `data/raw/corpus_law_pub.json`
- **18 văn bản luật, 3.352 articles**
- `aid` là global row ID, **KHÔNG phải số điều** → `aid = aid_min_của_luật + số_điều - 1`
- Đã build `LAW_AID_OFFSET` map trong `src/retrieval/law_name_map.py`

### 4b. Case Content API (⚠️ BẮT BUỘC cho private test)
```
Endpoint: POST /v1/case_segments/search
Auth:     Bearer alqac_ulXGWBIP32LwpxKFFVCGd0Cf53naKG1D
Rate:     1 request / 5 giây / team

Request:
{
  "case_id": "001",
  "query": "hợp đồng chuyển nhượng quyền sử dụng đất thửa 396"
}

Response:
{
  "case_id": "0001",
  "result": {
    "hash_id": "hashsdjfvhlisduhfliudh",
    "text": "Ngày 04/5/2018 nguyên đơn có nhận chuyển nhượng..."
  }
}
```

**Quan trọng về Case Content API:**
- Nội dung vụ án được **cắt nhỏ thành chunks** — mỗi lần gọi chỉ trả về **1 chunk**
- Private test chỉ có `case_query` → **bắt buộc phải gọi API** để có thông tin vụ án
- API budget **phụ thuộc kích thước case**: case lớn → được gọi nhiều hơn
- Gọi quá nhiều sẽ bị penalty ở điểm `Penalized Case Evidence Recall`

---

## 5. Công thức tính điểm

**Final Score = f(3 thành phần):**

| Thành phần | Ý nghĩa |
|---|---|
| **Outcome Accuracy** | Dự đoán đúng 1 trong 4 nhãn |
| **Penalized Case Evidence Recall** | Truy xuất đúng evidence − penalty nếu gọi API quá nhiều |
| **Micro Law Evidence F1** | Precision + Recall trên điều luật trích dẫn |

**API Efficiency Penalty:** budget API phụ thuộc kích thước case. Gọi ít và đúng > gọi nhiều.

---

## 6. Ràng buộc kỹ thuật

| Ràng buộc | Chi tiết |
|---|---|
| ✅ Model | Chỉ open-weight < 10B parameters |
| ❌ Cấm | ChatGPT, Claude, Gemini, mọi proprietary model |
| ✅ Cho phép | Query online legal databases (không phải pre-labeled QA datasets) |
| ⏱️ Rate limit | 1 request / 5 giây / team |
| 📤 Submit | Tối đa 3 lần/ngày |
| 📋 Public test | 50 cases (có label, dùng để test pipeline) |
| 🔒 Private test | Chỉ `case_id` + `case_query` — không có thông tin nào khác |

---

## 7. Data đã có & trạng thái

| File | Path | Trạng thái |
|---|---|---|
| Public test | `data/raw/ALQAC2026_public_test.json` | ✅ Có sẵn (50 cases, có label) |
| Law corpus | `data/raw/corpus_law_pub.json` | ✅ Có sẵn (18 luật, 3352 articles) |

> ⚠️ **Lưu ý:** Email 22/6 thông báo đã fix lỗi duplicate `case_id` trong public test.
> Nếu file được download trước 22/6, cần **re-download** từ Google Drive BTC.

---

## 8. Pipeline đề xuất

```
case_query
    │
    ├─────────────────────────────────┐
    │                                 │
    ▼                                 ▼
[LAW RETRIEVAL]                [CASE EVIDENCE RETRIEVAL]
BM25 toàn bộ corpus            Gọi Case Content API
→ Top-20 articles              (multi-round, trong budget)
→ Dense rerank (bge-m3)        → Thu thập evidence chunks
→ Top 3~5 articles final       → Rank theo relevance
    │                                 │
    └──────────────┬──────────────────┘
                   │
                   ▼
         [LLM REASONING + PREDICTION]
         Input: case_query
              + top case evidence chunks
              + top law articles
         → Chain-of-Thought prompt
         → Output: A_WIN / B_WIN / PARTIAL_A_WIN / PARTIAL_B_WIN
```

---

## 9. Model gợi ý (< 10B params)

| Model | Params | Dùng cho |
|---|---|---|
| **Qwen2.5-7B-Instruct** | 7B | LLM reasoning/prediction (tiếng Việt tốt nhất) |
| **Gemma 2 9B Instruct** | 9B | LLM reasoning |
| **Llama 3.1 8B Instruct** | 8B | LLM reasoning |
| **BAAI/bge-m3** | — | Dense retrieval embedding |
| **multilingual-e5-large** | — | Dense retrieval (nhanh hơn) |

---

## 10. Kết quả đã đo (public test, local eval)

| Method | top_k | Precision | Recall | **Law F1** |
|---|---|---|---|---|
| BM25 (case_query) | 5 | 0.044 | 0.015 | 0.022 |
| BM25 (case_query) | 10 | 0.032 | 0.028 | 0.028 |
| BM25 (case_query) | 20 | 0.032 | 0.076 | **0.040** |

**Nhận xét:** F1=0.04 thấp vì BM25 chỉ đếm từ trùng, không hiểu ngữ nghĩa pháp lý.
100% ground truth articles đều có trong corpus → vấn đề là **retrieval quality**, không phải thiếu data.

---

## 11. Lộ trình thực hiện

| Bước | Việc | Trạng thái |
|---|---|---|
| ① | Gọi thử Case Content API | ⬜ Chưa làm |
| ② | BM25 law retrieval + local eval | ✅ Done — F1=0.040 |
| ③ | LLM zero-shot predict + submit đầu tiên | ⬜ Tiếp theo |
| ④ | Dense retrieval (bge-m3) thay BM25 | ⬜ |
| ⑤ | Tích hợp Case Content API vào pipeline | ⬜ |
| ⑥ | Tối ưu số lần gọi API | ⬜ |

---

## 12. Cấu trúc project

```
MAIN/
├── .env.example              ← token template
├── .env                      ← token thật (KHÔNG push lên git)
├── .gitignore
├── requirements.txt
├── configs/config.yaml       ← tất cả hyperparams
├── data/
│   ├── raw/                  ← bị gitignore
│   └── processed/            ← bị gitignore
├── src/
│   ├── api/case_api.py       ← client gọi Case Content API
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   └── law_name_map.py   ← mapping tên luật → law_id + aid offset
│   ├── reranking/            ← slot cho cross-encoder/dense
│   └── prediction/           ← slot cho LLM
├── scripts/
│   └── evaluate.py           ← local eval trên public test
├── notebooks/
└── outputs/submissions/
```

---

## 13. Ghi chú kỹ thuật quan trọng

- **`aid` trong corpus = global row ID**, KHÔNG phải số điều. Công thức: `aid = aid_min + số_điều - 1`
- **Law name mapping** trong `src/retrieval/law_name_map.py` — ground truth dùng tên tiếng Việt, corpus dùng mã luật
- **Rate limit API:** 1 req/5s → 50 cases × 5 lần = 250 giây chờ → cần async
- **Private test format:** chỉ `case_id` + `case_query` — không hardcode bất kỳ thứ gì từ public test
- **Submit limit:** 3 lần/ngày → test kỹ local trước khi submit
- **PARTIAL labels** phân biệt bởi tỷ lệ >50% vs ≤50%, không phải "bị đơn thắng một phần"

---

*Cập nhật lần cuối: 30/06/2026 — từ email BTC (19/6, 22/6, 29/6) + website chính thức*
