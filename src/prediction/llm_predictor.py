"""Outcome prediction — Qwen3-8B direct-label (mặc định) + parse-stats tracking.

Lịch sử thay đổi:
  Giai đoạn 1: fallback không hard-code, track parse_stats (json/regex/hard_fallback)
  Giai đoạn 2: thử extract-then-rule (% chấp nhận) — làm GIẢM accuracy (40%→32%
               trên leaderboard), nên TẮT MẶC ĐỊNH (use_extract_rule=False).
               Vẫn giữ code lại để có thể A/B test khi cần.

Mặc định hiện tại: Qwen3-8B tự chọn 1/4 nhãn trực tiếp (direct-label),
đây là cấu hình cho accuracy tốt nhất đã đo được (40% local).
"""

from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.text import normalize_text, stable_hash

LOGGER = logging.getLogger(__name__)

VALID_LABELS = ["A_WIN", "PARTIAL_A_WIN", "B_WIN", "PARTIAL_B_WIN"]
_LABEL_SET   = set(VALID_LABELS)

# ── Direct-label system prompt (bản chính, dùng mặc định) ────────────
SYSTEM_PROMPT = """/no_think
Bạn là hệ thống dự đoán kết quả tòa án dân sự Việt Nam.

## Các nhãn kết quả (BẮT BUỘC chọn đúng 1 trong 4):

| Nhãn          | Ý nghĩa |
|---------------|---------|
| A_WIN         | Tòa CHẤP NHẬN TOÀN BỘ yêu cầu của nguyên đơn (A). Bị đơn thua hoàn toàn. |
| PARTIAL_A_WIN | Tòa CHẤP NHẬN MỘT PHẦN yêu cầu của A (A thắng hơn 50%). Bị đơn thắng một phần nhỏ. |
| B_WIN         | Tòa BÁC TOÀN BỘ yêu cầu của A. Bị đơn (B) thắng hoàn toàn. Nguyên đơn không được gì. |
| PARTIAL_B_WIN | Tòa chỉ chấp nhận DƯỚI 50% yêu cầu của A. Bị đơn thắng phần lớn. |

## Hướng dẫn phân loại:
- Nếu bị đơn có lý, yêu cầu nguyên đơn không có căn cứ pháp luật → B_WIN hoặc PARTIAL_B_WIN
- Nếu số tiền được chấp nhận < 50% số tiền yêu cầu → PARTIAL_B_WIN
- Nếu số tiền được chấp nhận > 50% số tiền yêu cầu → PARTIAL_A_WIN
- Nếu toàn bộ yêu cầu được chấp nhận → A_WIN

## Ví dụ few-shot:

Ví dụ 1 → B_WIN:
Nguyên đơn kiện đòi bồi thường 100 triệu nhưng không chứng minh được lỗi của bị đơn.
Tòa bác yêu cầu vì thiếu căn cứ.
→ {"label":"B_WIN","confidence":0.85,"reasoning":"Bác toàn bộ vì thiếu căn cứ"}

Ví dụ 2 → PARTIAL_B_WIN:
Nguyên đơn đòi 100 triệu, tòa chỉ chấp nhận 30 triệu (30%).
→ {"label":"PARTIAL_B_WIN","confidence":0.80,"reasoning":"Chỉ 30% yêu cầu được chấp nhận"}

Ví dụ 3 → PARTIAL_A_WIN:
Nguyên đơn đòi 100 triệu, tòa chấp nhận 70 triệu (70%).
→ {"label":"PARTIAL_A_WIN","confidence":0.82,"reasoning":"70% yêu cầu được chấp nhận"}

Ví dụ 4 → A_WIN:
Nguyên đơn đòi bồi thường thiệt hại, tòa chấp nhận toàn bộ yêu cầu.
→ {"label":"A_WIN","confidence":0.88,"reasoning":"Toàn bộ yêu cầu được chấp nhận"}

## Output format — CHỈ trả về JSON này, không thêm bất kỳ text nào khác:
{"label":"<NHÃN>","confidence":<0.0-1.0>,"reasoning":"<giải thích ngắn bằng tiếng Việt>"}"""


def build_user_prompt(case_query: str, law_articles: list[dict[str, Any]]) -> str:
    if law_articles:
        blocks = []
        for idx, art in enumerate(law_articles, 1):
            content = str(art.get("content", "")).replace("\n", " ")[:600]
            blocks.append(
                f"[{idx}] {art.get('law_id')} aid={art.get('aid')}\n{content}"
            )
        evidence = "\n\n".join(blocks)
    else:
        evidence = "Không có điều luật nào được truy xuất."

    return (
        f"## Nội dung vụ án:\n{case_query}\n\n"
        f"## Điều luật liên quan:\n{evidence}\n\n"
        f"Dự đoán kết quả. Chỉ trả về JSON:"
    )


def _strip_think_tags(text: str) -> str:
    """Xóa <think>...</think> block mà Qwen3 sinh ra trước JSON."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def parse_prediction(
    text: str,
    default_label: str = "PARTIAL_A_WIN",
) -> tuple[str, float, str, str]:
    """Parse label, confidence, reasoning từ output model.

    Returns:
        (label, confidence, reasoning, parse_status)
        parse_status: "json" | "regex" | "hard_fallback"
    """
    raw = _strip_think_tags(text)

    try:
        start = raw.find("{")
        end   = raw.rfind("}")
        if start >= 0 and end > start:
            payload    = json.loads(raw[start:end + 1])
            label      = str(payload.get("label", "")).strip().upper().rstrip(".")
            confidence = float(payload.get("confidence", 0.5))
            reasoning  = str(payload.get("reasoning", ""))
            if label in _LABEL_SET:
                return label, max(0.0, min(1.0, confidence)), reasoning, "json"
    except Exception:
        pass

    upper = raw.upper()
    for label in VALID_LABELS:
        if re.search(rf"\b{label}\b", upper):
            LOGGER.warning("[LLMPredictor] JSON parse failed, regex fallback: %s", label)
            return label, 0.5, raw[:200], "regex"

    LOGGER.error("[LLMPredictor] Could not parse output: %r", raw[:300])
    return default_label, 0.34, raw[:200] or "Fallback.", "hard_fallback"


# ── Giai đoạn 2 (extract-then-rule) — giữ lại, TẮT mặc định ──────────

EXTRACT_SYSTEM_PROMPT = """/no_think
Bạn là hệ thống trích xuất kết quả bản án dân sự Việt Nam.
Ước tính % yêu cầu của nguyên đơn được tòa chấp nhận (0-100).
Chỉ trả về JSON: {"accepted_percentage": <0-100>, "reasoning": "<ngắn gọn>"}"""


def label_from_percentage(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    if pct >= 99.999:
        return "A_WIN"
    if pct <= 0.001:
        return "B_WIN"
    if pct > 50.0:
        return "PARTIAL_A_WIN"
    return "PARTIAL_B_WIN"


def parse_extraction(
    text: str,
    default_label: str = "PARTIAL_A_WIN",
) -> tuple[str, float, float, str, str]:
    raw = _strip_think_tags(text)
    try:
        start = raw.find("{")
        end   = raw.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(raw[start:end + 1])
            pct_raw = payload.get("accepted_percentage", None)
            if pct_raw is not None:
                pct = max(0.0, min(100.0, float(pct_raw)))
                reasoning = str(payload.get("reasoning", ""))
                label = label_from_percentage(pct)
                confidence = 0.5 + abs(pct - 50.0) / 100.0
                return label, round(min(0.95, confidence), 3), pct, reasoning, "json"
    except Exception:
        pass

    pct_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", raw)
    if pct_match:
        pct = max(0.0, min(100.0, float(pct_match.group(1))))
        label = label_from_percentage(pct)
        return label, 0.5, pct, raw[:200], "regex"

    default_pct = {"A_WIN": 100.0, "PARTIAL_A_WIN": 70.0,
                   "PARTIAL_B_WIN": 30.0, "B_WIN": 0.0}.get(default_label, 50.0)
    return default_label, 0.34, default_pct, raw[:200] or "Fallback.", "hard_fallback"


@dataclass(frozen=True)
class PredictResult:
    label:                str
    reasoning:             str
    raw_output:            str
    confidence:            float
    parse_status:          str = "mock"
    accepted_percentage:   float | None = None


class LLMPredictor:
    """Predict one of the four ALQAC outcome labels — Qwen3-8B direct-label (mặc định)."""

    def __init__(
        self,
        mode: str = "local",
        model_name: str = "Qwen/Qwen3-8B",
        cache_dir: str | Path | None = None,
        default_label: str = "PARTIAL_A_WIN",
        use_extract_rule: bool = False,   # ✅ TẮT mặc định — quay lại direct-label
    ):
        self.mode              = mode
        self.model_name        = model_name
        self.default_label     = default_label
        self.use_extract_rule  = use_extract_rule

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            here = Path(__file__).resolve()
            self.cache_dir = next(
                (p / "models" for p in here.parents if (p / "models").is_dir()),
                None,
            )

        self._model     = None
        self._tokenizer = None
        self._parse_stats: dict[str, int] = {"json": 0, "regex": 0, "hard_fallback": 0}

        if mode == "local":
            self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "LLMPredictor local mode yêu cầu: transformers, accelerate, torch, bitsandbytes."
            ) from exc

        cache = str(self.cache_dir) if self.cache_dir else None
        LOGGER.info("[LLMPredictor] Loading %s (extract_rule=%s) ...",
                    self.model_name, self.use_extract_rule)

        # Load tokenizer FIRST (on CPU)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=cache,
            padding_side="left",  # ✅ Set padding side
        )
        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        LOGGER.info("[LLMPredictor] Tokenizer loaded (pad_token=%s)", 
                    repr(self._tokenizer.pad_token))

        # Load model with BitsAndBytes
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=cache,
                attn_implementation="flash_attention_2",  # ✅ Use flash attention for stability
            )
        except Exception as e:
            LOGGER.warning("[LLMPredictor] Flash attention failed (%s), retrying without it", e)
            # Fallback without flash attention
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=cache,
            )
        
        self._model.eval()
        LOGGER.info("[LLMPredictor] Model loaded and ready")

    def predict(self, case_query: str, law_articles: list[dict[str, Any]]) -> PredictResult:
        if self.mode == "mock":
            return self._mock_predict(case_query, law_articles)
        if self.use_extract_rule:
            return self._local_predict_extract_rule(case_query, law_articles)
        return self._local_predict_direct(case_query, law_articles)

    def _mock_predict(self, case_query: str, law_articles: list[dict[str, Any]]) -> PredictResult:
        text = normalize_text(case_query, strip_accents=True)
        if any(t in text for t in ("mot phan", "chia doi", "50%", "1/2")):
            label, confidence = "PARTIAL_A_WIN", 0.62
        elif any(t in text for t in ("bac yeu cau", "khong chap nhan", "khong duoc chap nhan")):
            label, confidence = "B_WIN", 0.58
        elif any(t in text for t in ("toan bo", "chap nhan yeu cau")):
            label, confidence = "A_WIN", 0.56
        else:
            label      = VALID_LABELS[stable_hash(case_query) % len(VALID_LABELS)]
            confidence = 0.42
        evidence_ids = [f"{a.get('law_id')}:{a.get('aid')}" for a in law_articles]
        raw = json.dumps({"label": label, "confidence": confidence, "reasoning": "mock"})
        return PredictResult(label=label, confidence=confidence,
                             reasoning=f"[MOCK] evidence={evidence_ids}",
                             raw_output=raw, parse_status="mock")

    def _generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        assert self._model is not None and self._tokenizer is not None
        import torch

        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    def _local_predict_direct(
        self, case_query: str, law_articles: list[dict[str, Any]]
    ) -> PredictResult:
        """Direct-label — Qwen3-8B tự chọn 1/4 nhãn trực tiếp (mặc định, tốt nhất đã đo)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(case_query, law_articles)},
        ]
        raw = self._generate(messages, max_new_tokens=200)
        LOGGER.debug("[LLMPredictor] raw: %r", raw[:200])

        label, confidence, reasoning, status = parse_prediction(raw, default_label=self.default_label)
        self._parse_stats[status] = self._parse_stats.get(status, 0) + 1

        return PredictResult(
            label=label, confidence=confidence, reasoning=reasoning,
            raw_output=raw, parse_status=status,
        )

    def _local_predict_extract_rule(
        self, case_query: str, law_articles: list[dict[str, Any]]
    ) -> PredictResult:
        """Giai đoạn 2 (extract-then-rule) — giữ lại để A/B test, KHÔNG dùng mặc định."""
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(case_query, law_articles)},
        ]
        raw = self._generate(messages, max_new_tokens=150)
        label, confidence, pct, reasoning, status = parse_extraction(raw, default_label=self.default_label)
        self._parse_stats[status] = self._parse_stats.get(status, 0) + 1
        return PredictResult(
            label=label, confidence=confidence, reasoning=reasoning,
            raw_output=raw, parse_status=status, accepted_percentage=pct,
        )

    def report_parse_stats(self) -> None:
        """Gọi cuối pipeline để xem % case rơi vào fallback."""
        total = sum(self._parse_stats.values())
        if not total:
            return
        LOGGER.info("[LLMPredictor] Parse stats over %d cases (extract_rule=%s):",
                    total, self.use_extract_rule)
        for status, cnt in sorted(self._parse_stats.items()):
            pct = cnt / total * 100 if total else 0
            LOGGER.info("  %-14s %5d (%.1f%%)", status, cnt, pct)

        fallback_pct = self._parse_stats.get("hard_fallback", 0) / total * 100 if total else 0
        if fallback_pct > 10:
            LOGGER.warning(
                "[LLMPredictor] hard_fallback rate = %.1f%% (>10%%). "
                "Prompt hoặc output format của model đang có vấn đề.",
                fallback_pct,
            )
        else:
            LOGGER.info("[LLMPredictor] Parse success rate = %.1f%%", 100 - fallback_pct)