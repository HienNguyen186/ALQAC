"""Outcome prediction with Qwen or deterministic mock mode."""

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

# ── System prompt được viết lại để:
# 1. Tắt Qwen3 thinking mode (/no_think)
# 2. Mô tả rõ B_WIN và PARTIAL_B_WIN với ví dụ cụ thể
# 3. Dùng Few-shot để model thấy tất cả 4 nhãn
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


def parse_prediction(text: str) -> tuple[str, float, str]:
    """Parse label, confidence, reasoning từ output model."""
    raw = _strip_think_tags(text)

    # Thử parse JSON
    try:
        start = raw.find("{")
        end   = raw.rfind("}")
        if start >= 0 and end > start:
            payload    = json.loads(raw[start:end + 1])
            label      = str(payload.get("label", "")).strip().upper().rstrip(".")
            confidence = float(payload.get("confidence", 0.5))
            reasoning  = str(payload.get("reasoning", ""))
            if label in _LABEL_SET:
                return label, max(0.0, min(1.0, confidence)), reasoning
    except Exception:
        pass

    # Regex fallback
    upper = raw.upper()
    for label in VALID_LABELS:
        if re.search(rf"\b{label}\b", upper):
            LOGGER.warning("[LLMPredictor] JSON parse failed, regex fallback: %s", label)
            return label, 0.5, raw[:200]

    LOGGER.error("[LLMPredictor] Could not parse output: %r", raw[:300])
    return "PARTIAL_A_WIN", 0.34, raw[:200] or "Fallback."


@dataclass(frozen=True)
class PredictResult:
    label:      str
    reasoning:  str
    raw_output: str
    confidence: float


class LLMPredictor:
    """Predict one of the four ALQAC outcome labels."""

    def __init__(
        self,
        mode: str = "local",
        model_name: str = "Qwen/Qwen3-8B",
        cache_dir: str | Path | None = None,
    ):
        self.mode       = mode
        self.model_name = model_name

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
        LOGGER.info("[LLMPredictor] Loading %s ...", self.model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=cache,
        )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache,
        )
        self._model.eval()
        LOGGER.info("[LLMPredictor] Ready")

    def predict(self, case_query: str, law_articles: list[dict[str, Any]]) -> PredictResult:
        if self.mode == "mock":
            return self._mock_predict(case_query, law_articles)
        return self._local_predict(case_query, law_articles)

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
                             reasoning=f"[MOCK] evidence={evidence_ids}", raw_output=raw)

    def _local_predict(self, case_query: str, law_articles: list[dict[str, Any]]) -> PredictResult:
        assert self._model is not None and self._tokenizer is not None
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(case_query, law_articles)},
        ]

        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,   # Tắt thinking mode Qwen3
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw     = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        LOGGER.debug("[LLMPredictor] raw: %r", raw[:200])
        label, confidence, reasoning = parse_prediction(raw)
        return PredictResult(label=label, confidence=confidence, reasoning=reasoning, raw_output=raw)