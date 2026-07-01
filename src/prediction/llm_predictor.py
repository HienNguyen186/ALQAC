import re
import random
from dataclasses import dataclass

VALID_LABELS = ["A_WIN", "B_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN"]

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích pháp lý dân sự Việt Nam với kinh nghiệm đọc và phân tích hàng nghìn bản án. Nhiệm vụ của bạn là dự đoán kết quả phán quyết của tòa án dựa trên thông tin vụ án và các điều luật liên quan.

BỐN KẾT QUẢ CÓ THỂ:
A_WIN         — Tòa chấp nhận TOÀN BỘ yêu cầu của nguyên đơn.
B_WIN         — Tòa BÁC TOÀN BỘ yêu cầu của nguyên đơn (nguyên đơn không được gì).
PARTIAL_A_WIN — Tòa chấp nhận MỘT PHẦN, phần được chấp nhận TRÊN 50% giá trị yêu cầu.
PARTIAL_B_WIN — Tòa chấp nhận MỘT PHẦN, phần được chấp nhận TỪ 50% TRỞ XUỐNG.

CÁCH PHÂN BIẮT KHI CÓ LỖI HỘN HỢP:
- Nguyên đơn yêu cầu 100, tòa chấp nhận 100               → A_WIN
- Nguyên đơn yêu cầu 100, tòa chấp nhận 0                 → B_WIN
- Nguyên đơn yêu cầu 100, tòa chấp nhận 70 (>50%)         → PARTIAL_A_WIN
- Nguyên đơn yêu cầu 100, tòa chấp nhận 50 hoặc ít hơn   → PARTIAL_B_WIN
- Cả hai bên đều có lỗi, chia đôi bồi thường              → PARTIAL_B_WIN (50%)

LƯU Ý:
- Tập trung vào YÊU CẦU CHÍNH được đề cập trong case_query.
- Nếu không chắc chắn, ưu tiên PARTIAL_A_WIN hoặc PARTIAL_B_WIN thay vì A_WIN/B_WIN tuyệt đối."""


def build_user_prompt(case_query: str, law_articles: list[dict]) -> str:
    if law_articles:
        articles_block = ""
        for i, art in enumerate(law_articles, 1):
            content_short = art["content"][:400].replace("\n", " ")
            suffix = "..." if len(art["content"]) > 400 else ""
            articles_block += (
                f"\n[Điều {i}] {art['law_id']} \u2013 aid {art['aid']}\n"
                f"{content_short}{suffix}\n"
            )
    else:
        articles_block = "\n(Không tìm được điều luật liên quan)\n"

    return (
        "VỤ ÁN:\n"
        f"{case_query}\n\n"
        "ĐIỀU LUẬT CÓ THỂ ÁP DỤNG:\n"
        f"{articles_block}\n"
        "PHÂN TÍCH TỮNG BƯỜC (dùng đúng cú pháp, không thêm bớt tiêu đề):\n\n"
        "NGUYÊN ĐƠN YÊu CẦU: [tóm tắt ngắn gọn yêu cầu chính của nguyên đơn]\n"
        "BỊ ĐƠN PHẢN BÁC: [lý lẽ phía bị đơn nếu có, hoặc 'không rõ']\n"
        "ĐIỀU LUẬT ÁP DỤNG: [điều luật nào phù hợp nhất với vụ việc này]\n"
        "LỖI VÀ TRÁCH NHIỆM: [ai có lỗi, mức độ lỗi, tỷ lệ chịu trách nhiệm ước tính]\n"
        "DỰ ĐOÁN TÒA ÁN: [tòa sẽ quyết định thế nào và tại sao]\n"
        "KẾT QUẢ: [chỉ ghi đúng MỘT trong bốn nhãn: A_WIN / B_WIN / PARTIAL_A_WIN / PARTIAL_B_WIN]"
    )


def parse_label(text: str) -> str:
    for pattern in [
        r"KẾT\s*QUẢ\s*:\s*(A_WIN|B_WIN|PARTIAL_A_WIN|PARTIAL_B_WIN)",
        r"KET\s*QUA\s*:\s*(A_WIN|B_WIN|PARTIAL_A_WIN|PARTIAL_B_WIN)",
        r"LABEL\s*:\s*(A_WIN|B_WIN|PARTIAL_A_WIN|PARTIAL_B_WIN)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines[-5:]):
        line_up = line.upper()
        for label in VALID_LABELS:
            if label in line_up:
                return label
    return "PARTIAL_A_WIN"


@dataclass
class PredictResult:
    label: str
    reasoning: str
    raw_output: str


class LLMPredictor:
    def __init__(self, mode: str = "mock", model_name: str = "Qwen/Qwen3-8B"):
        self.mode = mode
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        if mode == "local":
            self._load_model()

    def _load_model(self):
        print(f"[LLM] Loading {self.model_name} ...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
        )
        print("[LLM] Model loaded.")

    def predict(self, case_query: str, law_articles: list[dict]) -> PredictResult:
        if self.mode == "mock":
            return self._mock_predict()
        return self._local_predict(case_query, law_articles)

    def _mock_predict(self) -> PredictResult:
        weights = {"PARTIAL_A_WIN": 0.38, "A_WIN": 0.32, "B_WIN": 0.20, "PARTIAL_B_WIN": 0.10}
        label = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        return PredictResult(label=label, reasoning="[MOCK]", raw_output=f"KET QUA: {label}")

    def _local_predict(self, case_query: str, law_articles: list[dict]) -> PredictResult:
        import torch
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(case_query, law_articles)},
        ]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=512, temperature=0.1,
                do_sample=True, repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return PredictResult(label=parse_label(raw), reasoning=raw, raw_output=raw)
