"""
Module : src/prediction/llm_predictor.py
Engine : LLM Predictor (SLM #2)  .  Stage 4/4 trong pipeline
Model  : Qwen/Qwen3-8B  (~6GB VRAM voi 4-bit)
Dung boi: scripts/run_pipeline.py

NHIEM VU
  Nhan case_query + 2~5 articles da loc -> du doan ket qua phan quyet
  bang Chain-of-Thought (CoT) 6 buoc.

INPUT
  LLMPredictor(mode, model_name)
    mode       - "mock" (test) | "local" (Qwen3-8B that)
    model_name - "Qwen/Qwen3-8B"

  .predict(case_query: str, law_articles: list[dict]) -> PredictResult
    case_query   - noi dung vu an
    law_articles - 2~5 articles tu LLMReranker: [{"law_id","aid","content",...},...]

OUTPUT
  PredictResult (dataclass):
    .label      - "A_WIN" | "B_WIN" | "PARTIAL_A_WIN" | "PARTIAL_B_WIN"
    .reasoning  - chuoi phan tich CoT cua model
    .raw_output - raw text tu model (de debug)

NHAN VA Y NGHIA
  A_WIN         - Toa chap nhan TOAN BO yeu cau nguyen don
  B_WIN         - Toa BAC TOAN BO yeu cau nguyen don
  PARTIAL_A_WIN - Chap nhan mot phan, phan duoc chap nhan > 50%
  PARTIAL_B_WIN - Chap nhan mot phan, phan duoc chap nhan <= 50%

CoT 6 BUOC (trong prompt)
  1. NGUYEN DON YEU CAU  - nguyen don muon gi?
  2. BI DON PHAN BAC     - ly le phia bi don
  3. DIEU LUAT AP DUNG   - dieu luat nao phu hop nhat
  4. LOI VA TRACH NHIEM  - ai co loi, ty le bao nhieu
  5. DU DOAN TOA AN      - toa se quyet the nao
  6. KET QUA             - label cuoi cung
"""

import re
import random
from dataclasses import dataclass

VALID_LABELS = ["A_WIN", "B_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN"]

SYSTEM_PROMPT = """\
Ban la chuyen gia phan tich phap ly dan su Viet Nam voi kinh nghiem doc va phan tich \
hang nghin ban an. Nhiem vu cua ban la du doan ket qua phan quyet cua toa an dua tren \
thong tin vu an va cac dieu luat lien quan.

BON KET QUA CO THE:
A_WIN        - Toa chap nhan TOAN BO yeu cau cua nguyen don.
B_WIN        - Toa BAC TOAN BO yeu cau cua nguyen don (nguyen don khong duoc gi).
PARTIAL_A_WIN - Toa chap nhan MOT PHAN, phan duoc chap nhan TREN 50% gia tri yeu cau.
PARTIAL_B_WIN - Toa chap nhan MOT PHAN, phan duoc chap nhan TU 50% TRO XUONG.

CACH PHAN BIET KHI CO LOI HON HOP:
- Nguyen don yeu cau 100, toa chap nhan 100               -> A_WIN
- Nguyen don yeu cau 100, toa chap nhan 0                 -> B_WIN
- Nguyen don yeu cau 100, toa chap nhan 70 (>50%)         -> PARTIAL_A_WIN
- Nguyen don yeu cau 100, toa chap nhan 50 hoac it hon    -> PARTIAL_B_WIN
- Ca hai ben deu co loi, chia doi boi thuong              -> PARTIAL_B_WIN (50%)

LUU Y:
- Tap trung vao YEU CAU CHINH duoc de cap trong case_query.
- Neu khong chac chan, uu tien PARTIAL_A_WIN hoac PARTIAL_B_WIN thay vi A_WIN/B_WIN tuyet doi.\
"""


def build_user_prompt(case_query: str, law_articles: list[dict]) -> str:
    if law_articles:
        articles_block = ""
        for i, art in enumerate(law_articles, 1):
            content_short = art["content"][:400].replace("\n", " ")
            suffix = "..." if len(art["content"]) > 400 else ""
            articles_block += (
                f"\n[Dieu {i}] {art['law_id']} - aid {art['aid']}\n"
                f"{content_short}{suffix}\n"
            )
    else:
        articles_block = "\n(Khong tim duoc dieu luat lien quan)\n"

    return (
        "VU AN:\n"
        f"{case_query}\n\n"
        "DIEU LUAT CO THE AP DUNG:\n"
        f"{articles_block}\n"
        "PHAN TICH TUNG BUOC (dung dung cu phap, khong them bot tieu de):\n\n"
        "NGUYEN DON YEU CAU: [tom tat ngan gon yeu cau chinh]\n"
        "BI DON PHAN BAC: [ly le phia bi don neu co, hoac 'khong ro']\n"
        "DIEU LUAT AP DUNG: [dieu luat nao phu hop nhat]\n"
        "LOI VA TRACH NHIEM: [ai co loi, muc do loi, ty le chiu trach nhiem uoc tinh]\n"
        "DU DOAN TOA AN: [toa se quyet dinh the nao va tai sao]\n"
        "KET QUA: [chi ghi dung MOT trong bon nhan: A_WIN / B_WIN / PARTIAL_A_WIN / PARTIAL_B_WIN]"
    )


def parse_label(text: str) -> str:
    for pattern in [
        r"KET\s*QUA\s*:\s*(A_WIN|B_WIN|PARTIAL_A_WIN|PARTIAL_B_WIN)",
        r"KET\s*QUA\s*:\s*\**(A_WIN|B_WIN|PARTIAL_A_WIN|PARTIAL_B_WIN)\**",
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
        self.mode       = mode
        self.model_name = model_name
        self._model     = None
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
        weights = {
            "PARTIAL_A_WIN": 0.38,
            "A_WIN":         0.32,
            "B_WIN":         0.20,
            "PARTIAL_B_WIN": 0.10,
        }
        label = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        return PredictResult(
            label=label,
            reasoning="[MOCK] Pipeline test.",
            raw_output=f"KET QUA: {label}",
        )

    def _local_predict(self, case_query: str, law_articles: list[dict]) -> PredictResult:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(case_query, law_articles)},
        ]
        text   = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw     = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        label   = parse_label(raw)
        return PredictResult(label=label, reasoning=raw, raw_output=raw)
