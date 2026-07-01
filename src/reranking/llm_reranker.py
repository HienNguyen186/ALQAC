"""
Module : src/reranking/llm_reranker.py
Engine : LLM Relevance Filter (SLM #1)  .  Stage 3/4 trong pipeline
Model  : Qwen/Qwen2.5-3B-Instruct  (~2GB VRAM voi 4-bit)
Dung boi: scripts/run_pipeline.py

NHIEM VU
  Voi MOI article trong top-20 (tu BGE-M3), hoi SLM:
    "Dieu luat nay co truc tiep lien quan den vu an khong?"
  Chi giu lai cac article duoc tra loi "co".

INPUT
  LLMReranker(mode, model_name)
    mode       - "mock" (test) | "local" (Qwen2.5-3B that)
    model_name - "Qwen/Qwen2.5-3B-Instruct"

  .rerank(query, articles, min_keep=2, max_keep=5)
    query     - noi dung vu an
    articles  - top-20 tu DenseRetriever: [{"law_id","aid","content","dense_score","rank"},...]
    min_keep  - giu toi thieu N articles (fallback)
    max_keep  - giu toi da N articles -> vao Qwen3-8B

OUTPUT
  list[dict] - 2~5 articles, them field "llm_relevant": True
  [
    {"law_id": "91/2015/QH13", "aid": 53354, "content": "...",
     "dense_score": 0.872, "rank": 1, "llm_relevant": True},
    ...
  ]

FALLBACK
  Neu < min_keep articles duoc gan "co" -> lay top min_keep theo dense_score.
"""

import re

RELEVANCE_PROMPT = """\
Ban la chuyen gia phap ly Viet Nam. Nhiem vu: danh gia xem dieu luat co lien quan \
truc tiep den vu an khong.

VU AN:
{query}

DIEU LUAT ({law_id}):
{article}

Cau hoi: Dieu luat tren co truc tiep dieu chinh hoac lam can cu phap ly \
de giai quyet vu an khong?

Tra loi CHI mot tu: co / khong"""


def _parse_yes_no(text: str) -> bool:
    head = text.strip().lower()[:30]
    if head.startswith("khong"):
        return False
    if re.search(r"^co", head):
        return True
    idx_co    = text.lower().find("co")
    idx_khong = text.lower().find("khong")
    if idx_co >= 0 and (idx_khong < 0 or idx_co < idx_khong):
        return True
    return False


class LLMReranker:
    def __init__(self, mode: str = "mock", model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.mode       = mode
        self.model_name = model_name
        self._model     = None
        self._tokenizer = None

        if mode == "local":
            self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        print(f"  [LLMReranker] Loading {self.model_name} ...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb,
            device_map="auto",
        )
        self._model.eval()
        print(f"  [LLMReranker] Ready")

    def _is_relevant(self, query: str, article: dict) -> bool:
        prompt = RELEVANCE_PROMPT.format(
            query   = query[:600].strip(),
            law_id  = article.get("law_id", ""),
            article = article.get("content", "")[:800].strip(),
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        import torch
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        response   = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _parse_yes_no(response)

    def rerank(self, query: str, articles: list[dict],
               min_keep: int = 2, max_keep: int = 5) -> list[dict]:
        if self.mode == "mock":
            result = []
            for i, art in enumerate(articles[:max_keep]):
                entry = dict(art)
                entry["llm_relevant"] = (i < min_keep)
                result.append(entry)
            return [a for a in result if a["llm_relevant"]]

        tagged = []
        for art in articles:
            entry = dict(art)
            entry["llm_relevant"] = self._is_relevant(query, art)
            tagged.append(entry)

        passed = [a for a in tagged if a["llm_relevant"]]

        if len(passed) < min_keep:
            passed = sorted(
                tagged,
                key=lambda x: x.get("dense_score", x.get("score", 0)),
                reverse=True,
            )[:min_keep]
            for a in passed:
                a["llm_relevant"] = True

        return passed[:max_keep]
