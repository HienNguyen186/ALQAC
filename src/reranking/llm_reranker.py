"""LLM-based legal evidence reranker."""

from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import logging
import re
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

YES_NO_PATTERN = re.compile(r"\b(co|yes)\b",    re.IGNORECASE)
NO_PATTERN     = re.compile(r"\b(khong|no)\b",  re.IGNORECASE)

RELEVANCE_PROMPT = """You are a Vietnamese civil-law evidence filter.
Decide whether the legal article directly supports resolving the case.

Rules:
- Answer only one token: CO or KHONG.
- CO only if the article states a legal basis for the dispute, liability, procedure, fees, or remedy.
- KHONG for generic or unrelated articles.

CASE:
{query}

LEGAL ARTICLE ({law_id}, aid={aid}):
{article}

Answer:"""


def _parse_yes_no(text: str) -> bool:
    head       = text.strip().lower()[:80]
    normalized = head.replace("không", "khong").replace("có", "co")
    no_match   = NO_PATTERN.search(normalized)
    yes_match  = YES_NO_PATTERN.search(normalized)
    if no_match and (not yes_match or no_match.start() <= yes_match.start()):
        return False
    return bool(yes_match)


class LLMReranker:
    """Filter dense candidates with a small instruction-tuned LLM."""

    def __init__(
        self,
        mode: str = "local",
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        cache_dir: str | Path | None = None,
    ):
        self.mode       = mode
        self.model_name = model_name
        self.cache_dir  = Path(cache_dir) if cache_dir else self._default_cache()
        self._model     = None
        self._tokenizer = None
        if mode == "local":
            self._load_model()

    @staticmethod
    def _default_cache() -> Path | None:
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "models"
            if candidate.is_dir():
                return candidate
        return None

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "LLMReranker local mode yêu cầu: transformers, accelerate, torch, bitsandbytes.\n"
                "Cài đặt: pip install -r requirements.txt"
            ) from exc

        cache = str(self.cache_dir) if self.cache_dir else None
        LOGGER.info("[LLMReranker] Loading %s (cache=%s) ...", self.model_name, cache)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=cache,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache,
        )
        self._model.eval()
        LOGGER.info("[LLMReranker] Ready")

    def _is_relevant(self, query: str, article: dict[str, Any]) -> bool:
        assert self._model is not None and self._tokenizer is not None
        import torch

        prompt   = RELEVANCE_PROMPT.format(
            query=query[:500].strip(),      # Giới hạn query (giảm từ 900)
            law_id=article.get("law_id", ""),
            aid=article.get("aid", ""),
            article=str(article.get("content", ""))[:600].strip(),  # Giảm từ 1000
        )
        messages = [{"role": "user", "content": prompt}]
        
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback nếu apply_chat_template fail
            text = prompt

        # Tokenize với max_length — sẽ truncate thay vì crash
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,           # Giới hạn strictly
            truncation=True,
            padding="max_length",
        ).to(self._model.device)

        with torch.no_grad():
            try:
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    use_cache=False,  # Disable KV cache để tránh OOM
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            except RuntimeError as e:
                # Nếu vẫn crash → fallback: coi như relevant
                LOGGER.warning("[LLMReranker] Generate failed: %s, fallback to True", e)
                return True
            finally:
                # Clear GPU cache sau generate
                try:
                    torch.cuda.empty_cache()
                except:
                    pass

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        response   = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _parse_yes_no(response)

    def rerank(
        self,
        query: str,
        articles: list[dict[str, Any]],
        min_keep: int = 3,
        max_keep: int = 15,
    ) -> list[dict[str, Any]]:
        """Return between min_keep and max_keep relevant articles."""
        if not articles or max_keep <= 0:
            return []
        min_keep = max(0, min(min_keep, max_keep, len(articles)))

        if self.mode == "mock":
            kept = []
            for rank, art in enumerate(articles[:max_keep], 1):
                entry = dict(art)
                entry["llm_relevant"] = rank <= max(min_keep, 1)
                kept.append(entry)
            return [item for item in kept if item["llm_relevant"]]

        tagged = []
        for art in articles:
            entry = dict(art)
            entry["llm_relevant"] = self._is_relevant(query, art)
            tagged.append(entry)

        passed = [item for item in tagged if item["llm_relevant"]]
        if len(passed) < min_keep:
            passed = sorted(tagged, key=lambda x: x.get("dense_score", 0.0), reverse=True)[:min_keep]
            for item in passed:
                item["llm_relevant"] = True
        return passed[:max_keep]