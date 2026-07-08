"""Legal evidence reranker: cross-encoder scoring with a legacy LLM yes/no fallback.

Primary path (new flow): a cross-encoder such as BAAI/bge-reranker-v2-m3 scores
each (query, article) pair with a continuous relevance score. Articles scoring
above `threshold` are kept; if none clear the threshold, all candidates are
kept instead of returning an empty evidence set (a reranker being unsure about
every candidate is not the same as every candidate being irrelevant).

Backward-compat path: if `model_name` does not look like a cross-encoder
(e.g. "Qwen/Qwen2.5-3B-Instruct"), the reranker automatically falls back to the
original CO/KHONG instruction-following LLM behavior, so existing configs that
pass an instruct model keep working unchanged.
"""

from __future__ import annotations

# Auto-add project root when this file is run directly.
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


import math
import re
from typing import Any

from src.utils.text import tokenize

YES_NO_PATTERN = re.compile(r"\b(co|yes)\b", re.IGNORECASE)
NO_PATTERN = re.compile(r"\b(khong|no)\b", re.IGNORECASE)

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

# Substrings that identify a model_name as a cross-encoder reranker rather
# than an instruction-following causal LM. Extend this list if you adopt a
# different reranker family.
_CROSS_ENCODER_HINTS: tuple[str, ...] = ("bge-reranker", "cross-encoder", "reranker-v2", "mxbai-rerank", "jina-reranker")

DEFAULT_RERANK_THRESHOLD = 0.5


def _looks_like_cross_encoder(model_name: str) -> bool:
    lower = model_name.lower()
    return any(hint in lower for hint in _CROSS_ENCODER_HINTS)


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _parse_yes_no(text: str) -> bool:
    """Parse deterministic CO/KHONG style responses (legacy LLM backend)."""

    head = text.strip().lower()[:80]
    normalized = head.replace("không", "khong").replace("có", "co")
    no_match = NO_PATTERN.search(normalized)
    yes_match = YES_NO_PATTERN.search(normalized)
    if no_match and (not yes_match or no_match.start() <= yes_match.start()):
        return False
    return bool(yes_match)


class LLMReranker:
    """Filter/rank dense candidates with a cross-encoder (default) or a legacy LLM."""

    def __init__(
        self,
        mode: str = "mock",
        model_name: str = "BAAI/bge-reranker-v2-m3",
        threshold: float = DEFAULT_RERANK_THRESHOLD,
    ):
        self.mode = mode
        self.model_name = model_name
        self.threshold = threshold
        self.backend = "cross_encoder" if _looks_like_cross_encoder(model_name) else "llm"
        self._model = None
        self._tokenizer = None
        if mode == "local":
            self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self.backend == "cross_encoder":
            self._load_cross_encoder()
        else:
            self._load_llm()

    def _load_cross_encoder(self) -> None:
        """
        Load CrossEncoder completely offline from local HuggingFace cache.
        """

        from src.utils.model_cache import (
            configure_hf_cache,
            get_model_path,
        )

        # Configure HF cache BEFORE importing sentence_transformers
        configure_hf_cache()

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Cross-encoder reranker mode requires sentence-transformers and torch. "
                "Install requirements.txt or run with --rerank-mode mock."
            ) from exc

        model_path = get_model_path(self.model_name)

        print(f"  [LLMReranker] Loading local CrossEncoder")
        print(f"      {model_path}")

        self._model = CrossEncoder(
            model_path,
            local_files_only=True,
        )

        print("  [LLMReranker] Ready")


    def _load_llm(self) -> None:
        """
        Load legacy LLM reranker completely offline.
        """

        from src.utils.model_cache import (
            configure_hf_cache,
            get_model_path,
        )

        configure_hf_cache()

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "LLM reranker local mode requires transformers, accelerate, "
                "torch, and bitsandbytes."
            ) from exc

        model_path = get_model_path(self.model_name)

        print(f"  [LLMReranker] Loading local LLM")
        print(f"      {model_path}")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

        self._model.eval()

        print("  [LLMReranker] Ready")

    # ------------------------------------------------------------------
    # Legacy LLM yes/no backend
    # ------------------------------------------------------------------

    def _is_relevant(self, query: str, article: dict[str, Any]) -> bool:
        assert self._model is not None and self._tokenizer is not None
        prompt = RELEVANCE_PROMPT.format(
            query=query[:900].strip(),
            law_id=article.get("law_id", ""),
            aid=article.get("aid", ""),
            article=str(article.get("content", ""))[:1000].strip(),
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        import torch

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _parse_yes_no(response)

    # ------------------------------------------------------------------
    # Cross-encoder scoring
    # ------------------------------------------------------------------

    def _mock_cross_encoder_scores(self, query: str, articles: list[dict[str, Any]]) -> list[float]:
        """Deterministic term-overlap stand-in for a real cross-encoder score."""

        query_terms = set(tokenize(query))
        scores = []
        for art in articles:
            article_terms = set(tokenize(art.get("content", "")))
            overlap = len(query_terms & article_terms)
            denom = max(len(query_terms), 1)
            scores.append(min(1.0, overlap / denom * 1.5))
        return scores

    def _cross_encoder_scores(self, query: str, articles: list[dict[str, Any]]) -> list[float]:
        assert self._model is not None
        pairs = [(query, str(art.get("content", ""))[:2000]) for art in articles]
        raw_scores = self._model.predict(pairs)
        return [_sigmoid(float(score)) for score in raw_scores]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        articles: list[dict[str, Any]],
        min_keep: int = 2,
        max_keep: int = 5,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return relevant articles, tagged with a continuous `rerank_score`.

        Cross-encoder backend: score every article, keep those with
        `rerank_score > threshold`; if none pass, fall back to keeping all of
        them (still capped at `max_keep`, best-scored first, so a wide-open
        fallback doesn't blow up the downstream LLM prompt). Also guarantees
        at least `min_keep` articles when candidates exist, so borderline
        cases still get some evidence.

        Legacy LLM backend: unchanged CO/KHONG behavior for backward
        compatibility with non-cross-encoder model names.
        """

        if not articles or max_keep <= 0:
            return []
        min_keep = max(0, min(min_keep, max_keep, len(articles)))
        active_threshold = self.threshold if threshold is None else threshold

        if self.backend == "llm":
            return self._rerank_llm(query, articles, min_keep, max_keep)
        return self._rerank_cross_encoder(query, articles, min_keep, max_keep, active_threshold)

    def _rerank_cross_encoder(
        self,
        query: str,
        articles: list[dict[str, Any]],
        min_keep: int,
        max_keep: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        scores = (
            self._mock_cross_encoder_scores(query, articles)
            if self.mode == "mock"
            else self._cross_encoder_scores(query, articles)
        )

        tagged = [dict(art, rerank_score=round(float(score), 6)) for art, score in zip(articles, scores)]
        ranked = sorted(tagged, key=lambda item: item["rerank_score"], reverse=True)

        passed = [item for item in ranked if item["rerank_score"] > threshold]
        if not passed:
            # Fallback: reranker didn't clearly endorse anything, keep the
            # best-scored candidates instead of returning empty evidence.
            passed = ranked
        if len(passed) < min_keep:
            passed = ranked[:min_keep]

        for item in passed:
            item["llm_relevant"] = item["rerank_score"] > threshold

        return passed[:max_keep]

    def _rerank_llm(
        self,
        query: str,
        articles: list[dict[str, Any]],
        min_keep: int,
        max_keep: int,
    ) -> list[dict[str, Any]]:
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
            passed = sorted(tagged, key=lambda item: item.get("dense_score", 0.0), reverse=True)[:min_keep]
            for item in passed:
                item["llm_relevant"] = True
        return passed[:max_keep]
