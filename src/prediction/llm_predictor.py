"""Outcome prediction with Qwen3-8B (CoT) or deterministic mock mode."""

from __future__ import annotations

# Auto-add project root when this file is run directly.
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


import json
import re
from dataclasses import dataclass
from typing import Any

from src.utils.text import normalize_text, stable_hash

VALID_LABELS = ["A_WIN", "PARTIAL_A_WIN", "B_WIN", "PARTIAL_B_WIN"]
_LABEL_SET = set(VALID_LABELS)

SYSTEM_PROMPT = """You are a Vietnamese civil legal outcome predictor.
Predict the court outcome from the case, the retrieved case evidence, and the cited legal articles.

Allowed labels:
A_WIN: plaintiff's main claim is accepted completely.
PARTIAL_A_WIN: plaintiff wins more than 50% but not all of the main claim.
B_WIN: plaintiff's main claim is rejected completely.
PARTIAL_B_WIN: plaintiff wins 50% or less of the main claim.

First, think step by step in Vietnamese: identify the parties' claims, the
relevant facts from CASE_EVIDENCE, and which legal basis from
RETRIEVED_LEGAL_EVIDENCE applies. Write this reasoning as plain text.

Then, on the final line, output strict JSON only (nothing after it):
{"label":"A_WIN|PARTIAL_A_WIN|B_WIN|PARTIAL_B_WIN","confidence":0.0-1.0,"reasoning":"short Vietnamese explanation"}
Do not invent legal articles or facts. Use only the supplied evidence IDs and case evidence."""


def build_user_prompt(
    case_query: str,
    law_articles: list[dict[str, Any]],
    case_evidence: list[dict[str, Any]] | None = None,
) -> str:
    """Build a compact, evidence-grounded prediction prompt.

    `case_evidence` is the chunk evidence collected by
    `CaseAPIClient.retrieve_multi()` (private-test scenario, no `case_fact`).
    """

    if law_articles:
        blocks = []
        for idx, art in enumerate(law_articles, 1):
            content = str(art.get("content", "")).replace("\n", " ")[:700]
            blocks.append(
                f"[{idx}] law_id={art.get('law_id')} aid={art.get('aid')} "
                f"dense_score={art.get('dense_score', '')} rerank_score={art.get('rerank_score', '')}\n{content}"
            )
        evidence = "\n\n".join(blocks)
    else:
        evidence = "No legal evidence was retrieved."

    if case_evidence:
        case_blocks = []
        for idx, chunk in enumerate(case_evidence, 1):
            text = str(chunk.get("text", "")).replace("\n", " ")[:700]
            case_blocks.append(f"[{idx}] chunk_id={chunk.get('chunk_id')} score={chunk.get('score', '')}\n{text}")
        case_evidence_block = "\n\n".join(case_blocks)
    else:
        case_evidence_block = "No additional case evidence was retrieved (case_fact already provided in case_query)."

    return (
        f"CASE_QUERY:\n{case_query}\n\n"
        f"CASE_EVIDENCE:\n{case_evidence_block}\n\n"
        f"RETRIEVED_LEGAL_EVIDENCE:\n{evidence}\n\n"
        "Think step by step, then return the final JSON line."
    )


def parse_label(text: str) -> tuple[str, float, str]:
    """Parse label, confidence, and reasoning from model output.

    The model is prompted for CoT text followed by a final JSON line, so this
    looks for the LAST `{...}` block in the text (the JSON is expected at the
    end) rather than the first, falling back to a bare label search and then
    a deterministic default if nothing parses.
    """

    raw = text.strip()
    try:
        end = raw.rfind("}")
        start = raw.rfind("{", 0, end + 1) if end >= 0 else -1
        # rfind("{") could match a brace that isn't the start of the JSON
        # object if the model nested braces in its reasoning; fall back to
        # the first "{" only if the naive rfind span doesn't parse.
        candidates = [start] if start >= 0 else []
        first_start = raw.find("{")
        if first_start >= 0 and first_start not in candidates:
            candidates.append(first_start)

        for candidate_start in candidates:
            if candidate_start < 0 or end <= candidate_start:
                continue
            try:
                payload = json.loads(raw[candidate_start:end + 1])
            except Exception:
                continue
            label = str(payload.get("label", "")).upper()
            confidence = float(payload.get("confidence", 0.5))
            reasoning = str(payload.get("reasoning", raw))
            if label in _LABEL_SET:
                return label, max(0.0, min(1.0, confidence)), reasoning
    except Exception:
        pass

    upper = raw.upper()
    for label in VALID_LABELS:
        if re.search(rf"\b{label}\b", upper):
            return label, 0.5, raw
    return "PARTIAL_A_WIN", 0.34, raw or "Fallback label because no valid model label was found."


# Backward-compatible alias: earlier code/tests may still import parse_prediction.
parse_prediction = parse_label


@dataclass(frozen=True)
class PredictResult:
    """Structured predictor output."""

    label: str
    reasoning: str
    raw_output: str
    confidence: float


class LLMPredictor:
    """Predict one of the four ALQAC outcome labels."""

    def __init__(self, mode: str = "mock", model_name: str = "Qwen/Qwen3-8B"):
        self.mode = mode
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        if mode == "local":
            self._load_model()

    def _load_model(self) -> None:
        """
        Load Qwen model completely offline from local HuggingFace cache.
        """

        from src.utils.model_cache import (
            configure_hf_cache,
            get_model_path,
        )

        # Configure HF cache before importing transformers
        configure_hf_cache()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Predictor local mode requires transformers, accelerate, torch, and bitsandbytes. "
                "Install requirements.txt or run with --llm-mode mock."
            ) from exc

        model_path = get_model_path(self.model_name)

        print(f"  [LLMPredictor] Loading local model:")
        print(f"      {model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True,
            local_files_only=True,
        )

        self._model.eval()

        print("  [LLMPredictor] Ready")

    def predict(
        self,
        case_query: str,
        law_articles: list[dict[str, Any]],
        case_evidence: list[dict[str, Any]] | None = None,
    ) -> PredictResult:
        if self.mode == "mock":
            return self._mock_predict(case_query, law_articles, case_evidence)
        return self._local_predict(case_query, law_articles, case_evidence)

    def _mock_predict(
        self,
        case_query: str,
        law_articles: list[dict[str, Any]],
        case_evidence: list[dict[str, Any]] | None = None,
    ) -> PredictResult:
        text = normalize_text(case_query, strip_accents=True)
        if any(term in text for term in ("mot phan", "chia doi", "50%", "1/2")):
            label = "PARTIAL_A_WIN"
            confidence = 0.62
        elif any(term in text for term in ("bac yeu cau", "khong chap nhan", "khong duoc chap nhan")):
            label = "B_WIN"
            confidence = 0.58
        elif any(term in text for term in ("toan bo", "chap nhan yeu cau")):
            label = "A_WIN"
            confidence = 0.56
        else:
            labels = VALID_LABELS
            label = labels[stable_hash(case_query) % len(labels)]
            confidence = 0.42
        evidence_ids = [f"{art.get('law_id')}:{art.get('aid')}" for art in law_articles]
        case_chunk_ids = [chunk.get("chunk_id") for chunk in (case_evidence or [])]
        raw = json.dumps({"label": label, "confidence": confidence, "reasoning": "mock deterministic"})
        reasoning = f"[MOCK] law_evidence={evidence_ids} case_evidence={case_chunk_ids}"
        return PredictResult(label=label, confidence=confidence, reasoning=reasoning, raw_output=raw)

    def _local_predict(
        self,
        case_query: str,
        law_articles: list[dict[str, Any]],
        case_evidence: list[dict[str, Any]] | None = None,
    ) -> PredictResult:
        assert self._model is not None and self._tokenizer is not None
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(case_query, law_articles, case_evidence)},
        ]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=768,  # CoT reasoning + final JSON line needs more room than a bare label
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        label, confidence, reasoning = parse_label(raw)
        return PredictResult(label=label, confidence=confidence, reasoning=reasoning, raw_output=raw)
