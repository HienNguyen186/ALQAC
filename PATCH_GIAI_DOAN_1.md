# Giai đoạn 1 — Patch cho src/prediction/llm_predictor.py

## 1. Sửa `parse_prediction` để nhận `default_label` thay vì hard-code

Thay:

```python
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
```

Bằng:

```python
def parse_prediction(text: str, default_label: str = "PARTIAL_A_WIN") -> tuple[str, float, str, str]:
    """Parse label, confidence, reasoning từ output model.

    Trả về thêm parse_status ("json" | "regex" | "hard_fallback") để
    theo dõi tỷ lệ fallback qua toàn bộ run.
    """
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
                return label, max(0.0, min(1.0, confidence)), reasoning, "json"
    except Exception:
        pass

    # Regex fallback
    upper = raw.upper()
    for label in VALID_LABELS:
        if re.search(rf"\b{label}\b", upper):
            LOGGER.warning("[LLMPredictor] JSON parse failed, regex fallback: %s", label)
            return label, 0.5, raw[:200], "regex"

    LOGGER.error("[LLMPredictor] Could not parse output: %r", raw[:300])
    return default_label, 0.34, raw[:200] or "Fallback.", "hard_fallback"
```

## 2. Cập nhật `PredictResult` để lưu `parse_status`

Thay:

```python
@dataclass(frozen=True)
class PredictResult:
    label:      str
    reasoning:  str
    raw_output: str
    confidence: float
```

Bằng:

```python
@dataclass(frozen=True)
class PredictResult:
    label:        str
    reasoning:    str
    raw_output:   str
    confidence:   float
    parse_status: str = "mock"   # "json" | "regex" | "hard_fallback" | "mock"
```

## 3. Truyền `default_label` vào `LLMPredictor` và class-level counters

Thay `__init__`:

```python
    def __init__(
        self,
        mode: str = "local",
        model_name: str = "Qwen/Qwen3-8B",
        cache_dir: str | Path | None = None,
    ):
        self.mode       = mode
        self.model_name = model_name
```

Bằng:

```python
    def __init__(
        self,
        mode: str = "local",
        model_name: str = "Qwen/Qwen3-8B",
        cache_dir: str | Path | None = None,
        default_label: str = "PARTIAL_A_WIN",
    ):
        self.mode          = mode
        self.model_name    = model_name
        self.default_label = default_label
        # Đếm để biết % case phải rơi vào fallback qua cả run
        self._parse_stats: dict[str, int] = {"json": 0, "regex": 0, "hard_fallback": 0}
```

Đọc `default_label` từ `outputs/baseline_check.json` (kết quả Giai đoạn 0) khi khởi tạo trong
`scripts/run_pipeline.py` / `scripts/main.py`:

```python
import json as _json
_baseline_path = PROJECT_ROOT / "outputs" / "baseline_check.json"
_default_label = "PARTIAL_A_WIN"
if _baseline_path.exists():
    _default_label = _json.loads(_baseline_path.read_text(encoding="utf-8"))["majority_label"]

predictor = LLMPredictor(
    mode=args.llm_mode,
    model_name=pc.get("predictor_model", "Qwen/Qwen3-8B"),
    default_label=_default_label,
)
```

## 4. Cập nhật `_local_predict` để dùng `default_label` và log thống kê

Thay:

```python
        LOGGER.debug("[LLMPredictor] raw: %r", raw[:200])
        label, confidence, reasoning = parse_prediction(raw)
        return PredictResult(label=label, confidence=confidence, reasoning=reasoning, raw_output=raw)
```

Bằng:

```python
        LOGGER.debug("[LLMPredictor] raw: %r", raw[:200])
        label, confidence, reasoning, status = parse_prediction(raw, default_label=self.default_label)
        self._parse_stats[status] = self._parse_stats.get(status, 0) + 1
        return PredictResult(
            label=label, confidence=confidence, reasoning=reasoning,
            raw_output=raw, parse_status=status,
        )

    def report_parse_stats(self) -> None:
        """Gọi cuối pipeline để xem % case rơi vào fallback."""
        total = sum(self._parse_stats.values())
        if not total:
            return
        LOGGER.info("[LLMPredictor] Parse stats over %d cases:", total)
        for status, cnt in self._parse_stats.items():
            pct = cnt / total * 100
            LOGGER.info("  %-14s %5d (%.1f%%)", status, cnt, pct)
        fallback_pct = self._parse_stats.get("hard_fallback", 0) / total * 100
        if fallback_pct > 10:
            LOGGER.warning(
                "[LLMPredictor] hard_fallback rate = %.1f%% (>10%%). "
                "Prompt hoặc output format của model đang có vấn đề — "
                "kiểm tra raw_output mẫu trước khi tối ưu tiếp Giai đoạn 2/3.",
                fallback_pct,
            )
```

## 5. Gọi `report_parse_stats()` sau vòng lặp trong `run_pipeline.py` / `main.py`

Thêm ngay sau vòng `for case in tqdm(cases, ...)`:

```python
    predictor.report_parse_stats()
```

## 6. Cập nhật `test_mock_components.py` (mock path trả 5 giá trị không đổi vì mock không qua parse_prediction — không cần sửa test).
