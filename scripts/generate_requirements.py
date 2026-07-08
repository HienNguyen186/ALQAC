"""Generate a conservative requirements.txt for ALQAC."""

from __future__ import annotations

from pathlib import Path

REQUIREMENTS = [
    "# Core retrieval and utilities",
    "rank-bm25>=0.2.2",
    "numpy>=1.26.0",
    "tqdm>=4.66.0",
    "pyyaml>=6.0.2",
    "ftfy>=6.2.0",
    "requests>=2.32.0",
    "python-dotenv>=1.0.1",
    "",
    "# Local dense retrieval and LLM inference",
    "torch>=2.3.0",
    "sentence-transformers>=3.0.1",
    "transformers>=4.44.0",
    "accelerate>=0.33.0",
    "bitsandbytes>=0.43.0; platform_system != 'Windows'",
    "",
    "# Evaluation and tests",
    "scikit-learn>=1.5.0",
    "pandas>=2.2.0",
    "pytest>=8.2.0",
    "",
    "# Notebook/Colab convenience",
    "jupyter>=1.0.0",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    (root / "requirements.txt").write_text("\n".join(REQUIREMENTS) + "\n", encoding="utf-8")
    print(root / "requirements.txt")


if __name__ == "__main__":
    main()
