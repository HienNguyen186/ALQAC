@echo off
REM Windows batch file to run pipeline with local BGE-M3

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo ALQAC 2026 - Local Mode (BGE-M3 + Mock LLM)
echo ============================================================
echo.

REM Check if models directory exists
if not exist models\ (
    echo.
    echo [!] Models directory not found: %cd%\models
    echo.
    echo Please download models first:
    echo   python scripts\download_models.py
    echo.
    pause
    exit /b 1
)

echo [1/2] Checking GPU...
python -c "import torch; print('✓ CUDA:', torch.cuda.is_available())"

echo.
echo [2/2] Running pipeline...
echo.
python scripts\run_pipeline.py ^
    --rerank-mode local ^
    --llm-mode mock ^
    --limit 10

echo.
echo ============================================================
echo Done!
echo ============================================================
pause
