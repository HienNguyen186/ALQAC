#!/bin/bash
# Setup môi trường trên GPU VM (RunPod / Vast.ai)
# Chạy 1 lần duy nhất sau khi clone repo:
#   bash setup_vm.sh

set -e   # Dừng ngay nếu có lỗi

echo "======================================"
echo "  ALQAC 2026 — VM Setup"
echo "======================================"

# 1. PyTorch với CUDA (VM thường đã có CUDA 12.x)
echo "[1/4] Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# 2. Phần còn lại
echo "[2/4] Installing requirements..."
pip install -r requirements.txt -q

# 3. Tạo .env nếu chưa có
echo "[3/4] Setting up .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  → Đã tạo .env. Nhớ điền ALQAC_API_TOKEN!"
else
    echo "  → .env đã tồn tại, bỏ qua."
fi

# 4. Tạo thư mục cần thiết
echo "[4/4] Creating directories..."
mkdir -p data/raw data/processed outputs/submissions

echo ""
echo "======================================"
echo "  Setup xong! Các bước tiếp theo:"
echo ""
echo "  1. Điền token vào .env:"
echo "     nano .env"
echo ""
echo "  2. Copy data lên VM (chạy từ laptop):"
echo "     scp data/raw/*.json root@<VM-IP>:~/alqac2026/data/raw/"
echo ""
echo "  3. Test pipeline (không cần GPU):"
echo "     python scripts/run_pipeline.py --rerank-mode mock --llm-mode mock --limit 3"
echo ""
echo "  4. Chạy thật:"
echo "     python scripts/run_pipeline.py --rerank-mode llm --llm-mode local"
echo "======================================"
