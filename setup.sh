#!/bin/bash
# ============================================
# Setup script for Political Bias Experiment
# Mac Studio M1 / Apple Silicon
# ============================================

echo "=== Political Bias Cosine Similarity Gap Experiment ==="
echo "=== Setup for Mac Studio M1 / Apple Silicon ==="
echo ""

# 1. 检查 Python 版本
echo "[1/4] Checking Python..."
python3 --version

# 2. 创建虚拟环境 (推荐)
echo ""
echo "[2/4] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
echo "Virtual environment activated: .venv"

# 3. 安装依赖
echo ""
echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision  # Apple Silicon 自动使用 MPS
pip install transformers accelerate
pip install matplotlib numpy tqdm scipy
pip install huggingface_hub

# 4. 验证 MPS 可用
echo ""
echo "[4/4] Verifying MPS (Metal Performance Shaders) availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✅ MPS is ready! Your Mac GPU will be used for inference.')
else:
    print('⚠️  MPS not available, will fall back to CPU.')
"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To run the quick demo (1.5B model, ~5 min):"
echo "  python run_quick_demo.py"
echo ""
echo "To run the full experiment (7B model, ~20 min):"
echo "  python run_experiment.py"
echo ""
echo "To run with 4-bit quantization (saves memory):"
echo "  pip install bitsandbytes"
echo "  python run_experiment.py --quantize"
