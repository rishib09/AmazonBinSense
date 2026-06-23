#!/bin/bash
# BinSense AI — Colab bootstrap cell
# Paste this into the FIRST cell of every notebook as:
#   %%bash
#   bash /content/drive/MyDrive/BinSense/setup/install_colab.sh
#
# Or copy-paste the block below directly into a code cell.

set -e

echo "=== BinSense Colab Setup ==="
echo "Python: $(python --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Detect GPU runtime
GPU_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

# Install FAISS — GPU build if CUDA available, else CPU
if [ "$GPU_AVAILABLE" = "True" ]; then
  echo "Installing faiss-gpu..."
  pip install -q faiss-gpu
else
  echo "Installing faiss-cpu..."
  pip install -q faiss-cpu
fi

# Core packages (Colab already has torch, numpy, pandas, sklearn, cv2, PIL)
pip install -q \
  ultralytics \
  timm \
  mlflow \
  prometheus-client \
  streamlit \
  gradio \
  python-dotenv

# Segment Anything Model (SAM) from Meta
pip install -q git+https://github.com/facebookresearch/segment-anything.git

echo "=== Setup complete ==="
python -c "
import torch, ultralytics, timm, faiss, mlflow
print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'ultralytics={ultralytics.__version__}')
print(f'timm={timm.__version__}')
print(f'mlflow={mlflow.__version__}')
"
