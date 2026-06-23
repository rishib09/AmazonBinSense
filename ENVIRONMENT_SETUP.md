# BinSense AI — Environment Setup Guide

Two parallel setups that stay in sync via the **same `.ipynb` files** stored on Google Drive.

---

## The problem: Python version mismatch

| Environment | Python | Key constraint |
|---|---|---|
| Google Colab | **3.11** | locked by Google |
| Your local machine | 3.14 | too new — PyTorch wheels don't ship for 3.14 yet |

**Fix: already solved** — Python 3.11.15 is installed via `uv` at
`C:\Users\rishi\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none`.

> ⚠️ **Venv must NOT live on Google Drive** — Drive doesn't support hardlinks and
> will corrupt metadata on thousands of tiny files. The binsense venv lives at
> `C:\Users\rishi\envs\binsense` (local SSD).

---

## Setup A — Google Colab (primary / GPU training)

### One-time steps
1. Go to [colab.research.google.com](https://colab.research.google.com) → **Settings → Colab Pro** (upgrade if not done).
2. Open any BinSense notebook → **Runtime → Change runtime type → T4 GPU**.
3. Mount Drive + install packages by pasting the first cell below.

### Standard first cell (paste into every notebook)
```python
# Cell 1 — always run first
import subprocess, sys

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Add project root to Python path so 'from utils.env_utils import setup_env' works
sys.path.insert(0, '/content/drive/MyDrive/BinSense')

# Install packages not pre-installed in Colab
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'ultralytics', 'timm', 'mlflow', 'gradio',
    'git+https://github.com/facebookresearch/segment-anything.git'
], check=True)

# GPU-aware FAISS
import torch
faiss_pkg = 'faiss-gpu' if torch.cuda.is_available() else 'faiss-cpu'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', faiss_pkg], check=True)

# Verify
from utils.env_utils import setup_env, print_gpu_info
cfg = setup_env()
print_gpu_info()
```

---

## Setup B — VS Code local (development / CPU work)

Use this for: data exploration, code editing, debugging, non-GPU steps.
GPU-heavy steps (YOLO training, SAM labeling) should still run on Colab.

### Step 1 — Python 3.11 + venv (already done)

Python 3.11.15 is managed by `uv`. The venv is at `C:\Users\rishi\envs\binsense`.

To **activate** in a terminal:
```powershell
C:\Users\rishi\envs\binsense\Scripts\Activate.ps1
```

To **recreate from scratch** (e.g. on a new machine):
```powershell
# Install uv if needed:  winget install astral-sh.uv
uv venv C:\Users\rishi\envs\binsense --python cpython-3.11
uv pip install --python C:\Users\rishi\envs\binsense\Scripts\python.exe `
    numpy pandas scipy scikit-learn matplotlib seaborn tqdm pillow requests python-dotenv pyyaml
uv pip install --python C:\Users\rishi\envs\binsense\Scripts\python.exe `
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python C:\Users\rishi\envs\binsense\Scripts\python.exe `
    ultralytics timm opencv-python-headless faiss-cpu
uv pip install --python C:\Users\rishi\envs\binsense\Scripts\python.exe `
    jupyter jupyterlab ipywidgets ipython mlflow prometheus-client streamlit gradio boto3
# SAM (Meta, not on PyPI):
uv pip install --python C:\Users\rishi\envs\binsense\Scripts\python.exe `
    git+https://github.com/facebookresearch/segment-anything.git
```

### Step 3 — VS Code extensions
Install these extensions (Ctrl+Shift+X, search by ID):

| Extension ID | Purpose |
|---|---|
| `ms-toolsai.jupyter` | Run `.ipynb` notebooks — **required** |
| `ms-toolsai.vscode-jupyter-renderers` | Rich outputs (plots, DataFrames) |
| `ms-toolsai.jupyter-keymap` | Jupyter keyboard shortcuts |
| `ms-toolsai.jupyter-cell-tags` | Cell tags (Colab uses these) |
| `ms-python.python` | Python language support |
| `ms-python.vscode-pylance` | IntelliSense + type checking |
| `ms-python.black-formatter` | Auto-format on save |
| `ms-azuretools.vscode-docker` | Docker (Week 4+) |
| `eamodio.gitlens` | Git history inline |

**Bulk install (PowerShell):**
```powershell
@(
  'ms-toolsai.jupyter',
  'ms-toolsai.vscode-jupyter-renderers',
  'ms-toolsai.jupyter-keymap',
  'ms-toolsai.jupyter-cell-tags',
  'ms-python.python',
  'ms-python.vscode-pylance',
  'ms-python.black-formatter',
  'ms-azuretools.vscode-docker',
  'eamodio.gitlens'
) | ForEach-Object { code --install-extension $_ }
```

### Step 4 — Select kernel in VS Code
1. Open any `.ipynb` file.
2. Click **"Select Kernel"** (top-right of notebook).
3. Choose **"Python Environments"** → pick `.venv (Python 3.11)` or your conda `binsense` env.

---

## Option C — VS Code connected to Colab runtime (SSH tunnel)

This lets VS Code be the editor while Colab provides the GPU.

> **Requires Colab Pro+** (standard Colab drops idle connections too aggressively).

1. In Colab, run:
   ```python
   # Install colab-ssh
   !pip install -q colab_ssh
   from colab_ssh import launch_ssh_cloudflared, init_git_cloudflared
   launch_ssh_cloudflared(password="pick-a-password")
   ```
   Copy the printed hostname.

2. In VS Code: install `ms-vscode-remote.remote-ssh`.
3. **Ctrl+Shift+P** → "Remote-SSH: Connect to Host" → paste hostname.
4. VS Code now runs on the Colab VM — full GPU access, local editor experience.

**Verdict:** Nice for debugging GPU issues. For day-to-day BinSense work, Setup B (local kernel) + plain Colab in browser is simpler.

---

## Local Jupyter server options

If you prefer browser-based notebooks locally (without VS Code):

| Server | Command | Best for |
|---|---|---|
| **JupyterLab** | `jupyter lab` | Rich UI, file browser, terminal — recommended |
| Classic Notebook | `jupyter notebook` | Lightweight |
| VS Code built-in | Open `.ipynb` in VS Code | Fastest, no browser needed |

**JupyterLab + "Connect to local runtime" in Colab browser:**
1. `jupyter lab --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --no-browser`
2. In Colab: **Runtime → Connect to local runtime** → paste URL with token.
3. Colab UI runs in browser; compute is your local machine.

---

## Keeping both setups in sync

The single source of truth is the `.ipynb` files on **Google Drive** (mounted in Colab at `/content/drive/MyDrive/BinSense/notebooks/`).

In VS Code, open Drive as a local folder:
- On Windows with Google Drive desktop app: `G:\My Drive\Interview Kickstart\Capstone Project\Amazon BinSense\`
- Open this folder in VS Code → edit notebooks → they auto-sync to Drive → open in Colab.

**Path compatibility** is handled by `utils/env_utils.py`:
```python
from utils.env_utils import setup_env
cfg = setup_env()
# cfg.images_dir resolves to Drive path on Colab, local path elsewhere
```

---

## Quick-start checklist (Week 0)

- [x] Python 3.11 installed locally — `cpython-3.11.15` via `uv` at `C:\Users\rishi\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none`
- [x] venv created at `C:\Users\rishi\envs\binsense` (local, not on Drive)
- [x] All libraries installed and smoke-tested — numpy/pandas/scipy/sklearn/matplotlib/torch/torchvision/ultralytics/timm/opencv/faiss/mlflow/streamlit/gradio/boto3/jupyter ✅
- [x] VS Code kernel auto-set via `.vscode/settings.json` → `binsense` venv
- [x] VS Code extensions — jupyter/python/pylance/black/docker/gitlens installed
- [x] `from utils.env_utils import setup_env; cfg = setup_env()` prints correct paths locally ✅
- [ ] VS Code: open a `.ipynb` notebook, confirm kernel shows `binsense (Python 3.11.15)`, run a cell
- [ ] Colab: open notebook, mount Drive, run the standard first cell above
- [ ] `setup_env()` prints correct Drive paths on Colab
- [ ] Docker Desktop installed → `docker run hello-world` works
- [ ] GitHub repo created + project files pushed
