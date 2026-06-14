# Installation Guide

Instructions for setting up the Cassiopea pipeline on a new machine from the GitHub repository.

---

## Prerequisites

Before running the setup script, install these manually:

| Requirement | Version | Download |
| --- | --- | --- |
| Python | 3.10 or 3.11 | python.org |
| CUDA driver | 12.x | nvidia.com/drivers (must match your GPU) |
| Git | any recent | git-scm.com |
| ffmpeg | any recent | ffmpeg.org (add to PATH) |

Verify your CUDA driver is installed:

```powershell
nvidia-smi
```

You should see your GPU and driver version. The CUDA version shown must be ≥ 12.0.

---

## Step 1 — Clone the repository

```powershell
git clone https://github.com/csdingzhen/Jellyfish_Behavior_Tracker.git
cd Jellyfish_Behavior_Tracker
```

---

## Step 2 — Run the automated setup

```powershell
.\setup.ps1
```

This script does the following in order:

1. Creates a Python virtual environment at `venv/`
2. Installs PyTorch + CUDA 12.4 wheels from the PyTorch index
3. Installs core dependencies from `requirements.txt`
4. Installs SAM2 directly from GitHub (`facebookresearch/sam2`)
5. Installs CoTracker directly from GitHub (`facebookresearch/co-tracker`)
6. Downloads model weights (~260 MB total):
   - SAM2 tiny: `weights/sam2/sam2.1_hiera_tiny.pt`
   - CoTracker3: `weights/cotracker/scaled_offline.pth`
7. Runs a CUDA smoke test (`scripts/test_cuda.py`)

When complete you should see:

```text
Setup complete.
```

---

## Step 3 — Configure paths

Open `config.py` and set `VIDEO_DIR` to wherever your recordings are stored, for example:

```python
VIDEO_DIR = Path(r"D:\LabData\Cassiopea\recordings")
```

You can leave all other paths at their defaults unless you are storing model weights in a non-standard location.

---

## Step 4 — Verify the installation

Launch the graphical interface:

```powershell
.\venv\Scripts\python scripts\run_ui.py
```

The napari window should open with **Calibrate** and **Process** tabs.

Or verify via CLI:

```powershell
.\venv\Scripts\python scripts\run_pipeline.py --help
```

---

## Manual installation (only do this if setup.ps1 fails)

If the automated script fails, install each component manually:

### Python environment

```powershell
python -m venv venv
.\venv\Scripts\pip install --upgrade pip
```

### PyTorch with CUDA

Replace `cu124` with your CUDA version if different (cu118, cu121, cu124, cu126):

```powershell
.\venv\Scripts\pip install torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu124
```

### Core dependencies

```powershell
.\venv\Scripts\pip install -r requirements.txt
```

### SAM2

```powershell
.\venv\Scripts\pip install "git+https://github.com/facebookresearch/sam2.git"
```

### CoTracker

```powershell
.\venv\Scripts\pip install "git+https://github.com/facebookresearch/co-tracker.git"
```

### napari (for the UI)

```powershell
.\venv\Scripts\pip install "napari[all]"
.\venv\Scripts\pip install magicgui superqt
```

### Model weights

Download manually and place at the expected paths:

| Model | URL | Target path |
| --- | --- | --- |
| SAM2 tiny | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt` | `weights/sam2/sam2.1_hiera_tiny.pt` |
| CoTracker3 | `https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth` | `weights/cotracker/scaled_offline.pth` |

If you prefer the `base_plus` SAM2 model (larger, slower):

| Model | URL | Target path |
| --- | --- | --- |
| SAM2 base+ | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt` | `weights/sam2/sam2.1_hiera_base_plus.pt` |

Then update `config.py`:

```python
SAM2_WEIGHTS = WEIGHTS_DIR / "sam2" / "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG  = "configs/sam2.1/sam2.1_hiera_b+.yaml"
```

---

## Linux / macOS

The project is developed on Windows but the Python code is cross-platform. Replace PowerShell commands with bash equivalents:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"
pip install "git+https://github.com/facebookresearch/co-tracker.git"
pip install "napari[all]" magicgui superqt
```

Weight download:

```bash
mkdir -p weights/sam2 weights/cotracker
wget -O weights/sam2/sam2.1_hiera_tiny.pt \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
wget -O weights/cotracker/scaled_offline.pth \
    "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
```

---

## Troubleshooting

### `nvidia-smi` not found

The CUDA driver is not installed. Download from nvidia.com/drivers. The pipeline requires an NVIDIA GPU; it will not run on CPU only.

### `torch.cuda.is_available()` returns False

1. Confirm the driver version with `nvidia-smi`.
2. Reinstall PyTorch with the correct CUDA index URL for your driver version.
3. Check that the venv Python is the one you expect: `.\venv\Scripts\python -c "import sys; print(sys.executable)"`.

### SAM2 install fails (pip build error)

SAM2 requires a C++ compiler. On Windows:

```powershell
# Install Visual Studio Build Tools (free):
winget install Microsoft.VisualStudio.2022.BuildTools
```

Then re-run the SAM2 install.

### CoTracker install fails

Same C++ compiler requirement. See above.

### `napari` window does not open

Install Qt backend explicitly:

```powershell
.\venv\Scripts\pip install pyqt5
```

Or alternatively:

```powershell
.\venv\Scripts\pip install pyside2
```

### Out of GPU memory during pipeline

1. Reduce SAM2 model size: `--sam2-model tiny`.
2. Reduce `--image-size 512` (already the default on the performance branch).
3. Close other GPU-intensive applications.
4. If still failing: run SAM2 and CoTracker in separate processes (use individual scripts rather than `run_pipeline.py`).

---

## Repository layout (what is and is not in git)

| Path | In git | Notes |
| --- | --- | --- |
| `src/`, `ui/`, `scripts/` | Yes | All source code |
| `calibration/*.json` | Yes | Rhopalium calibration files — do not delete |
| `requirements.txt`, `setup.ps1` | Yes | Dependency management |
| `config.py` | Yes | Defaults only; your `VIDEO_DIR` is a local edit |
| `weights/` | No (.gitignore) | Downloaded by setup.ps1 |
| `outputs/` | No (.gitignore) | Pipeline results; can be regenerated |
| `data/` | No (.gitignore) | Raw video clips |
| `venv/` | No (.gitignore) | Virtual environment |
| `assets/` | Yes | Resource related to UI |
| `docs/` | Yes | Knowledge base for developer or LLM agents |
