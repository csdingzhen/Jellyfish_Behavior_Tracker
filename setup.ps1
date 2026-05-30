# Jellyfish project environment setup
# Run from d:\Jellyfish: .\setup.ps1
# Requires: Python 3.10+ and CUDA 12.x driver already installed.

$ErrorActionPreference = "Stop"
$VenvDir = "venv"

# ── 1. Create virtual environment ────────────────────────────────────────────
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv $VenvDir
} else {
    Write-Host "Virtual environment already exists, skipping creation." -ForegroundColor Yellow
}

$Pip  = "$VenvDir\Scripts\pip.exe"
$Python = "$VenvDir\Scripts\python.exe"

# ── 2. Upgrade pip ───────────────────────────────────────────────────────────
Write-Host "`nUpgrading pip..." -ForegroundColor Cyan
& $Pip install --upgrade pip

# ── 3. PyTorch with CUDA 12.4 (compatible with driver CUDA 12.5) ─────────────
Write-Host "`nInstalling PyTorch + CUDA 12.4 wheels..." -ForegroundColor Cyan
& $Pip install torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu124

# ── 4. Core dependencies ─────────────────────────────────────────────────────
Write-Host "`nInstalling core dependencies..." -ForegroundColor Cyan
& $Pip install -r requirements.txt

# ── 5. SAM2 ──────────────────────────────────────────────────────────────────
Write-Host "`nInstalling SAM2..." -ForegroundColor Cyan
& $Pip install "git+https://github.com/facebookresearch/sam2.git"

# ── 6. CoTracker ─────────────────────────────────────────────────────────────
Write-Host "`nInstalling CoTracker..." -ForegroundColor Cyan
& $Pip install "git+https://github.com/facebookresearch/co-tracker.git"

# ── 7. Download model weights ─────────────────────────────────────────────────
Write-Host "`nDownloading model weights..." -ForegroundColor Cyan

# SAM2 weights (using sam2.1 hiera-base+ — good balance of speed/accuracy)
$WeightsDir = "weights"
New-Item -ItemType Directory -Force -Path $WeightsDir | Out-Null
New-Item -ItemType Directory -Force -Path "$WeightsDir\sam2" | Out-Null
New-Item -ItemType Directory -Force -Path "$WeightsDir\cotracker" | Out-Null

$Sam2Url    = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
$Sam2Config = "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_b%2B.yaml"
$CoTrackerUrl = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"

if (-not (Test-Path "$WeightsDir\sam2\sam2.1_hiera_base_plus.pt")) {
    Write-Host "  Downloading SAM2 weights (~160 MB)..."
    Invoke-WebRequest -Uri $Sam2Url -OutFile "$WeightsDir\sam2\sam2.1_hiera_base_plus.pt"
} else {
    Write-Host "  SAM2 weights already present."
}

if (-not (Test-Path "$WeightsDir\cotracker\scaled_offline.pth")) {
    Write-Host "  Downloading CoTracker3 weights (~100 MB)..."
    Invoke-WebRequest -Uri $CoTrackerUrl -OutFile "$WeightsDir\cotracker\scaled_offline.pth"
} else {
    Write-Host "  CoTracker weights already present."
}

# ── 8. CUDA smoke test ────────────────────────────────────────────────────────
Write-Host "`nRunning CUDA smoke test..." -ForegroundColor Cyan
& $Python scripts\test_cuda.py

Write-Host "`nSetup complete." -ForegroundColor Green
