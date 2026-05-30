"""
Central path configuration for the Jellyfish pipeline.

Edit VIDEO_DIR to point at wherever your recordings live.
Everything else is derived from project root.
"""

from pathlib import Path

# ── Project root (this file's directory) ─────────────────────────────────────
ROOT = Path(__file__).parent

# ── Where your raw video recordings are stored ───────────────────────────────
# Set this to the actual folder on your machine; do NOT move videos into ROOT.
# Examples:
#   VIDEO_DIR = Path(r"D:\LabData\Jellyfish\recordings")
#   VIDEO_DIR = Path(r"E:\Cassiopea_videos")
VIDEO_DIR = Path(r"D:\Jellyfish\data")   # default: project data/ folder

# ── Model weights ─────────────────────────────────────────────────────────────
WEIGHTS_DIR  = ROOT / "weights"
SAM2_WEIGHTS = WEIGHTS_DIR / "sam2" / "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG  = "configs/sam2.1/sam2.1_hiera_b+.yaml"   # path used by build_sam2_video_predictor
COTRACKER_WEIGHTS = WEIGHTS_DIR / "cotracker" / "scaled_offline.pth"

# ── Cached pipeline outputs ───────────────────────────────────────────────────
OUTPUTS_DIR = ROOT / "outputs"

# ── Recording parameters (update once you have confirmed values) ───────────────
FPS = 120                 # recording frame rate — confirm from your actual clips
