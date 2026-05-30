"""
Smoke test: verifies CUDA, PyTorch, SAM2, CoTracker, torchvision (RAFT),
OpenCV, and scipy are all importable and CUDA tensors work on the RTX 4060.
"""

import sys

def check(label, fn):
    try:
        result = fn()
        print(f"  [OK]  {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False

print("=" * 60)
print("Jellyfish pipeline — environment smoke test")
print("=" * 60)

failures = 0

# ── PyTorch + CUDA ────────────────────────────────────────────────────────────
import torch

ok = check("PyTorch version", lambda: torch.__version__)
failures += not ok

ok = check("CUDA available", lambda: torch.cuda.is_available())
failures += not ok

if torch.cuda.is_available():
    check("GPU name",        lambda: torch.cuda.get_device_name(0))
    check("VRAM (GB)",       lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
    ok = check("Tensor on CUDA", lambda: str(torch.zeros(3, 3).cuda().device))
    failures += not ok
else:
    print("  [WARN] Skipping GPU tensor test — no CUDA device found.")
    failures += 1

# ── torchvision (RAFT lives here) ─────────────────────────────────────────────
ok = check("torchvision", lambda: __import__("torchvision").__version__)
failures += not ok

ok = check("torchvision RAFT import", lambda: (
    __import__("torchvision.models.optical_flow", fromlist=["raft_large"]),
    "ok"
)[1])
failures += not ok

# ── SAM2 ─────────────────────────────────────────────────────────────────────
ok = check("SAM2 import", lambda: (
    __import__("sam2.build_sam", fromlist=["build_sam2_video_predictor"]),
    "ok"
)[1])
failures += not ok

# ── CoTracker ────────────────────────────────────────────────────────────────
ok = check("CoTracker import", lambda: (
    __import__("cotracker.predictor", fromlist=["CoTrackerPredictor"]),
    "ok"
)[1])
failures += not ok

# ── OpenCV ───────────────────────────────────────────────────────────────────
ok = check("OpenCV", lambda: __import__("cv2").__version__)
failures += not ok

# ── scipy ────────────────────────────────────────────────────────────────────
ok = check("scipy", lambda: __import__("scipy").__version__)
failures += not ok

# ── imageio ──────────────────────────────────────────────────────────────────
ok = check("imageio", lambda: __import__("imageio").__version__)
failures += not ok

# ── Summary ──────────────────────────────────────────────────────────────────
print("=" * 60)
if failures == 0:
    print("All checks passed. Environment is ready.")
else:
    print(f"{failures} check(s) failed. See above.")
    sys.exit(1)
