"""
scripts/run_raft.py

Stage 5 -- Optical flow divergence and contraction signal.

Uses RAFT-small (torchvision) to compute optical flow between consecutive
frames, derives the divergence field (local contraction/expansion rate),
integrates it over the bell mask, and detects pulses as peaks.

Two-pass processing
--------------------
  Pass 1 -- stream all frame pairs, compute scalar contraction signal, write CSV.
  Pass 2 -- for each detected peak, reload the PRE_WINDOW frames before it,
            compute divergence fields, save as .npy for Stage 6 pulse analysis.

Bell mask
----------
  Primary:  per-frame PNG from outputs/<stem>_masks/  (if --save-all-masks was used)
  Fallback: circular mask from (cx, cy, radius) in *_seg.csv  (always available)

Usage:
    venv\Scripts\python scripts\run_raft.py
    venv\Scripts\python scripts\run_raft.py --video data/test_clip_1min.mp4
    venv\Scripts\python scripts\run_raft.py --stride 2 --min-distance 30

Outputs (in outputs/):
    <stem>_contraction.csv        -- frame_idx, timestamp_s, contraction, mask_area
    <stem>_peaks.csv              -- peak frame, time, prominence, width
    <stem>_peak_divfields/        -- divergence .npy files near each peak (for Stage 6)
    <stem>_contraction_plot.png   -- signal plot with peaks annotated
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
from tqdm import tqdm
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUTS_DIR, FPS

PRE_WINDOW   = 10    # frames before each peak to save divergence fields
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Frame / mask helpers ──────────────────────────────────────────────────────

def load_frame_rgb(path: Path) -> np.ndarray:
    """Load JPEG frame as uint8 RGB numpy array (H, W, 3)."""
    bgr = cv2.imread(str(path))
    return bgr[:, :, ::-1].copy()


def to_tensor(frame_rgb: np.ndarray, device) -> torch.Tensor:
    """(H, W, 3) uint8 -> (1, 3, H, W) uint8 tensor on device."""
    return (torch.from_numpy(frame_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device))


def load_seg(seg_csv: Path) -> dict[int, tuple[float, float, float]]:
    """Returns {frame_idx: (cx, cy, radius_px)} from SAM2 seg CSV."""
    out = {}
    with open(seg_csv) as f:
        for row in csv.DictReader(f):
            out[int(row["frame_idx"])] = (
                float(row["cx"]), float(row["cy"]), float(row["radius_px"])
            )
    return out


def get_bell_mask(frame_idx: int, h: int, w: int,
                  seg: dict, mask_dir: Path | None) -> np.ndarray:
    """
    Return float32 (H, W) bell mask in [0, 1].
    Tries PNG mask first; falls back to circle from seg.csv.
    """
    if mask_dir is not None:
        png = mask_dir / f"{frame_idx:06d}.png"
        if png.exists():
            m = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            return (m > 127).astype(np.float32)

    # Circular fallback — interpolate nearest seg entry
    if not seg:
        return np.ones((h, w), dtype=np.float32)

    # Find nearest available frame in seg
    nearest = min(seg.keys(), key=lambda k: abs(k - frame_idx))
    cx, cy, r = seg[nearest]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (round(cx), round(cy)), round(r), 1.0, -1)
    return mask


# ── Divergence ────────────────────────────────────────────────────────────────

def flow_to_divergence(flow: torch.Tensor) -> np.ndarray:
    """
    flow: (1, 2, H, W) float32.
    Returns divergence (H, W) float32: du/dx + dv/dy.
    Negative = local contraction (pixels converging).
    """
    u = flow[0, 0]   # horizontal displacement
    v = flow[0, 1]   # vertical displacement
    du_dx = torch.gradient(u, dim=1)[0]
    dv_dy = torch.gradient(v, dim=0)[0]
    return (du_dx + dv_dy).cpu().numpy()


# ── Pass 1: scalar contraction signal ────────────────────────────────────────

def compute_contraction_signal(
    frames_dir: Path,
    seg: dict,
    mask_dir: Path | None,
    fps_eff: float,
    stride: int,
    transforms,
    model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stream all consecutive frame pairs, compute scalar contraction signal.

    Returns:
        frame_indices  : (N,) int   -- extracted-frame index for each sample
        timestamps     : (N,) float -- seconds
        signal         : (N,) float -- integrated -div over bell mask
                                       positive = contraction, negative = expansion
    """
    all_jpgs = sorted(frames_dir.glob("*.jpg"))
    n = len(all_jpgs)
    if n < 2:
        sys.exit(f"Need at least 2 frames in {frames_dir}")

    h, w = cv2.imread(str(all_jpgs[0])).shape[:2]

    frame_indices = []
    timestamps    = []
    signal_vals   = []

    # Pre-load first frame
    prev_frame = load_frame_rgb(all_jpgs[0])
    prev_t1    = to_tensor(prev_frame, DEVICE)

    for i in tqdm(range(1, n), desc="RAFT pass 1", unit="fr"):
        curr_frame = load_frame_rgb(all_jpgs[i])
        curr_t2    = to_tensor(curr_frame, DEVICE)

        t1, t2 = transforms(prev_t1, curr_t2)
        with torch.no_grad():
            flow = model(t1, t2)[-1]   # finest flow: (1, 2, H, W)

        div  = flow_to_divergence(flow)

        raw_frame_idx = i * stride
        mask = get_bell_mask(raw_frame_idx, h, w, seg, mask_dir)
        area = mask.sum()

        # Contraction = -divergence integrated over bell
        # (negative div = pixels converging = contraction -> we negate to get positive)
        contraction = float(-( div * mask).sum())

        frame_indices.append(raw_frame_idx)
        timestamps.append(raw_frame_idx / (fps_eff * stride))
        signal_vals.append(contraction)

        prev_t1 = curr_t2
        prev_frame = curr_frame

    return (np.array(frame_indices),
            np.array(timestamps),
            np.array(signal_vals, dtype=np.float32))


# ── Peak detection ────────────────────────────────────────────────────────────

def detect_peaks(signal: np.ndarray, fps_eff: float,
                 min_distance_s: float, prominence_frac: float
                 ) -> tuple[np.ndarray, dict]:
    """
    Find contraction peaks.
    min_distance_s : minimum time between peaks (seconds)
    prominence_frac: peak prominence >= this fraction of signal range
    """
    min_dist   = max(1, round(min_distance_s * fps_eff))
    prom_level = (signal.max() - signal.min()) * prominence_frac
    peaks, props = scipy.signal.find_peaks(
        signal,
        distance=min_dist,
        prominence=prom_level,
    )
    return peaks, props


# ── Pass 2: save divergence fields near each peak ────────────────────────────

def save_peak_divfields(
    frames_dir: Path,
    peak_frame_indices: np.ndarray,  # raw frame indices of peaks
    div_dir: Path,
    seg: dict,
    mask_dir: Path | None,
    stride: int,
    transforms,
    model,
    pre_window: int,
) -> None:
    """
    For each detected peak, reload the PRE_WINDOW frames before it and
    save the divergence fields as .npy arrays.

    Saved file naming: <div_dir>/peak_<peak_frame>_t<t-k>.npy
    where t-k counts backwards from the peak.
    """
    div_dir.mkdir(exist_ok=True)
    all_jpgs = sorted(frames_dir.glob("*.jpg"))
    n        = len(all_jpgs)
    h, w     = cv2.imread(str(all_jpgs[0])).shape[:2]

    for peak_raw in tqdm(peak_frame_indices, desc="RAFT pass 2 (peaks)", unit="peak"):
        peak_extracted = peak_raw // stride   # index in extracted frames
        start = max(0, peak_extracted - pre_window)
        end   = min(n - 1, peak_extracted)

        # Load window of frames
        window_frames = [load_frame_rgb(all_jpgs[i]) for i in range(start, end + 1)]

        for j in range(len(window_frames) - 1):
            t1 = to_tensor(window_frames[j],     DEVICE)
            t2 = to_tensor(window_frames[j + 1], DEVICE)
            t1, t2 = transforms(t1, t2)
            with torch.no_grad():
                flow = model(t1, t2)[-1]
            div  = flow_to_divergence(flow)
            mask = get_bell_mask((start + j) * stride, h, w, seg, mask_dir)
            div_masked = div * mask

            frames_before_peak = end - (start + j)
            fname = div_dir / f"peak_{peak_raw:06d}_minus{frames_before_peak:02d}.npy"
            np.save(str(fname), div_masked.astype(np.float32))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_contraction(timestamps: np.ndarray, signal: np.ndarray,
                     peak_idxs: np.ndarray, out_path: Path,
                     fps_eff: float) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(timestamps, signal, lw=0.8, color="#4A9EFF", label="contraction signal")
    ax.axhline(0, lw=0.5, color="grey", linestyle="--")
    if len(peak_idxs):
        ax.scatter(timestamps[peak_idxs], signal[peak_idxs],
                   color="red", s=40, zorder=5, label=f"{len(peak_idxs)} peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("-div(flow) * mask  [a.u.]")
    ax.set_title("Bell contraction signal  (positive = contracting)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5 -- RAFT optical flow + contraction")
    ap.add_argument("--video", default="data/test_clip_1min.mp4")
    ap.add_argument("--stride", type=int, default=1,
                    help="Frame stride used when extracting frames (must match SAM2 stride)")
    ap.add_argument("--min-distance", type=float, default=0.42,
                    help="Minimum time (s) between detected peaks (default 0.42 = 50 frames at 120fps)")
    ap.add_argument("--prominence", type=float, default=0.05,
                    help="Peak prominence threshold as fraction of signal range (default 0.05)")
    ap.add_argument("--pre-window", type=int, default=25,
                    help="Frames before each peak to save divergence fields (default 25)")
    ap.add_argument("--skip-divfields", action="store_true",
                    help="Skip pass 2 (don't save per-peak divergence fields)")
    args = ap.parse_args()

    root       = Path(__file__).parent.parent
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = root / video_path

    stem       = video_path.stem
    frames_dir = OUTPUTS_DIR / f"{stem}_frames"
    mask_dir   = OUTPUTS_DIR / f"{stem}_masks"
    seg_csv    = OUTPUTS_DIR / f"{stem}_seg.csv"

    if not frames_dir.exists() or not list(frames_dir.glob("*.jpg")):
        sys.exit(f"No extracted frames found at {frames_dir}\n"
                 "Run run_sam2.py first (it extracts frames as a side effect).")

    # Load seg CSV if available
    seg: dict = {}
    if seg_csv.exists():
        seg = load_seg(seg_csv)
        print(f"Seg CSV  : {seg_csv.name}  ({len(seg)} frames)")
    else:
        print("[warn] No seg.csv found — using whole-frame mask (run run_sam2.py first)")

    mask_dir_arg = mask_dir if mask_dir.exists() else None

    # Effective FPS
    cap     = cv2.VideoCapture(str(video_path))
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps_eff = fps_raw / args.stride

    all_jpgs = sorted(frames_dir.glob("*.jpg"))
    print(f"Frames   : {len(all_jpgs)} extracted  |  {fps_eff:.1f} fps effective")
    print(f"Device   : {DEVICE}")

    # Load RAFT-small
    print("Loading RAFT-small...")
    weights    = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model      = raft_small(weights=weights).to(DEVICE).eval()

    # Pass 1 -- contraction signal
    print("\nPass 1: computing contraction signal...")
    frame_indices, timestamps, signal = compute_contraction_signal(
        frames_dir, seg, mask_dir_arg, fps_eff, args.stride, transforms, model
    )

    # Save CSV
    cont_csv = OUTPUTS_DIR / f"{stem}_contraction.csv"
    with open(cont_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_idx", "timestamp_s", "contraction"])
        for fi, ts, s in zip(frame_indices, timestamps, signal):
            wr.writerow([fi, f"{ts:.4f}", f"{s:.6f}"])
    print(f"Contraction CSV: {cont_csv}")

    # Peak detection
    peak_idxs, props = detect_peaks(signal, fps_eff, args.min_distance, args.prominence)
    peak_raw_frames  = frame_indices[peak_idxs] if len(peak_idxs) else np.array([], int)

    peaks_csv = OUTPUTS_DIR / f"{stem}_peaks.csv"
    with open(peaks_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["peak_id", "frame_idx", "timestamp_s", "contraction", "prominence"])
        for pid, (pi, pf) in enumerate(zip(peak_idxs, peak_raw_frames)):
            prom = props["prominences"][pid] if "prominences" in props else float("nan")
            wr.writerow([pid, pf, f"{timestamps[pi]:.4f}",
                         f"{signal[pi]:.6f}", f"{prom:.6f}"])
    print(f"Peaks CSV ({len(peak_idxs)} peaks): {peaks_csv}")

    # Plot
    plot_path = OUTPUTS_DIR / f"{stem}_contraction_plot.png"
    plot_contraction(timestamps, signal, peak_idxs, plot_path, fps_eff)

    # Pass 2 -- save divergence fields near each peak
    if not args.skip_divfields and len(peak_raw_frames):
        div_dir = OUTPUTS_DIR / f"{stem}_peak_divfields"
        print(f"\nPass 2: saving divergence windows for {len(peak_raw_frames)} peaks...")
        save_peak_divfields(
            frames_dir, peak_raw_frames, div_dir,
            seg, mask_dir_arg, args.stride, transforms, model,
            args.pre_window,
        )
        print(f"Divergence fields: {div_dir}/")
    elif args.skip_divfields:
        print("Pass 2 skipped (--skip-divfields).")
    else:
        print("No peaks detected — pass 2 skipped.")

    print("\nDone.")
    print(f"  Signal : {cont_csv}")
    print(f"  Peaks  : {peaks_csv}  ({len(peak_idxs)} events)")
    print(f"  Plot   : {plot_path}")


if __name__ == "__main__":
    main()
