"""
scripts/run_approach_b.py

Approach B -- Polar margin intensity difference for pulse initiation detection.

For each consecutive frame pair, a thin annular strip at the bell margin is
extracted via warpPolar and converted to body frame.  The frame-to-frame pixel
intensity difference in that strip, per angle, is the core signal.

Why this complements Approach A
--------------------------------
  Approach A measures WHERE the bell boundary is (structural signal).
  Approach B measures HOW MUCH each margin angle changes (appearance signal).
  B is sensitive to tissue deformation and the expansion crescent just outside
  the bell (inner_frac-outer_frac can exceed 1.0), which Approach A cannot see.
  The two approaches can be combined to improve confidence.

Two-phase execution
--------------------
  Phase 1  Compute and save margin_diff.npy   (expensive, ~1-3 min)
  Phase 2  Load margin_diff.npy, detect pulses, find initiation angles
           (fast, can be re-run instantly with different --pre-window etc.)

  Phase 1 is skipped automatically if margin_diff.npy already exists.
  Force recompute with --recompute.

Usage
-----
  venv\Scripts\python scripts\run_approach_b.py
  venv\Scripts\python scripts\run_approach_b.py --inner-frac 0.80
  venv\Scripts\python scripts\run_approach_b.py --no-video
  venv\Scripts\python scripts\run_approach_b.py --recompute
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUTS_DIR, FPS

CALIB_DIR  = Path(__file__).parent.parent / "calibration"
N_ANGLES   = 360
N_RADII    = 128
PRE_WINDOW = 30
BASELINE_W = 15
MIN_PULSE_DISTANCE_S  = 0.42
MAX_ASSIGNMENT_DIST   = 11.25   # half of 360/16

C_CENTROID = (0,   0, 210)
C_DYE      = (0, 220,  50)
C_AXIS     = (0, 180,  60)
C_INIT     = (0, 100, 255)
C_RHOP     = (210, 80,   0)


# ── Data helpers (shared with Approach A) ────────────────────────────────────

def load_seg(path: Path) -> dict[int, tuple]:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["frame_idx"])] = (
                float(row["cx"]), float(row["cy"]), float(row["radius_px"]))
    return out


def load_dye(path: Path) -> dict[int, tuple]:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["frame_idx"])] = (float(row["x"]), float(row["y"]))
    return out


def load_calibration(calib_dir: Path) -> dict | None:
    jsons = sorted(calib_dir.glob("*.json"))
    return json.loads(jsons[0].read_text()) if jsons else None


def nearest(d: dict, k: int):
    if k in d:
        return d[k]
    return d[min(d, key=lambda x: abs(x - k))]


def phi_dye_deg(frame_idx: int, seg: dict, dye: dict) -> float | None:
    s = nearest(seg, frame_idx)
    d = nearest(dye, frame_idx) if dye else None
    if d is None or s[2] == 0:
        return None
    cx, cy, _ = s
    dx, dy = d
    return math.degrees(math.atan2(dy - cy, dx - cx))


def to_body_frame(signal_1d: np.ndarray, phi_deg: float) -> np.ndarray:
    shift = round(phi_deg * N_ANGLES / 360) % N_ANGLES
    return np.roll(signal_1d, -shift)


def match_rhopalium(init_angle_deg: int, calib: dict) -> tuple[int, float]:
    phi_init = init_angle_deg % 360
    best_id, best_dist = -1, float("inf")
    for r in calib["rhopalia"]:
        phi_rho = r["phi_body_deg"] % 360
        diff    = abs(phi_init - phi_rho)
        diff    = min(diff, 360 - diff)
        if diff < best_dist:
            best_dist = diff
            best_id   = r["id"]
    return best_id, best_dist


# ── Phase 1: margin difference computation ────────────────────────────────────

def margin_strip(frame_gray: np.ndarray, cx: float, cy: float,
                 radius: float, inner_frac: float, outer_frac: float
                 ) -> np.ndarray:
    """
    Extract the annular margin strip via warpPolar and return the mean
    pixel intensity per angle (shape: (N_ANGLES,)).

    inner_frac and outer_frac are fractions of `radius`, so
    outer_frac > 1.0 captures the region OUTSIDE the bell boundary —
    this is where the expansion crescent appears (see user diagram).
    """
    max_r = radius * outer_frac
    polar = cv2.warpPolar(
        frame_gray.astype(np.float32),
        dsize=(N_RADII, N_ANGLES),          # (width=radii cols, height=angle rows)
        center=(cx, cy),
        maxRadius=max_r,
        flags=cv2.WARP_POLAR_LINEAR,
    )
    # polar: (N_ANGLES, N_RADII)
    # Extract the margin band
    col_inner = round(inner_frac / outer_frac * N_RADII)
    col_outer = N_RADII   # up to max_r = outer_frac * radius
    strip = polar[:, col_inner:col_outer]   # (N_ANGLES, band_width)
    return strip.mean(axis=1)              # (N_ANGLES,) — mean intensity per angle


def compute_margin_diff(
    frames_dir: Path,
    seg:        dict,
    dye:        dict,
    stride:     int,
    inner_frac: float,
    outer_frac: float,
) -> np.ndarray:
    """
    Phase 1: compute per-angle margin intensity difference for all frame pairs.

    Returns margin_diff: (N_pairs, N_ANGLES) float32, already in BODY FRAME.
    N_pairs = n_extracted_frames - 1.
    """
    all_jpgs = sorted(frames_dir.glob("*.jpg"))
    n = len(all_jpgs)
    if n < 2:
        sys.exit(f"Need at least 2 frames in {frames_dir}")

    result = np.zeros((n - 1, N_ANGLES), dtype=np.float32)

    # Pre-load first frame
    prev_gray = cv2.imread(str(all_jpgs[0]), cv2.IMREAD_GRAYSCALE)

    for i in tqdm(range(1, n), desc="Margin diff", unit="fr"):
        curr_gray = cv2.imread(str(all_jpgs[i]), cv2.IMREAD_GRAYSCALE)
        raw_idx   = i * stride

        cx, cy, radius = nearest(seg, raw_idx)
        if radius < 5:
            prev_gray = curr_gray
            continue

        strip_prev = margin_strip(prev_gray, cx, cy, radius, inner_frac, outer_frac)
        strip_curr = margin_strip(curr_gray, cx, cy, radius, inner_frac, outer_frac)
        diff = np.abs(strip_curr - strip_prev)   # (N_ANGLES,)

        # Convert to body frame
        phi = phi_dye_deg(raw_idx, seg, dye)
        if phi is not None:
            diff = to_body_frame(diff, phi)

        result[i - 1] = diff
        prev_gray = curr_gray

    return result


# ── Phase 2: pulse detection and initiation ───────────────────────────────────

def detect_pulses(total_activity: np.ndarray, fps_eff: float,
                  min_dist_s: float, prominence_frac: float
                  ) -> tuple[np.ndarray, dict]:
    min_dist = max(1, round(min_dist_s * fps_eff))
    prom     = (total_activity.max() - total_activity.min()) * prominence_frac
    return scipy.signal.find_peaks(total_activity, distance=min_dist,
                                   prominence=prom)


def find_initiation_angle(
    margin_diff: np.ndarray,   # (N_pairs, 360) body frame
    peak_idx:    int,
    pre_window:  int,
    baseline_w:  int,
) -> tuple[int, float, bool]:
    """
    Find the body-frame angle that first shows elevated margin activity
    in the frames leading up to the contraction peak.

    Returns (init_angle_deg, signal_at_init, confident).

    Unlike Approach A (which looks for the FIRST significant frame), here
    we use the EARLIEST frame where a localised angle peak emerges above
    the baseline noise.  We look for the angle that is consistently high
    in the early pre-window, not just a single-frame spike.
    """
    n = len(margin_diff)
    bs_start = max(0, peak_idx - pre_window - baseline_w)
    bs_end   = max(0, peak_idx - pre_window)
    pw_start = max(0, peak_idx - pre_window)
    pw_end   = min(n, peak_idx + 1)

    if bs_end <= bs_start or pw_end <= pw_start:
        return 0, 0.0, False

    baseline = margin_diff[bs_start:bs_end].mean(axis=0)    # (360,)
    window   = margin_diff[pw_start:pw_end]                  # (T, 360)

    # Excess activity above baseline noise
    excess = np.clip(window - baseline, 0, None)             # (T, 360)

    # Smooth over angle axis to suppress single-pixel noise
    smooth = scipy.ndimage.gaussian_filter1d(excess, sigma=5, axis=1, mode='wrap')

    # Confidence: is there meaningful excess signal?
    peak_excess = smooth.max()
    confident   = peak_excess > baseline.mean() * 0.5  # 50% above baseline mean

    threshold = peak_excess * 0.25
    init_frame_offset = len(smooth) - 1   # default: last frame (peak)
    for t, row in enumerate(smooth):
        if row.max() > threshold:
            init_frame_offset = t
            break

    init_angle = int(smooth[init_frame_offset].argmax())
    sig_val    = float(smooth[init_frame_offset, init_angle])
    return init_angle, sig_val, confident


# ── Space-time plot ───────────────────────────────────────────────────────────

def spacetime_plot(
    margin_diff: np.ndarray,
    peak_idx:    int,
    init_angle:  int,
    result:      dict,
    calib:       dict,
    fps_eff:     float,
    pre_window:  int,
    baseline_w:  int,
    out_path:    Path,
) -> None:
    n = len(margin_diff)
    bs_start = max(0, peak_idx - pre_window - baseline_w)
    bs_end   = max(0, peak_idx - pre_window)
    pw_start = max(0, peak_idx - pre_window)
    pw_end   = min(n, peak_idx + 1)

    baseline = margin_diff[bs_start:bs_end].mean(axis=0)
    window   = margin_diff[pw_start:pw_end]
    excess   = np.clip(window - baseline, 0, None)
    smooth   = scipy.ndimage.gaussian_filter1d(excess, sigma=3, axis=1, mode='wrap')

    times = np.arange(-pre_window, 1) / fps_eff * 1000   # ms

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    im = ax.imshow(
        smooth.T,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, 360],
        cmap="hot",
        vmin=0,
        interpolation="bilinear",
    )

    for rr in calib["rhopalia"]:
        phi   = rr["phi_body_deg"] % 360
        label = f"R{rr['id']}" if abs(phi - init_angle) < 30 else ""
        ax.axhline(phi, color="#6688AA", lw=0.5, linestyle="--", alpha=0.6)
        if label:
            ax.text(times[-1] + 1, phi, label, color="#AAAACC",
                    fontsize=7, va="center")

    ax.axhline(init_angle, color="cyan", lw=1.5, linestyle="-", alpha=0.9)
    ax.axvline(0, color="white", lw=0.8, linestyle="--", alpha=0.5,
               label="peak")

    threshold = smooth.max() * 0.25
    for t_off, row in enumerate(smooth):
        if row.max() > threshold:
            t_ms = times[t_off]
            ax.axvline(t_ms, color="cyan", lw=1.2, linestyle=":",
                       alpha=0.8, label=f"initiation (t={t_ms:.0f}ms)")
            break

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Excess margin activity (px intensity change)",
                   color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    ax.set_xlabel("Time relative to peak (ms)", color="white")
    ax.set_ylabel("Body-frame angle (°, 0° = dye mark)", color="white")
    ax.tick_params(colors="white")

    rid  = result["rhopalium_id"]
    dist = result["angular_dist_deg"]
    conf = "CONFIDENT" if result["signal_confident"] else "UNCERTAIN"
    ax.set_title(
        f"Approach B  Pulse {result['peak_id']}  t={result['timestamp_s']:.2f}s  "
        f"init={init_angle}°  R{rid} (dist={dist:.1f}°)  {conf}",
        color="white", fontsize=9, pad=8,
    )
    ax.legend(fontsize=8, labelcolor="white",
              facecolor="#222222", edgecolor="#555555")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Annotated video ───────────────────────────────────────────────────────────

def render_annotated_video(
    results:    list[dict],
    calib:      dict,
    video_path: Path,
    out_path:   Path,
    seg:        dict,
    dye_track:  dict,
    fps:        float,
    stride:     int,
    show_window: int = 20,
) -> None:
    pulse_index: dict[int, dict] = {}
    for r in results:
        peak = r["peak_frame"]
        for fi in range(peak - show_window, peak + show_window + 1):
            if fi not in pulse_index or abs(fi - peak) < abs(fi - pulse_index[fi]["peak_frame"]):
                pulse_index[fi] = r

    cap    = cv2.VideoCapture(str(video_path))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps / stride, (w, h))

    raw_idx = 0
    with tqdm(total=(total + stride - 1) // stride,
              desc="Rendering video", unit="fr") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if raw_idx % stride != 0:
                raw_idx += 1
                continue

            cx, cy, r_bell = nearest(seg, raw_idx)
            dye_pos = nearest(dye_track, raw_idx) if dye_track else None

            cv2.circle(frame, (round(cx), round(cy)), 5, C_CENTROID, -1, cv2.LINE_AA)
            if dye_pos:
                dx, dy = dye_pos
                cv2.circle(frame, (round(dx), round(dy)), 4, C_DYE, -1, cv2.LINE_AA)
                cv2.arrowedLine(frame, (round(cx), round(cy)),
                                (round(dx), round(dy)),
                                C_AXIS, 1, cv2.LINE_AA, tipLength=0.15)

            ts = raw_idx / fps
            cv2.putText(frame, f"f{raw_idx:05d}  t={ts:.2f}s",
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (160, 160, 160), 1, cv2.LINE_AA)

            pulse = pulse_index.get(raw_idx)
            if pulse is not None:
                peak = pulse["peak_frame"]
                fade = max(0.4, 1.0 - abs(raw_idx - peak) / (show_window + 1))
                ia_rad = math.radians(pulse["init_angle_deg"])
                if dye_pos:
                    phi_dye_r = math.atan2(dye_pos[1]-cy, dye_pos[0]-cx)
                    phi_lab   = phi_dye_r + ia_rad
                    ix = round(cx + r_bell * math.cos(phi_lab))
                    iy = round(cy + r_bell * math.sin(phi_lab))
                    rid = pulse["rhopalium_id"]
                    phi_rho = next((rr["phi_body_deg"] for rr in calib["rhopalia"]
                                    if rr["id"] == rid), None)
                    overlay = frame.copy()
                    cv2.circle(overlay, (ix, iy), 10, C_INIT, -1, cv2.LINE_AA)
                    cv2.circle(overlay, (ix, iy), 12, (255, 255, 255), 2, cv2.LINE_AA)
                    if phi_rho is not None:
                        phi_rho_lab = phi_dye_r + math.radians(phi_rho)
                        rx = round(cx + r_bell * math.cos(phi_rho_lab))
                        ry = round(cy + r_bell * math.sin(phi_rho_lab))
                        cv2.circle(overlay, (rx, ry), 10, C_RHOP, -1, cv2.LINE_AA)
                        cv2.circle(overlay, (rx, ry), 12, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.line(overlay, (ix, iy), (rx, ry), (200, 200, 200), 1)
                    cv2.addWeighted(overlay, fade, frame, 1 - fade, 0, frame)

                info = (f"B  Pulse {pulse['peak_id']}  R{pulse['rhopalium_id']}  "
                        f"init={pulse['init_angle_deg']}°  "
                        f"dist={pulse['angular_dist_deg']:.1f}°  "
                        f"{'OK' if pulse['signal_confident'] else 'low-sig'}")
                cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
                cv2.putText(frame, info, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (220, 220, 220), 1, cv2.LINE_AA)

            writer.write(frame)
            raw_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()


# ── Summary plot ──────────────────────────────────────────────────────────────

def summary_plot(results: list[dict], calib: dict,
                 total_activity: np.ndarray, peaks: np.ndarray,
                 frame_indices: np.ndarray, fps_raw: float,
                 out_path: Path) -> None:
    valid = [r for r in results if r["signal_confident"]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("#111111")

    # Top-left: total margin activity signal
    ax = axes[0, 0]
    ax.set_facecolor("#1a1a1a")
    ts = frame_indices[:-1] / fps_raw
    ax.plot(ts, total_activity, lw=0.7, color="#4A9EFF")
    conf_peaks  = [r for r in results if r["signal_confident"]]
    unconf      = [r for r in results if not r["signal_confident"]]
    if conf_peaks:
        ax.scatter([r["timestamp_s"] for r in conf_peaks],
                   [total_activity[peaks[r["peak_id"]]] for r in conf_peaks],
                   color="lime", s=40, zorder=5, label=f"confident ({len(conf_peaks)})")
    if unconf:
        ax.scatter([r["timestamp_s"] for r in unconf],
                   [total_activity[peaks[r["peak_id"]]] for r in unconf],
                   color="orange", s=25, marker="x", zorder=5,
                   label=f"uncertain ({len(unconf)})")
    ax.set_title("Total margin activity", color="white")
    ax.set_xlabel("Time (s)", color="white"); ax.set_ylabel("Activity", color="white")
    ax.tick_params(colors="white"); ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#222222")

    # Top-right: rhopalium firing histogram
    ax2 = axes[0, 1]
    ax2.set_facecolor("#1a1a1a")
    counts = {}
    rho_phis = {rr["id"]: rr["phi_body_deg"] for rr in calib["rhopalia"]}
    for r in valid:
        counts[r["rhopalium_id"]] = counts.get(r["rhopalium_id"], 0) + 1
    if counts:
        ids    = sorted(counts)
        labels = [f"R{i}\n{rho_phis.get(i,0):+.0f}°" for i in ids]
        ax2.bar(range(len(ids)), [counts[i] for i in ids],
                color="#4A9EFF", edgecolor="#222222")
        ax2.set_xticks(range(len(ids)))
        ax2.set_xticklabels(labels, fontsize=7, color="white")
    ax2.set_title("Pulses per rhopalium (confident only)", color="white")
    ax2.set_ylabel("Count", color="white"); ax2.tick_params(colors="white")
    ax2.grid(axis="y", alpha=0.2)

    # Bottom-left: initiation angle histogram
    ax3 = axes[1, 0]
    ax3.set_facecolor("#1a1a1a")
    angles = [r["init_angle_deg"] for r in valid]
    if angles:
        ax3.hist(angles, bins=36, range=(0, 360), color="#FF8844", edgecolor="#222222")
        for rr in calib["rhopalia"]:
            ax3.axvline(rr["phi_body_deg"] % 360, color="cyan",
                        lw=0.7, linestyle="--", alpha=0.6)
    ax3.set_title("Init angle distribution (cyan = rhopalium positions)",
                  color="white")
    ax3.set_xlabel("Body-frame angle (°)", color="white")
    ax3.set_ylabel("Count", color="white"); ax3.tick_params(colors="white")

    # Bottom-right: firing timeline
    ax4 = axes[1, 1]
    ax4.set_facecolor("#1a1a1a")
    cmap = plt.cm.tab20
    for r in valid:
        ax4.scatter(r["timestamp_s"], r["rhopalium_id"],
                    color=cmap(r["rhopalium_id"] % 20), s=50, zorder=3)
    ax4.set_xlabel("Time (s)", color="white"); ax4.set_ylabel("Rhopalium ID", color="white")
    ax4.set_title("Firing timeline", color="white")
    ax4.tick_params(colors="white"); ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Approach B -- Polar margin intensity difference")
    ap.add_argument("--video",       default="data/test_clip_1min.mp4")
    ap.add_argument("--stride",      type=int,   default=1)
    ap.add_argument("--inner-frac",  type=float, default=0.75,
                    help="Inner edge of margin ring as fraction of bell radius (default 0.75)")
    ap.add_argument("--outer-frac",  type=float, default=1.05,
                    help="Outer edge of margin ring (>1.0 includes outside bell, default 1.05)")
    ap.add_argument("--pre-window",  type=int,   default=PRE_WINDOW)
    ap.add_argument("--min-distance",type=float, default=MIN_PULSE_DISTANCE_S)
    ap.add_argument("--prominence",  type=float, default=0.05)
    ap.add_argument("--recompute",   action="store_true",
                    help="Recompute margin_diff.npy even if it already exists")
    ap.add_argument("--no-video",    action="store_true")
    ap.add_argument("--no-spacetime",action="store_true")
    args = ap.parse_args()

    root       = Path(__file__).parent.parent
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = root / video_path
    stem = video_path.stem

    seg_csv  = OUTPUTS_DIR / f"{stem}_seg.csv"
    dye_csv  = OUTPUTS_DIR / f"{stem}_track.csv"
    frames_dir = OUTPUTS_DIR / f"{stem}_frames"

    for p, name in [(seg_csv, "seg.csv"), (frames_dir, "frames dir")]:
        if not p.exists():
            sys.exit(f"Not found: {p}  ({name} — run run_sam2.py first)")

    seg       = load_seg(seg_csv)
    dye_track = load_dye(dye_csv) if dye_csv.exists() else {}
    calib     = load_calibration(CALIB_DIR)
    if calib is None:
        sys.exit(f"No calibration JSON in {CALIB_DIR}")

    cap     = cv2.VideoCapture(str(video_path))
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps_eff = fps_raw / args.stride

    all_jpgs     = sorted(frames_dir.glob("*.jpg"))
    n_extracted  = len(all_jpgs)
    frame_indices = np.arange(n_extracted) * args.stride

    print(f"Video      : {video_path.name}  ({fps_raw:.0f} fps)")
    print(f"Frames     : {n_extracted} extracted")
    print(f"Margin ring: {args.inner_frac:.2f} – {args.outer_frac:.2f} × radius")
    print(f"Calib      : {calib['n_rhopalia']} rhopalia")

    # ── Phase 1: compute margin_diff ─────────────────────────────────────────
    diff_npy = OUTPUTS_DIR / f"{stem}_margin_diff.npy"
    if diff_npy.exists() and not args.recompute:
        print(f"\nLoading cached margin_diff ({diff_npy.stat().st_size // 1024} KB)...")
        margin_diff = np.load(str(diff_npy))
    else:
        print("\nPhase 1: computing margin differences...")
        margin_diff = compute_margin_diff(
            frames_dir, seg, dye_track, args.stride,
            args.inner_frac, args.outer_frac,
        )
        np.save(str(diff_npy), margin_diff)
        print(f"Saved margin_diff: {diff_npy}  "
              f"({diff_npy.stat().st_size // 1024} KB)")

    # ── Phase 2: pulse detection ──────────────────────────────────────────────
    total_activity = margin_diff.sum(axis=1)   # (N_pairs,)
    peaks, props   = detect_pulses(total_activity, fps_eff,
                                   args.min_distance, args.prominence)
    print(f"\nDetected : {len(peaks)} pulse events")

    # ── Phase 3: initiation ───────────────────────────────────────────────────
    results = []
    for pid, peak_idx in enumerate(tqdm(peaks, desc="Initiation", unit="pulse")):
        peak_frame = int(frame_indices[min(peak_idx, n_extracted - 1)])
        ts         = peak_frame / fps_raw

        pre_w = args.pre_window
        if pid > 0:
            pre_w = min(pre_w, (peaks[pid] - peaks[pid - 1]) // 2)

        init_angle, sig_val, confident = find_initiation_angle(
            margin_diff, peak_idx, pre_w, BASELINE_W)
        rhop_id, rhop_dist = match_rhopalium(init_angle, calib)
        signal_confident   = confident and rhop_dist <= MAX_ASSIGNMENT_DIST

        results.append({
            "peak_id":          pid,
            "peak_frame":       peak_frame,
            "timestamp_s":      round(ts, 4),
            "activity":         round(float(total_activity[peak_idx]), 2),
            "init_angle_deg":   init_angle,
            "rhopalium_id":     rhop_id,
            "angular_dist_deg": round(rhop_dist, 2),
            "signal_confident": int(signal_confident),
        })
        flag = "OK" if signal_confident else "uncertain"
        print(f"  Pulse {pid:3d}  f{peak_frame:5d}  t={ts:.2f}s  "
              f"init={init_angle:3d}°  R{rhop_id}  dist={rhop_dist:.1f}°  {flag}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(exist_ok=True)

    init_csv = OUTPUTS_DIR / f"{stem}_initiation_b.csv"
    if results:
        with open(init_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    print(f"\nInitiation CSV  : {init_csv}")

    plot_out = OUTPUTS_DIR / f"{stem}_initiation_b_plot.png"
    summary_plot(results, calib, total_activity, peaks,
                 frame_indices, fps_raw, plot_out)
    print(f"Summary plot    : {plot_out}")

    if not args.no_spacetime:
        st_dir = OUTPUTS_DIR / f"{stem}_spacetime_pulse_b"
        st_dir.mkdir(exist_ok=True)
        print(f"\nGenerating space-time plots -> {st_dir}/")
        for result in tqdm(results, desc="Space-time", unit="pulse"):
            pid   = result["peak_id"]
            pre_w = min(args.pre_window,
                        (peaks[pid] - peaks[pid-1]) // 2 if pid > 0 else args.pre_window)
            st_out = st_dir / f"pulse_{pid:03d}.png"
            spacetime_plot(margin_diff, peaks[pid], result["init_angle_deg"],
                           result, calib, fps_eff, pre_w, BASELINE_W, st_out)

    if not args.no_video and results:
        vid_out = OUTPUTS_DIR / f"{stem}_initiation_b_annotated.mp4"
        print(f"\nRendering annotated video -> {vid_out}")
        render_annotated_video(results, calib, video_path, vid_out,
                               seg, dye_track, fps_raw, args.stride)

    confident = sum(r["signal_confident"] for r in results)
    print(f"\nDone.  {confident}/{len(results)} confident assignments.")


if __name__ == "__main__":
    main()
