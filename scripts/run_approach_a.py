"""
scripts/run_approach_a.py

Approach A -- Bell contour radial tracking for pulse initiation detection.

Instead of RAFT optical flow, this uses the SAM2 bell boundary directly.
For each angle θ around the centroid, r(θ, t) is the distance to the bell
edge.  During a pulse, r(θ, t) decreases first at the initiating rhopalium's
angle before spreading around the bell.

Pipeline
---------
  1. Load contour_radii.npy  (N_frames × 360, output of run_sam2.py)
  2. Convert to body frame: rotate each frame's radii by -φ_dye(t) so
     angle 0 always points toward the dye mark.
  3. Detect pulses: peaks in -mean_r(t)  (mean radius decrease = contraction)
  4. Per pulse: find the earliest angle to show significant radius decrease
     in the pre-window → that is the initiation angle.
  5. Match initiation angle to nearest rhopalium from calibration.json.

Outputs (in outputs/)
----------------------
  <stem>_contour_pulses.csv        -- detected pulse events (frame, time, depth)
  <stem>_initiation_a.csv          -- per-pulse rhopalium assignment + confidence
  <stem>_initiation_a_plot.png     -- firing histogram + timeline
  <stem>_spacetime_pulse_<id>.png  -- angle × time heatmap per pulse (key figure)
  <stem>_initiation_a_annotated.mp4 -- full video with pulse labels

Usage
-----
  venv\Scripts\python scripts\run_approach_a.py
  venv\Scripts\python scripts\run_approach_a.py --pre-window 30
  venv\Scripts\python scripts\run_approach_a.py --pulse-id 0  # single pulse spacetime
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
PRE_WINDOW = 30     # frames before peak to analyse for initiation
BASELINE_W = 15     # frames before pre-window used as resting baseline
MIN_PULSE_DISTANCE_S = 0.42   # minimum seconds between pulses

# overlay colours (BGR)
C_CENTROID = (0,   0, 210)
C_DYE      = (0, 220,  50)
C_AXIS     = (0, 180,  60)
C_INIT     = (0, 100, 255)
C_RHOP     = (210, 80,   0)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_seg(path: Path) -> dict[int, tuple[float, float, float]]:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["frame_idx"])] = (
                float(row["cx"]), float(row["cy"]), float(row["radius_px"]))
    return out


def load_dye(path: Path) -> dict[int, tuple[float, float]]:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["frame_idx"])] = (float(row["x"]), float(row["y"]))
    return out


def nearest(d: dict, k: int):
    if k in d:
        return d[k]
    return d[min(d, key=lambda x: abs(x - k))]


def load_calibration(calib_dir: Path) -> dict | None:
    jsons = sorted(calib_dir.glob("*.json"))
    return json.loads(jsons[0].read_text()) if jsons else None


# ── Body-frame conversion ─────────────────────────────────────────────────────

def phi_dye_deg(frame_idx: int, seg: dict, dye: dict) -> float | None:
    """Lab-frame angle of the dye mark from the centroid (degrees)."""
    s = nearest(seg, frame_idx)
    d = nearest(dye, frame_idx) if dye else None
    if d is None or s[2] == 0:
        return None
    cx, cy, _ = s
    dx, dy = d
    return math.degrees(math.atan2(dy - cy, dx - cx))


def to_body_frame(contour: np.ndarray, phi_deg: float) -> np.ndarray:
    """
    Rotate contour (360,) so index 0 = dye-mark direction.
    phi_deg is the lab-frame angle of the dye mark.
    """
    shift = round(phi_deg * N_ANGLES / 360) % N_ANGLES
    return np.roll(contour, -shift)


# ── Pulse detection ───────────────────────────────────────────────────────────

def detect_pulses(mean_r: np.ndarray, fps_eff: float,
                  min_dist_s: float = MIN_PULSE_DISTANCE_S,
                  prominence_frac: float = 0.05
                  ) -> tuple[np.ndarray, dict]:
    """Detect peaks in the NEGATIVE mean radius (= contraction events)."""
    min_dist   = max(1, round(min_dist_s * fps_eff))
    signal     = -mean_r
    prom       = (signal.max() - signal.min()) * prominence_frac
    peaks, props = scipy.signal.find_peaks(signal, distance=min_dist,
                                           prominence=prom)
    return peaks, props


# ── Initiation detection ──────────────────────────────────────────────────────

def find_initiation_angle(
    contour_body: np.ndarray,    # (N_frames, 360) body-frame radii
    peak_idx:     int,
    pre_window:   int,
    baseline_w:   int,
) -> tuple[int, float, bool]:
    """
    Returns (init_angle_deg, delta_at_init, confident).

    Algorithm:
      1. Baseline: mean r(θ) over frames just before the pre-window.
      2. For each frame in the pre-window, compute delta(θ) = r(θ,t) - baseline(θ).
      3. Smooth delta over the angle axis (wrap-mode Gaussian) to suppress noise.
      4. Scan frames from earliest to latest; the first frame where ANY angle
         shows a decrease > threshold is the initiation frame.
      5. The angle of maximum decrease in that frame = initiation angle.
    """
    n = len(contour_body)
    bs_start = max(0, peak_idx - pre_window - baseline_w)
    bs_end   = max(0, peak_idx - pre_window)
    pw_start = max(0, peak_idx - pre_window)
    pw_end   = min(n, peak_idx + 1)

    if bs_end <= bs_start or pw_end <= pw_start:
        return 0, 0.0, False

    baseline = contour_body[bs_start:bs_end].mean(axis=0)   # (360,)
    window   = contour_body[pw_start:pw_end]                 # (T, 360)

    delta = window - baseline   # negative = contraction

    # Smooth over angle axis (circular)
    smooth = scipy.ndimage.gaussian_filter1d(delta, sigma=5, axis=1, mode='wrap')

    # Confidence: is there meaningful signal at all?
    range_val = smooth.min()  # most negative = strongest contraction
    confident = range_val < -0.5  # at least 0.5px radius decrease

    # Find earliest frame where some angle shows significant decrease
    threshold = range_val * 0.25   # 25% of max contraction
    init_frame_offset = pre_window  # default: peak itself
    for t, row in enumerate(smooth):
        if row.min() < threshold:
            init_frame_offset = t
            break

    init_angle = int(smooth[init_frame_offset].argmin())
    delta_val  = float(smooth[init_frame_offset, init_angle])
    return init_angle, delta_val, confident


# ── Rhopalium matching ────────────────────────────────────────────────────────

MAX_ASSIGNMENT_DIST = 11.25   # half of 360/16 — beyond this the match is ambiguous

def match_rhopalium(init_angle_deg: int, calib: dict) -> tuple[int, float]:
    """
    Shortest-arc distance on the circle between init_angle_deg [0, 360) and
    each rhopalium phi_body_deg (-180, 180].  Both are normalised to [0, 360)
    before subtraction so the result is always a positive value in [0, 180].
    """
    phi_init = init_angle_deg % 360
    best_id, best_dist = -1, float("inf")
    for r in calib["rhopalia"]:
        phi_rho = r["phi_body_deg"] % 360    # normalise (-180,180] → [0,360)
        diff    = abs(phi_init - phi_rho)
        diff    = min(diff, 360 - diff)       # shortest arc, always ≤ 180
        if diff < best_dist:
            best_dist = diff
            best_id   = r["id"]
    return best_id, best_dist


# ── Space-time plot (key figure) ──────────────────────────────────────────────

def spacetime_plot(
    contour_body: np.ndarray,    # (N_frames, 360)
    peak_idx:     int,
    init_angle:   int,
    result:       dict,
    calib:        dict,
    fps_eff:      float,
    pre_window:   int,
    baseline_w:   int,
    out_path:     Path,
) -> None:
    """
    Angle × time heatmap showing the bell radius change around a pulse.

    Colour = delta r(θ, t) = r(θ,t) - baseline r(θ).
    Blue = contraction (radius decreasing), red = expansion.
    """
    n = len(contour_body)
    bs_start = max(0, peak_idx - pre_window - baseline_w)
    bs_end   = max(0, peak_idx - pre_window)
    pw_start = max(0, peak_idx - pre_window)
    pw_end   = min(n, peak_idx + 1)

    baseline = contour_body[bs_start:bs_end].mean(axis=0)
    window   = contour_body[pw_start:pw_end]
    delta    = window - baseline   # (T, 360)

    smooth = scipy.ndimage.gaussian_filter1d(delta, sigma=3, axis=1, mode='wrap')

    times = np.arange(-pre_window, 1) / fps_eff * 1000  # ms relative to peak

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    vmax = max(abs(smooth.min()), abs(smooth.max()), 0.5)
    im = ax.imshow(
        smooth.T,   # (360 angles, T time) so angles on Y axis
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, 360],
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        interpolation="bilinear",
    )

    # Rhopalium positions as horizontal dashed lines
    for rr in calib["rhopalia"]:
        phi = rr["phi_body_deg"] % 360
        label = f"R{rr['id']}" if abs(phi - init_angle) < 30 else ""
        ax.axhline(phi, color="#AAAAAA", lw=0.6, linestyle="--", alpha=0.5)
        if label:
            ax.text(times[-1] + 1, phi, label, color="#AAAAAA",
                    fontsize=7, va="center")

    # Initiation site marker
    ax.axhline(init_angle, color="orange", lw=1.5, linestyle="-", alpha=0.9)
    ax.axvline(0, color="white", lw=0.8, linestyle="--", alpha=0.5,
               label="peak contraction")

    # Find initiation time marker
    threshold = smooth.min() * 0.25
    for t_off, row in enumerate(smooth):
        if row.min() < threshold:
            t_ms = times[t_off]
            ax.axvline(t_ms, color="orange", lw=1.2, linestyle=":",
                       alpha=0.8, label=f"initiation (t={t_ms:.0f}ms)")
            break

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Delta r (px)  [blue=contraction]", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    ax.set_xlabel("Time relative to peak (ms)", color="white")
    ax.set_ylabel("Body-frame angle (°, 0° = dye mark)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    rid   = result["rhopalium_id"]
    dist  = result["angular_dist_deg"]
    conf  = "CONFIDENT" if result["signal_confident"] else "LOW SIGNAL"
    title = (f"Pulse {result['peak_id']}  |  t={result['timestamp_s']:.2f}s  |  "
             f"Initiation: {init_angle}° body-frame  |  "
             f"Matched R{rid} (dist={dist:.1f}°)  |  {conf}")
    ax.set_title(title, color="white", fontsize=9, pad=8)
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
            if fi not in pulse_index or abs(fi-peak) < abs(fi-pulse_index[fi]["peak_frame"]):
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

            # Base: centroid + dye + axis
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

            # Pulse overlay
            pulse = pulse_index.get(raw_idx)
            if pulse is not None:
                peak  = pulse["peak_frame"]
                fade  = max(0.4, 1.0 - abs(raw_idx - peak) / (show_window + 1))

                # Initiation site on bell margin
                ia_rad = math.radians(pulse["init_angle_deg"])
                if dye_pos:
                    phi_dye_r = math.atan2(dye_pos[1]-cy, dye_pos[0]-cx)
                    phi_lab   = phi_dye_r + ia_rad
                    ix = round(cx + r_bell * math.cos(phi_lab))
                    iy = round(cy + r_bell * math.sin(phi_lab))
                    overlay = frame.copy()
                    cv2.circle(overlay, (ix, iy), 10, C_INIT, -1, cv2.LINE_AA)
                    cv2.circle(overlay, (ix, iy), 12, (255,255,255), 2, cv2.LINE_AA)

                    # Matched rhopalium
                    rid = pulse["rhopalium_id"]
                    phi_rho = next((rr["phi_body_deg"] for rr in calib["rhopalia"]
                                    if rr["id"] == rid), None)
                    if phi_rho is not None:
                        phi_rho_lab = phi_dye_r + math.radians(phi_rho)
                        rx = round(cx + r_bell * math.cos(phi_rho_lab))
                        ry = round(cy + r_bell * math.sin(phi_rho_lab))
                        cv2.circle(overlay, (rx, ry), 10, C_RHOP, -1, cv2.LINE_AA)
                        cv2.circle(overlay, (rx, ry), 12, (255,255,255), 2, cv2.LINE_AA)
                        cv2.line(overlay, (ix, iy), (rx, ry), (200,200,200), 1, cv2.LINE_AA)

                    cv2.addWeighted(overlay, fade, frame, 1 - fade, 0, frame)

                rid  = pulse["rhopalium_id"]
                dist = pulse["angular_dist_deg"]
                info = (f"PULSE {pulse['peak_id']}  R{rid}  "
                        f"init={pulse['init_angle_deg']}d  dist={dist:.1f}d  "
                        f"{'OK' if pulse['signal_confident'] else 'low-sig'}")
                cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
                cv2.putText(frame, info, (8, 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (220, 220, 220), 1, cv2.LINE_AA)

            writer.write(frame)
            raw_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()


# ── Summary plot ──────────────────────────────────────────────────────────────

def summary_plot(results: list[dict], calib: dict, out_path: Path) -> None:
    valid = [r for r in results if r["rhopalium_id"] >= 0 and r["signal_confident"]]
    if not valid:
        print("[warn] No confident pulse assignments for summary plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#111111")

    counts = {}
    rho_phis = {rr["id"]: rr["phi_body_deg"] for rr in calib["rhopalia"]}
    for r in valid:
        counts[r["rhopalium_id"]] = counts.get(r["rhopalium_id"], 0) + 1

    ax = axes[0]
    ax.set_facecolor("#1a1a1a")
    ids    = sorted(counts)
    labels = [f"R{i}\n{rho_phis.get(i,0):+.0f}°" for i in ids]
    bars   = ax.bar(range(len(ids)), [counts[i] for i in ids],
                    color="#4A9EFF", edgecolor="#222222")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(labels, fontsize=8, color="white")
    ax.set_ylabel("Confident pulse count", color="white")
    ax.set_title("Pulses initiated per rhopalium", color="white")
    ax.tick_params(colors="white")
    ax.grid(axis="y", alpha=0.2)

    ax2 = axes[1]
    ax2.set_facecolor("#1a1a1a")
    cmap = plt.cm.tab20
    for r in valid:
        ax2.scatter(r["timestamp_s"], r["rhopalium_id"],
                    color=cmap(r["rhopalium_id"] % 20), s=60, zorder=3)
    ax2.set_xlabel("Time (s)", color="white")
    ax2.set_ylabel("Rhopalium ID", color="white")
    ax2.set_title("Pulse initiation timeline (confident only)", color="white")
    ax2.tick_params(colors="white")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Approach A -- Bell contour radial tracking")
    ap.add_argument("--video",      default="data/test_clip_1min.mp4")
    ap.add_argument("--stride",     type=int, default=1,
                    help="Frame stride used when running SAM2 (default 1)")
    ap.add_argument("--pre-window", type=int, default=PRE_WINDOW,
                    help=f"Frames before peak to scan for initiation (default {PRE_WINDOW})")
    ap.add_argument("--min-distance", type=float, default=MIN_PULSE_DISTANCE_S,
                    help="Min seconds between pulses (default 0.42)")
    ap.add_argument("--prominence", type=float, default=0.05,
                    help="Peak prominence fraction (default 0.05)")
    ap.add_argument("--no-video",   action="store_true")
    ap.add_argument("--no-spacetime", action="store_true",
                    help="Skip per-pulse space-time plots")
    args = ap.parse_args()

    root       = Path(__file__).parent.parent
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = root / video_path
    stem = video_path.stem

    # Required inputs
    contour_npy = OUTPUTS_DIR / f"{stem}_contour_radii.npy"
    seg_csv     = OUTPUTS_DIR / f"{stem}_seg.csv"
    dye_csv     = OUTPUTS_DIR / f"{stem}_track.csv"

    for p, name in [(contour_npy, "contour_radii.npy (run run_sam2.py first)"),
                    (seg_csv,     "seg.csv")]:
        if not p.exists():
            sys.exit(f"Not found: {p}  --  {name}")

    contour_radii = np.load(str(contour_npy))          # (N, 360) lab frame
    seg           = load_seg(seg_csv)
    dye_track     = load_dye(dye_csv) if dye_csv.exists() else {}
    calib         = load_calibration(CALIB_DIR)

    if calib is None:
        sys.exit(f"No calibration JSON in {CALIB_DIR} -- run calibrate_rhopalia.py first.")

    cap = cv2.VideoCapture(str(video_path))
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps_eff = fps_raw / args.stride

    n_frames = len(contour_radii)
    # Frame indices in the contour array correspond to extracted frames
    frame_indices = np.arange(n_frames) * args.stride

    print(f"Video    : {video_path.name}  ({fps_raw:.0f} fps)")
    print(f"Contour  : {contour_radii.shape}  ({contour_npy.stat().st_size//1024} KB)")
    print(f"Seg      : {len(seg)} frames   Dye: {len(dye_track)} frames")
    print(f"Calib    : {calib['n_rhopalia']} rhopalia")

    # ── Convert to body frame ─────────────────────────────────────────────────
    print("\nConverting to body frame...")
    contour_body = np.zeros_like(contour_radii)
    for i, fi in enumerate(tqdm(frame_indices, desc="Body-frame", unit="fr")):
        phi = phi_dye_deg(int(fi), seg, dye_track)
        if phi is not None:
            contour_body[i] = to_body_frame(contour_radii[i], phi)
        else:
            contour_body[i] = contour_radii[i]

    # ── Mean radius signal + pulse detection ──────────────────────────────────
    mean_r = contour_body.mean(axis=1)   # (N,) — zero-radius frames pull it down
    # Replace zero-radius frames with local median to avoid spurious dips
    zero_mask = (contour_radii.max(axis=1) < 1.0)
    if zero_mask.any():
        mean_r[zero_mask] = np.median(mean_r[~zero_mask])

    peaks, props = detect_pulses(mean_r, fps_eff,
                                 args.min_distance, args.prominence)
    print(f"Detected : {len(peaks)} pulse events")

    # ── Per-pulse initiation ──────────────────────────────────────────────────
    results = []
    for pid, peak_idx in enumerate(tqdm(peaks, desc="Initiation", unit="pulse")):
        peak_frame = int(frame_indices[peak_idx])
        ts         = peak_frame / fps_raw

        # Adaptive pre-window: don't overlap previous peak
        if pid > 0:
            gap = int(peaks[pid] - peaks[pid-1])
            pre_w = min(args.pre_window, gap // 2)
        else:
            pre_w = args.pre_window

        init_angle, delta_val, confident = find_initiation_angle(
            contour_body, peak_idx, pre_w, BASELINE_W)

        rhop_id, rhop_dist = match_rhopalium(init_angle, calib)

        results.append({
            "peak_id":          pid,
            "peak_frame":       peak_frame,
            "timestamp_s":      round(ts, 4),
            "mean_r_drop_px":   round(float(-props["prominences"][pid]), 3),
            "init_angle_deg":   init_angle,
            "rhopalium_id":     rhop_id,
            "angular_dist_deg": round(rhop_dist, 2),
            "signal_confident": int(confident and rhop_dist <= MAX_ASSIGNMENT_DIST),
        })

        if confident:
            print(f"  Pulse {pid:3d}  f{peak_frame:5d}  t={ts:.2f}s  "
                  f"init={init_angle}°  R{rhop_id}  dist={rhop_dist:.1f}°")
        else:
            print(f"  Pulse {pid:3d}  f{peak_frame:5d}  t={ts:.2f}s  "
                  f"[low signal — skipping confident assignment]")

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Contraction signal CSV
    cont_csv = OUTPUTS_DIR / f"{stem}_contour_pulses.csv"
    with open(cont_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp_s", "mean_radius_px"])
        for i, fi in enumerate(frame_indices):
            w.writerow([fi, f"{fi/fps_raw:.4f}", f"{mean_r[i]:.3f}"])
    print(f"\nContraction CSV : {cont_csv}")

    # Contraction signal plot (mean radius over time with pulse markers)
    sig_plot = OUTPUTS_DIR / f"{stem}_contraction_a_plot.png"
    timestamps = frame_indices / fps_raw
    confident_peaks = [r for r in results if r["signal_confident"]]
    unconfident_peaks = [r for r in results if not r["signal_confident"]]

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.plot(timestamps, mean_r, lw=0.8, color="#4A9EFF", label="mean bell radius")
    ax.axhline(mean_r[mean_r > 0].mean(), lw=0.5, color="grey",
               linestyle="--", alpha=0.5, label="baseline")
    if confident_peaks:
        ax.scatter([r["timestamp_s"] for r in confident_peaks],
                   [mean_r[peaks[r["peak_id"]]] for r in confident_peaks],
                   color="lime", s=40, zorder=5,
                   label=f"confident ({len(confident_peaks)})")
    if unconfident_peaks:
        ax.scatter([r["timestamp_s"] for r in unconfident_peaks],
                   [mean_r[peaks[r["peak_id"]]] for r in unconfident_peaks],
                   color="orange", s=25, marker="x", zorder=5,
                   label=f"uncertain — dist>{MAX_ASSIGNMENT_DIST:.0f}° ({len(unconfident_peaks)})")
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Mean bell radius (px)", color="white")
    ax.set_title(f"Bell contraction signal — {len(results)} pulses detected  "
                 f"({len(confident_peaks)} confident)", color="white")
    ax.tick_params(colors="white")
    ax.legend(fontsize=8, labelcolor="white",
              facecolor="#222222", edgecolor="#555555")
    ax.grid(True, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    plt.tight_layout()
    fig.savefig(str(sig_plot), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Signal plot     : {sig_plot}")

    # Initiation CSV
    init_csv = OUTPUTS_DIR / f"{stem}_initiation_a.csv"
    with open(init_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        w.writeheader()
        w.writerows(results)
    print(f"Initiation CSV  : {init_csv}")

    # Summary plot
    plot_out = OUTPUTS_DIR / f"{stem}_initiation_a_plot.png"
    summary_plot(results, calib, plot_out)
    print(f"Summary plot    : {plot_out}")

    # Per-pulse space-time plots
    if not args.no_spacetime:
        st_dir = OUTPUTS_DIR / f"{stem}_spacetime_pulse_a"
        st_dir.mkdir(exist_ok=True)
        print(f"\nGenerating space-time plots -> {st_dir}/")
        for result in tqdm(results, desc="Space-time plots", unit="pulse"):
            pid       = result["peak_id"]
            peak_idx  = peaks[pid]
            pre_w     = min(args.pre_window,
                            (peaks[pid] - peaks[pid-1]) // 2 if pid > 0 else args.pre_window)
            st_out = st_dir / f"pulse_{pid:03d}.png"
            spacetime_plot(
                contour_body, peak_idx,
                result["init_angle_deg"], result, calib,
                fps_eff, pre_w, BASELINE_W, st_out,
            )

    # Annotated video
    if not args.no_video and results:
        vid_out = OUTPUTS_DIR / f"{stem}_initiation_a_annotated.mp4"
        print(f"\nRendering annotated video -> {vid_out}")
        render_annotated_video(
            results, calib, video_path, vid_out,
            seg, dye_track, fps_raw, args.stride,
        )

    confident_count = sum(r["signal_confident"] for r in results)
    print(f"\nDone.  {confident_count}/{len(results)} pulses with confident assignment.")
    print(f"  Contraction signal : {cont_csv}")
    print(f"  Initiation CSV     : {init_csv}")


if __name__ == "__main__":
    main()
