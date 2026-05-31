"""
scripts/run_stage6.py

Stage 6 -- Pulse initiation analysis.

For each detected contraction peak, identifies WHICH rhopalium initiated
the pulse by finding where the contractile wave first appeared in the
pre-pulse divergence fields.

Algorithm (per peak)
---------------------
1. Determine a clean pre-window: min(PRE_WINDOW, half the gap to the previous
   peak).  This prevents overlap between consecutive pulses.

2. Load the saved divergence .npy files for that window.

3. Use only the FIRST HALF of the window (earliest frames, where the wave is
   still spatially localised) to locate the initiation site.
   Summing over the earliest frames suppresses noise while staying in the
   phase where contraction has not yet spread across the whole bell.

4. Spatial argmax of the early-phase sum = initiation pixel (ix, iy).

5. Convert to body-frame angle:
       phi_origin = atan2(iy - cy, ix - cx) - phi_dye(t*)

6. Match to the nearest rhopalium from calibration.json.

Outputs (in outputs/)
----------------------
  <stem>_initiation.csv         -- per-pulse: peak_frame, rhopalium_id, phi_body, dist
  <stem>_initiation_plot.png    -- polar histogram + timeline of which rhopalium fires
  <stem>_initiation_clips/      -- short annotated video clips around each pulse

Usage
-----
  venv\Scripts\python scripts\run_stage6.py
  venv\Scripts\python scripts\run_stage6.py --video data/test_clip_1min.mp4
  venv\Scripts\python scripts\run_stage6.py --no-clips   (skip video clips)
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUTS_DIR, FPS

PULSE_SHOW_WINDOW = 20   # frames either side of each peak to show pulse overlay
CALIB_DIR         = Path(__file__).parent.parent / "calibration"

# overlay colours (BGR)
C_INIT    = (  0,  50, 255)   # orange — initiation site
C_RHOP    = (210,  80,   0)   # blue — matched rhopalium
C_CENTROID = ( 0,   0, 210)   # red
C_DYE      = ( 0, 220,  50)   # green
C_AXIS     = ( 0, 180,  60)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_peaks(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_seg(path: Path) -> dict[int, tuple[float, float, float]]:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["frame_idx"])] = (
                float(row["cx"]), float(row["cy"]), float(row["radius_px"])
            )
    return out


def load_dye(path: Path) -> dict[int, tuple[float, float]]:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["frame_idx"])] = (float(row["x"]), float(row["y"]))
    return out


def load_calibration(calib_dir: Path, stem_hint: str = "") -> dict | None:
    """Auto-find calibration JSON. Prefers one matching stem_hint."""
    jsons = sorted(calib_dir.glob("*.json"))
    if not jsons:
        return None
    if stem_hint:
        matches = [j for j in jsons if stem_hint.lower() in j.stem.lower()]
        if matches:
            return json.loads(matches[0].read_text())
    return json.loads(jsons[0].read_text())


def nearest_seg(frame_idx: int, seg: dict) -> tuple[float, float, float]:
    """Interpolate seg to the nearest available frame."""
    if frame_idx in seg:
        return seg[frame_idx]
    nearest = min(seg.keys(), key=lambda k: abs(k - frame_idx))
    return seg[nearest]


def nearest_dye(frame_idx: int, dye: dict) -> tuple[float, float] | None:
    if not dye:
        return None
    if frame_idx in dye:
        return dye[frame_idx]
    nearest = min(dye.keys(), key=lambda k: abs(k - frame_idx))
    return dye[nearest]


# ── Divfield helpers ──────────────────────────────────────────────────────────

def load_divfields(div_dir: Path, peak_frame: int) -> list[np.ndarray]:
    """
    Load all saved divergence fields for a given peak, sorted from earliest
    (largest minus-k) to latest (minus00 = peak itself).
    """
    pattern = f"peak_{peak_frame:06d}_minus*.npy"
    files   = sorted(div_dir.glob(pattern),
                     key=lambda p: -int(p.stem.split("minus")[1]))
    return [np.load(str(f)) for f in files]


MIN_SIGNAL_FRACTION = 0.05   # cumsum peak must exceed this fraction of its own
                             # range to be considered real signal vs noise

# Annular margin where rhopalia are located (as fractions of bell radius)
MARGIN_INNER_FRAC = 0.70    # search from 70% of radius inward from margin
MARGIN_OUTER_FRAC = 1.00    # up to the bell edge (saved divfields are already masked)


def _margin_mask(shape: tuple, cx: float, cy: float, radius: float) -> np.ndarray:
    """
    Float32 mask selecting the annular ring [MARGIN_INNER_FRAC, MARGIN_OUTER_FRAC]
    of the bell radius.  Rhopalia are structurally at the bell margin so the
    initiation wave must appear here first — searching the whole interior
    introduces noise without adding information.
    """
    h, w  = shape
    ys, xs = np.ogrid[:h, :w]
    dist  = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return (
        (dist >= MARGIN_INNER_FRAC * radius) &
        (dist <= MARGIN_OUTER_FRAC * radius)
    ).astype(np.float32)


def find_initiation_site(
    divfields:    list[np.ndarray],
    cx:           float,
    cy:           float,
    radius:       float,
    smooth_sigma: float = 3.0,
    top_pct:      float = 0.05,
) -> tuple[tuple[int, int], bool] | None:
    """
    Returns ((ix, iy), confident) of the initiation site.

    Key design decisions:
    1. Use only the FIRST HALF of the pre-window (wave is still localised).
    2. Restrict search to the MARGIN ANNULUS — rhopalia are at the bell edge,
       so the initiation physically cannot start in the interior.  This removes
       the majority of RAFT noise from consideration.
    3. Gaussian-smooth before searching to suppress sub-pixel spikes.
    4. Centroid of the top `top_pct` region rather than single-pixel argmax.
    5. Report `confident=False` when peak is near the noise floor.
    """
    import scipy.ndimage as ndi

    if not divfields:
        return None

    h, w  = divfields[0].shape
    margin = _margin_mask((h, w), cx, cy, radius)

    n     = len(divfields)
    early = divfields[: max(1, n // 2)]
    cumsum = np.zeros((h, w), dtype=np.float32)
    for d in early:
        cumsum += np.clip(-d, 0, None) * margin   # restrict to margin only

    smoothed = ndi.gaussian_filter(cumsum, sigma=smooth_sigma)

    # Confidence: is the marginal peak meaningfully above the noise floor?
    s_min, s_max  = smoothed.min(), smoothed.max()
    peak_strength = s_max - s_min
    confident = peak_strength > MIN_SIGNAL_FRACTION * s_max if s_max > 0 else False

    # Centroid of the top top_pct fraction within the margin
    threshold = s_min + (1.0 - top_pct) * peak_strength
    region    = (smoothed >= threshold) & (margin > 0)
    ys, xs    = np.where(region)
    if len(xs) == 0:
        yx = np.unravel_index(np.argmax(smoothed * margin), smoothed.shape)
        return (int(yx[1]), int(yx[0])), confident

    ix = int(np.round(xs.mean()))
    iy = int(np.round(ys.mean()))
    return (ix, iy), confident


# ── Body-frame geometry ───────────────────────────────────────────────────────

def body_frame_angle(point: tuple[float, float],
                     centroid: tuple[float, float],
                     dye: tuple[float, float]) -> float:
    """
    phi_body = atan2(py - cy, px - cx) - atan2(dy - cy, dx - cx)
    Normalised to (-180, 180].
    """
    phi_p   = math.degrees(math.atan2(point[1] - centroid[1],
                                      point[0] - centroid[0]))
    phi_dye = math.degrees(math.atan2(dye[1] - centroid[1],
                                      dye[0] - centroid[0]))
    phi_body = phi_p - phi_dye
    return (phi_body + 180) % 360 - 180


def nearest_rhopalium(phi_body_deg: float, calib: dict) -> tuple[int, float]:
    """
    Returns (rhopalium_id, angular_distance_deg) for the closest rhopalium.
    Angular distance is the shortest arc on the circle.
    """
    best_id, best_dist = -1, float("inf")
    for r in calib["rhopalia"]:
        diff = abs(phi_body_deg - r["phi_body_deg"])
        diff = min(diff, 360 - diff)   # shortest arc
        if diff < best_dist:
            best_dist = diff
            best_id   = r["id"]
    return best_id, best_dist


# ── Per-pulse analysis ────────────────────────────────────────────────────────

def analyze_pulses(
    peaks:    list[dict],
    div_dir:  Path,
    seg:      dict,
    dye:      dict,
    calib:    dict,
) -> list[dict]:
    """
    Run initiation analysis for every peak.
    Returns list of result dicts, one per peak.
    """
    results = []
    peak_frames = [int(p["frame_idx"]) for p in peaks]

    for i, peak in enumerate(tqdm(peaks, desc="Analysing pulses", unit="pulse")):
        peak_frame = int(peak["frame_idx"])

        # Adaptive pre-window: don't overlap the previous peak
        if i > 0:
            gap = peak_frame - peak_frames[i - 1]
            effective_window_size = None   # determined from loaded files but capped
        else:
            gap = None

        divfields = load_divfields(div_dir, peak_frame)

        if gap is not None:
            cap = max(1, gap // 2)
            divfields = divfields[:cap]   # drop earliest frames if too close

        if not divfields:
            print(f"  [warn] peak {peak_frame}: no divergence fields found, skipping")
            continue

        cx, cy, radius = nearest_seg(peak_frame, seg)
        result_site = find_initiation_site(divfields, cx, cy, radius)
        if result_site is None:
            continue
        init_site, signal_confident = result_site
        dye_pos = nearest_dye(peak_frame, dye)

        if dye_pos is None:
            print(f"  [warn] peak {peak_frame}: no dye track — can't compute body-frame angle")
            phi_body = float("nan")
            rhop_id, rhop_dist = -1, float("nan")
        else:
            phi_body = body_frame_angle(init_site, (cx, cy), dye_pos)
            rhop_id, rhop_dist = nearest_rhopalium(phi_body, calib)

        if not signal_confident:
            print(f"  [low-signal] peak {peak_frame}: divergence near noise floor — "
                  f"initiation site unreliable")

        results.append({
            "peak_id":          int(peak["peak_id"]),
            "peak_frame":       peak_frame,
            "timestamp_s":      float(peak["timestamp_s"]),
            "contraction":      float(peak["contraction"]),
            "init_x":           init_site[0],
            "init_y":           init_site[1],
            "phi_origin_body":  round(phi_body, 2),
            "rhopalium_id":     rhop_id,
            "angular_dist_deg": round(rhop_dist, 2),
            "signal_confident": int(signal_confident),
        })

    return results


# ── Outputs ───────────────────────────────────────────────────────────────────

def save_initiation_csv(results: list[dict], out_path: Path) -> None:
    if not results:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Initiation CSV: {out_path}")


def plot_initiation(results: list[dict], calib: dict, out_path: Path) -> None:
    if not results:
        return

    valid = [r for r in results if r["rhopalium_id"] >= 0]
    if not valid:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: polar histogram of initiation angles ────────────────────────────
    ax = axes[0]
    angles = [r["phi_origin_body"] for r in valid]
    rho_labels = {r["id"]: r["phi_body_deg"] for r in calib["rhopalia"]}
    counts = {}
    for r in valid:
        counts[r["rhopalium_id"]] = counts.get(r["rhopalium_id"], 0) + 1

    ids    = sorted(counts.keys())
    labels = [f"R{i}\n({rho_labels.get(i, 0):+.0f}d)" for i in ids]
    bars   = ax.bar(range(len(ids)), [counts[i] for i in ids],
                    color="#4A9EFF", edgecolor="white")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Pulse count")
    ax.set_title("Pulses initiated per rhopalium")
    ax.grid(axis="y", alpha=0.3)
    for bar, rid in zip(bars, ids):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(counts[rid]), ha="center", va="bottom", fontsize=8)

    # ── Right: timeline — which rhopalium fires when ──────────────────────────
    ax2 = axes[1]
    cmap = plt.cm.tab20
    for r in valid:
        ax2.scatter(r["timestamp_s"], r["rhopalium_id"],
                    color=cmap(r["rhopalium_id"] % 20), s=60, zorder=3)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Rhopalium ID")
    ax2.set_title("Pulse initiation timeline")
    ax2.grid(True, alpha=0.3)
    ax2.set_yticks(sorted(set(r["rhopalium_id"] for r in valid)))

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Plot: {out_path}")


# ── Annotated full video ──────────────────────────────────────────────────────

def _build_pulse_index(results: list[dict], show_window: int) -> dict[int, dict]:
    """
    Map every video frame that falls within show_window of a pulse peak
    to that pulse's result dict.  Frames near multiple peaks get the
    nearest peak assigned.
    """
    index: dict[int, dict] = {}
    for r in results:
        peak = r["peak_frame"]
        for fi in range(peak - show_window, peak + show_window + 1):
            if fi not in index or abs(fi - peak) < abs(fi - index[fi]["peak_frame"]):
                index[fi] = r
    return index


def _rhopalium_video_px(result: dict, calib: dict,
                        cx: float, cy: float, r: float,
                        dye_pos: tuple) -> tuple[int, int] | None:
    """Project the matched rhopalium's body-frame angle onto the bell circle."""
    if result["rhopalium_id"] < 0 or dye_pos is None:
        return None
    phi_rho = next(
        (rr["phi_body_deg"] for rr in calib["rhopalia"]
         if rr["id"] == result["rhopalium_id"]),
        None,
    )
    if phi_rho is None:
        return None
    phi_dye_lab = math.atan2(dye_pos[1] - cy, dye_pos[0] - cx)
    phi_rho_lab = phi_dye_lab + math.radians(phi_rho)
    return (round(cx + r * math.cos(phi_rho_lab)),
            round(cy + r * math.sin(phi_rho_lab)))


def render_annotated_video(
    results:    list[dict],
    calib:      dict,
    video_path: Path,
    out_path:   Path,
    seg:        dict,
    dye_track:  dict,
    fps:        float,
    stride:     int,
    show_window: int,
) -> None:
    """
    Stream the source video once, writing a single annotated output video.

    Every frame gets:
        - Bell circle + centroid (red)
        - Dye mark (green) + body-axis arrow
        - Frame / timestamp HUD

    Frames within show_window of a pulse peak additionally get:
        - Orange filled circle at initiation site
        - Blue filled circle at matched rhopalium on the bell edge
        - Connecting line between them
        - Pulse info bar (pulse ID, rhopalium, phi_body, angular distance)
        - Fading opacity: full at peak, 50% at edges of show_window
    """
    pulse_index = _build_pulse_index(results, show_window)

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

            cx, cy, r = nearest_seg(raw_idx, seg)
            dye_pos   = nearest_dye(raw_idx, dye_track)

            # ── Base overlay (every frame) ─────────────────────────────────
            cv2.circle(frame, (round(cx), round(cy)), 5, C_CENTROID, -1, cv2.LINE_AA)
            if dye_pos:
                dx, dy = dye_pos
                cv2.circle(frame, (round(dx), round(dy)), 4, C_DYE, -1, cv2.LINE_AA)
                cv2.arrowedLine(frame,
                                (round(cx), round(cy)),
                                (round(dx), round(dy)),
                                C_AXIS, 1, cv2.LINE_AA, tipLength=0.15)

            # Base HUD
            ts = raw_idx / fps
            cv2.putText(frame, f"f{raw_idx:05d}  t={ts:.2f}s",
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (160, 160, 160), 1, cv2.LINE_AA)

            # ── Pulse overlay (near-peak frames only) ──────────────────────
            pulse = pulse_index.get(raw_idx)
            if pulse is not None:
                peak    = pulse["peak_frame"]
                fade    = 1.0 - abs(raw_idx - peak) / (show_window + 1)
                alpha   = max(0.4, fade)   # min 40% opacity at window edges

                ix, iy  = pulse["init_x"], pulse["init_y"]
                rhop_px = _rhopalium_video_px(pulse, calib, cx, cy, r, dye_pos)

                # Overlay layer for alpha blending
                overlay = frame.copy()
                cv2.circle(overlay, (ix, iy), 10, C_INIT, -1, cv2.LINE_AA)
                cv2.circle(overlay, (ix, iy), 12, (255, 255, 255), 2, cv2.LINE_AA)
                if rhop_px:
                    cv2.circle(overlay, rhop_px, 10, C_RHOP, -1, cv2.LINE_AA)
                    cv2.circle(overlay, rhop_px, 12, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.line(overlay, (ix, iy), rhop_px, (210, 210, 210), 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Pulse info bar — solid, always full opacity
                rid  = pulse["rhopalium_id"]
                info = (f"PULSE {pulse['peak_id']}  R{rid}  "
                        f"phi={pulse['phi_origin_body']:+.1f}d  "
                        f"dist={pulse['angular_dist_deg']:.1f}d  "
                        f"peak@f{peak}")
                cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
                cv2.putText(frame, info, (8, 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220),
                            1, cv2.LINE_AA)

            writer.write(frame)
            raw_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()


# ── Side-by-side comparison ───────────────────────────────────────────────────

def render_side_by_side(
    video_path: Path,
    annotated_path: Path,
    out_path: Path,
    fps: float,
    stride: int,
) -> None:
    """
    Combine the original video (downsampled to match stride) and the
    annotated video into a single side-by-side MP4.

    Left  = original (no overlays)
    Right = annotated (pulse labels, dye, centroid)

    Both sides are labelled with a header bar.
    """
    cap_orig = cv2.VideoCapture(str(video_path))
    cap_ann  = cv2.VideoCapture(str(annotated_path))

    total_ann = int(cap_ann.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_fps = fps / stride

    # Header bar height
    hdr = 28
    out_h = h + hdr
    out_w = w * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))

    raw_idx = 0
    with tqdm(total=total_ann, desc="Side-by-side render", unit="fr") as pbar:
        while True:
            ret_ann, frame_ann = cap_ann.read()
            if not ret_ann:
                break

            # Seek original to the matching raw frame
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
            ret_orig, frame_orig = cap_orig.read()
            if not ret_orig:
                break

            # Build combined frame
            combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)

            # Header labels
            cv2.rectangle(combined, (0, 0), (out_w, hdr), (25, 25, 25), -1)
            cv2.putText(combined, "Original",
                        (w // 2 - 35, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(combined, "Annotated (pulse initiation)",
                        (w + w // 2 - 110, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.line(combined, (w, 0), (w, out_h), (60, 60, 60), 1)

            # Video frames
            combined[hdr:, :w]  = frame_orig
            combined[hdr:, w:]  = frame_ann

            writer.write(combined)
            raw_idx += stride
            pbar.update(1)

    cap_orig.release()
    cap_ann.release()
    writer.release()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6 -- Pulse initiation analysis")
    ap.add_argument("--video", default="data/test_clip_1min.mp4")
    ap.add_argument("--stride", type=int, default=4,
                    help="Frame stride for annotated video render (default 4 = 30fps from 120fps)")
    ap.add_argument("--show-window", type=int, default=PULSE_SHOW_WINDOW,
                    help=f"Frames either side of each peak to show pulse overlay (default {PULSE_SHOW_WINDOW})")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip annotated video generation")
    ap.add_argument("--side-by-side", action="store_true",
                    help="Also generate a side-by-side comparison of original vs annotated")
    args = ap.parse_args()

    root       = Path(__file__).parent.parent
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = root / video_path

    stem = video_path.stem

    # Required inputs
    peaks_csv  = OUTPUTS_DIR / f"{stem}_peaks.csv"
    div_dir    = OUTPUTS_DIR / f"{stem}_peak_divfields"
    seg_csv    = OUTPUTS_DIR / f"{stem}_seg.csv"
    dye_csv    = OUTPUTS_DIR / f"{stem}_track.csv"

    for p, name in [(peaks_csv, "peaks CSV"), (div_dir, "divfields dir"),
                    (seg_csv,   "seg CSV")]:
        if not p.exists():
            sys.exit(f"Required file/dir not found: {p}\n"
                     f"  ({name} — run the preceding stages first)")

    peaks     = load_peaks(peaks_csv)
    seg       = load_seg(seg_csv)
    dye_track = load_dye(dye_csv) if dye_csv.exists() else {}
    calib     = load_calibration(CALIB_DIR, stem_hint=stem)

    if calib is None:
        sys.exit(f"No calibration JSON found in {CALIB_DIR}\n"
                 "  Run scripts/calibrate_rhopalia.py first.")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"Video    : {video_path.name}  ({fps:.0f} fps)")
    print(f"Peaks    : {len(peaks)}")
    print(f"Seg      : {len(seg)} frames")
    print(f"Dye      : {len(dye_track)} frames")
    print(f"Calib    : {calib['n_rhopalia']} rhopalia from {Path(calib['source_image']).name}")
    print()

    # Analyse
    results = analyze_pulses(peaks, div_dir, seg, dye_track, calib)

    valid = [r for r in results if r["rhopalium_id"] >= 0]
    print(f"\n{len(valid)}/{len(peaks)} pulses assigned to a rhopalium")

    if not results:
        sys.exit("No results — check that divfields exist and peaks CSV is not empty.")

    # Save outputs
    init_csv  = OUTPUTS_DIR / f"{stem}_initiation.csv"
    init_plot = OUTPUTS_DIR / f"{stem}_initiation_plot.png"
    vid_out   = OUTPUTS_DIR / f"{stem}_initiation_annotated.mp4"

    save_initiation_csv(results, init_csv)
    plot_initiation(results, calib, init_plot)

    sbs_out = OUTPUTS_DIR / f"{stem}_side_by_side.mp4"

    if not args.no_video:
        print(f"\nRendering annotated video -> {vid_out}")
        render_annotated_video(
            results, calib, video_path, vid_out,
            seg, dye_track, fps,
            stride=args.stride,
            show_window=args.show_window,
        )

        if args.side_by_side:
            print(f"Rendering side-by-side -> {sbs_out}")
            render_side_by_side(video_path, vid_out, sbs_out, fps, args.stride)

    print("\nDone.")
    print(f"  Initiation CSV  : {init_csv}")
    print(f"  Plot            : {init_plot}")
    if not args.no_video:
        print(f"  Annotated video : {vid_out}")
        if args.side_by_side:
            print(f"  Side-by-side    : {sbs_out}")


if __name__ == "__main__":
    main()
