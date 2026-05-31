"""
scripts/visualize_initiation.py

Visualises how RAFT determines the pulse initiation site for one pulse.

Two outputs per selected pulse:

  1. <stem>_pulse_<id>_breakdown.png
     A publication-quality multi-panel figure showing:
       Row A  — raw frame at each sampled pre-window timestep
       Row B  — divergence heatmap  (blue = contraction, red = expansion)
       Row C  — overlay (raw + heatmap)
     Plus a summary panel: cumulative early-window divergence with the
     identified initiation site (crosshair) and matched rhopalium (dot).

  2. <stem>_pulse_<id>_divergence.mp4
     Frame-by-frame animation of the divergence heatmap overlaid on the raw
     video, playing through the pre-window so you can watch the wave spread.

Usage:
    venv\Scripts\python scripts\visualize_initiation.py
    venv\Scripts\python scripts\visualize_initiation.py --pulse-id 3
    venv\Scripts\python scripts\visualize_initiation.py --all-pulses
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
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUTS_DIR, FPS

CALIB_DIR = Path(__file__).parent.parent / "calibration"

# Divergence colormap: blue = contraction (negative), red = expansion (positive)
DIV_CMAP = plt.cm.RdBu_r


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_initiation_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_calibration(calib_dir: Path) -> dict | None:
    jsons = sorted(calib_dir.glob("*.json"))
    return json.loads(jsons[0].read_text()) if jsons else None


def load_divfields(div_dir: Path, peak_frame: int) -> list[tuple[int, np.ndarray]]:
    """
    Returns [(frames_before_peak, divergence_array), ...] sorted from
    earliest (largest k) to latest (k=0 = peak).
    """
    files = sorted(
        div_dir.glob(f"peak_{peak_frame:06d}_minus*.npy"),
        key=lambda p: -int(p.stem.split("minus")[1]),
    )
    return [(int(p.stem.split("minus")[1]), np.load(str(p))) for p in files]


def load_raw_frame(frames_dir: Path, peak_frame: int,
                   frames_before: int, stride: int) -> np.ndarray | None:
    """Load the raw video frame corresponding to (peak - frames_before)."""
    target_raw   = peak_frame - frames_before * stride
    target_extracted = target_raw // stride
    all_jpgs     = sorted(frames_dir.glob("*.jpg"))
    if target_extracted < 0 or target_extracted >= len(all_jpgs):
        return None
    bgr = cv2.imread(str(all_jpgs[target_extracted]))
    return bgr[:, :, ::-1].copy() if bgr is not None else None  # BGR -> RGB


# ── Rendering helpers ─────────────────────────────────────────────────────────

def div_to_rgba(div: np.ndarray, vmax: float) -> np.ndarray:
    """Convert divergence array to RGBA image using the diverging colormap."""
    normed = np.clip(div / (vmax + 1e-8), -1, 1) * 0.5 + 0.5   # 0..1
    rgba   = (DIV_CMAP(normed) * 255).astype(np.uint8)
    return rgba   # (H, W, 4)


def overlay_div_on_frame(frame_rgb: np.ndarray, div: np.ndarray,
                         vmax: float, alpha: float = 0.55) -> np.ndarray:
    """Blend divergence heatmap over raw frame (both uint8 RGB)."""
    rgba   = div_to_rgba(div, vmax)
    heatmap_rgb = rgba[:, :, :3]
    blended = (frame_rgb.astype(np.float32) * (1 - alpha)
               + heatmap_rgb.astype(np.float32) * alpha).clip(0, 255)
    return blended.astype(np.uint8)


def draw_initiation_marker(ax, ix: int, iy: int, label: str = "",
                           color: str = "orange", size: int = 14) -> None:
    ax.plot(ix, iy, "+", color=color, markersize=size, markeredgewidth=2.5)
    ax.plot(ix, iy, "o", color=color, markersize=size // 2,
            markerfacecolor="none", markeredgewidth=2)
    if label:
        ax.text(ix + size, iy - size, label, color=color, fontsize=8,
                fontweight="bold")


def draw_rhopalium_marker(ax, rx: int, ry: int, rhop_id: int) -> None:
    ax.plot(rx, ry, "s", color="#4A9EFF", markersize=10,
            markerfacecolor="#4A9EFF", markeredgewidth=1.5,
            markeredgecolor="white")
    ax.text(rx + 8, ry + 8, f"R{rhop_id}", color="#4A9EFF", fontsize=8,
            fontweight="bold")


def rhopalium_video_px(result: dict, calib: dict,
                       cx: float, cy: float, r: float,
                       dye_x: float, dye_y: float) -> tuple[int, int] | None:
    rid = result["rhopalium_id"]
    if int(rid) < 0:
        return None
    phi_rho = next((rr["phi_body_deg"] for rr in calib["rhopalia"]
                    if rr["id"] == int(rid)), None)
    if phi_rho is None:
        return None
    phi_dye_lab = math.atan2(dye_y - cy, dye_x - cx)
    phi_rho_lab = phi_dye_lab + math.radians(float(phi_rho))
    return (round(cx + r * math.cos(phi_rho_lab)),
            round(cy + r * math.sin(phi_rho_lab)))


# ── Static breakdown panel ────────────────────────────────────────────────────

def make_breakdown_figure(
    result:      dict,
    divfields:   list[tuple[int, np.ndarray]],   # [(k, div), ...] early→peak
    frames_dir:  Path,
    calib:       dict | None,
    seg_row:     tuple[float, float, float],      # (cx, cy, r)
    dye_row:     tuple[float, float] | None,
    stride:      int,
    out_path:    Path,
) -> None:
    """
    Build the multi-row breakdown figure.

    Columns: sample frames spanning the pre-window  +  one summary panel
    Rows:    A) raw frame   B) divergence heatmap   C) overlay
    """
    # Sample ~5 evenly-spaced frames from the pre-window
    n_total   = len(divfields)
    n_cols    = min(5, n_total)
    sample_ks = [divfields[i][0]
                 for i in np.linspace(0, n_total - 1, n_cols, dtype=int)]
    sampled   = [(k, d) for k, d in divfields if k in sample_ks]

    # Include the peak frame (k=0) if not already sampled
    if divfields[-1][0] == 0 and 0 not in sample_ks:
        sampled.append(divfields[-1])
    sampled.sort(key=lambda x: -x[0])   # earliest first

    n_panels = len(sampled)
    vmax = max(np.abs(d).max() for _, d in divfields) or 1.0

    # Cumulative early divergence (first half of pre-window)
    early     = divfields[: max(1, n_total // 2)]
    cum_early = sum(np.clip(-d, 0, None) for _, d in early)

    peak_frame = int(result["peak_frame"])
    ix, iy     = int(result["init_x"]), int(result["init_y"])
    cx, cy, r  = seg_row

    fig_w = max(12, n_panels * 2.5 + 2.5)
    fig, axes = plt.subplots(3, n_panels + 1, figsize=(fig_w, 8))
    fig.patch.set_facecolor("#111111")

    def style_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # ── Row labels ────────────────────────────────────────────────────────────
    row_labels = ["Raw frame", "Divergence\n(blue=contract)", "Overlay"]
    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, color="white", fontsize=9,
                                rotation=90, labelpad=4)

    for col, (k, div) in enumerate(sampled):
        raw = load_raw_frame(frames_dir, peak_frame, k, stride)
        if raw is None:
            for row in range(3):
                axes[row, col].set_visible(False)
            continue

        # Row A — raw frame
        ax = axes[0, col]
        ax.imshow(raw)
        ax.set_title(f"t* − {k} fr" if k > 0 else "t* (peak)",
                     color="white", fontsize=8, pad=3)
        draw_initiation_marker(ax, ix, iy)
        if dye_row:
            ax.plot(dye_row[0], dye_row[1], "o", color="#44DD66",
                    markersize=6, markerfacecolor="#44DD66")
        ax.plot(cx, cy, "o", color="#DD4444", markersize=6)
        style_ax(ax)

        # Row B — divergence heatmap
        ax = axes[1, col]
        ax.imshow(raw, alpha=0.15, cmap="gray")
        im = ax.imshow(div, cmap=DIV_CMAP, vmin=-vmax, vmax=vmax, alpha=0.85)
        draw_initiation_marker(ax, ix, iy, color="orange")
        # Bell circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                "w--", lw=0.8, alpha=0.5)
        style_ax(ax)

        # Row C — overlay
        ax = axes[2, col]
        ax.imshow(overlay_div_on_frame(raw, div, vmax))
        draw_initiation_marker(ax, ix, iy, color="orange")
        if calib and dye_row:
            rp = rhopalium_video_px(result, calib, cx, cy, r,
                                    dye_row[0], dye_row[1])
            if rp:
                draw_rhopalium_marker(ax, rp[0], rp[1],
                                      int(result["rhopalium_id"]))
        style_ax(ax)

    # ── Summary panel (rightmost column) ─────────────────────────────────────
    import scipy.ndimage as ndi
    smoothed_early = ndi.gaussian_filter(cum_early.astype(np.float32), sigma=5.0)

    ax_sum_top = axes[0, n_panels]
    ax_sum_top.imshow(smoothed_early, cmap="Blues", vmin=0)
    ax_sum_top.set_title("Cumulative\nearly-window\ncontraction",
                         color="white", fontsize=8, pad=3)
    draw_initiation_marker(ax_sum_top, ix, iy, label="init", color="orange")
    ax_sum_top.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                    "w--", lw=0.8, alpha=0.6)
    style_ax(ax_sum_top)

    # Middle: what the algorithm actually finds (argmax of cum_early)
    ax_sum_mid = axes[1, n_panels]
    peak_div = divfields[-1][1]
    ax_sum_mid.imshow(peak_div, cmap=DIV_CMAP, vmin=-vmax, vmax=vmax)
    ax_sum_mid.set_title("Peak\ndivergence\n(t*)",
                         color="white", fontsize=8, pad=3)
    draw_initiation_marker(ax_sum_mid, ix, iy, color="orange")
    style_ax(ax_sum_mid)

    # Bottom: text summary
    ax_txt = axes[2, n_panels]
    ax_txt.set_facecolor("#1a1a1a")
    style_ax(ax_txt)
    rid      = int(result["rhopalium_id"])
    phi      = float(result["phi_origin_body"])
    dist     = float(result["angular_dist_deg"])
    ts       = float(result["timestamp_s"])
    summary  = (
        f"Pulse {result['peak_id']}\n\n"
        f"t = {ts:.2f} s\n"
        f"peak frame: {peak_frame}\n\n"
        f"Init site:\n  ({ix}, {iy})\n"
        f"  phi_body = {phi:.2f}°\n\n"
        f"Matched:\n  R{rid}\n"
        f"  dist = {dist:.2f}°\n\n"
        f"{'CONFIDENT' if dist < 11 else 'UNCERTAIN'}"
    )
    color = "#66FF66" if dist < 11 else "#FF9966"
    ax_txt.text(0.5, 0.5, summary, color=color, fontsize=8.5,
                ha="center", va="center", transform=ax_txt.transAxes,
                fontfamily="monospace",
                bbox=dict(facecolor="#222222", edgecolor="#444444",
                          boxstyle="round,pad=0.5"))

    # Colorbar
    cbar_ax = fig.add_axes([0.02, 0.05, 0.01, 0.25])
    norm    = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    sm      = plt.cm.ScalarMappable(cmap=DIV_CMAP, norm=norm)
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("div(flow)", color="white", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)

    fig.suptitle(
        f"Pulse initiation breakdown — Pulse {result['peak_id']}  "
        f"t={ts:.2f}s  R{rid}  phi={phi:.2f}°  dist={dist:.2f}°",
        color="white", fontsize=10, y=0.99,
    )
    plt.tight_layout(rect=[0.03, 0.0, 1.0, 0.98])
    fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Breakdown figure: {out_path.name}")


# ── Divergence animation ──────────────────────────────────────────────────────

def make_divergence_animation(
    result:     dict,
    divfields:  list[tuple[int, np.ndarray]],
    frames_dir: Path,
    calib:      dict | None,
    seg_row:    tuple[float, float, float],
    dye_row:    tuple[float, float] | None,
    stride:     int,
    fps_out:    float,
    out_path:   Path,
) -> None:
    """
    MP4 animation: each frame shows raw video + divergence heatmap overlay,
    playing from the earliest pre-window frame through to the peak.
    """
    if not divfields:
        return

    peak_frame = int(result["peak_frame"])
    ix, iy     = int(result["init_x"]), int(result["init_y"])
    cx, cy, r  = seg_row
    vmax = max(np.abs(d).max() for _, d in divfields) or 1.0

    # Get frame size from first raw frame
    sample_raw = load_raw_frame(frames_dir, peak_frame, divfields[0][0], stride)
    if sample_raw is None:
        print(f"  [warn] no raw frames found for animation")
        return
    h, w = sample_raw.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (w, h))

    theta = np.linspace(0, 2 * np.pi, 300)
    bell_pts_x = (cx + r * np.cos(theta)).astype(int)
    bell_pts_y = (cy + r * np.sin(theta)).astype(int)

    for k, div in divfields:
        raw = load_raw_frame(frames_dir, peak_frame, k, stride)
        if raw is None:
            continue

        overlay = overlay_div_on_frame(raw, div, vmax, alpha=0.6)

        # Convert RGB -> BGR for OpenCV
        frame_bgr = overlay[:, :, ::-1].copy()

        # Bell circle
        for bx, by in zip(bell_pts_x, bell_pts_y):
            if 0 <= bx < w and 0 <= by < h:
                cv2.circle(frame_bgr, (bx, by), 1, (200, 200, 200), -1)

        # Initiation site
        cv2.drawMarker(frame_bgr, (ix, iy), (0, 140, 255),
                       cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, (ix, iy), 12, (0, 140, 255), 2, cv2.LINE_AA)

        # Matched rhopalium
        if calib and dye_row:
            rp = rhopalium_video_px(result, calib, cx, cy, r,
                                    dye_row[0], dye_row[1])
            if rp:
                cv2.circle(frame_bgr, rp, 12, (255, 100, 50), 2, cv2.LINE_AA)
                cv2.putText(frame_bgr,
                            f"R{result['rhopalium_id']}",
                            (rp[0] + 14, rp[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 50),
                            1, cv2.LINE_AA)

        # HUD
        label = (f"t*-{k:02d}  div_range [{div.min():.3f},{div.max():.3f}]  "
                 f"Pulse {result['peak_id']}  R{result['rhopalium_id']}  "
                 f"phi={result['phi_origin_body']}deg")
        cv2.rectangle(frame_bgr, (0, 0), (w, 26), (10, 10, 10), -1)
        cv2.putText(frame_bgr, label, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210),
                    1, cv2.LINE_AA)

        writer.write(frame_bgr)

    writer.release()
    print(f"  Animation: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualise RAFT-based initiation detection for one or more pulses")
    ap.add_argument("--video",    default="data/test_clip_1min.mp4")
    ap.add_argument("--pulse-id", type=int, default=0,
                    help="Which pulse to visualise (default: 0 = first)")
    ap.add_argument("--all-pulses", action="store_true",
                    help="Generate figures for every pulse in initiation.csv")
    ap.add_argument("--stride",   type=int, default=1,
                    help="Frame stride used during RAFT (default 1)")
    ap.add_argument("--anim-fps", type=float, default=8.0,
                    help="Frame rate for the divergence animation (default 8)")
    ap.add_argument("--no-anim",  action="store_true",
                    help="Skip animation, generate static figure only")
    args = ap.parse_args()

    root       = Path(__file__).parent.parent
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = root / video_path
    stem = video_path.stem

    init_csv  = OUTPUTS_DIR / f"{stem}_initiation.csv"
    div_dir   = OUTPUTS_DIR / f"{stem}_peak_divfields"
    seg_csv   = OUTPUTS_DIR / f"{stem}_seg.csv"
    dye_csv   = OUTPUTS_DIR / f"{stem}_track.csv"
    frames_dir = OUTPUTS_DIR / f"{stem}_frames"

    for p, name in [(init_csv, "initiation.csv"), (div_dir, "peak_divfields"),
                    (frames_dir, "frames dir")]:
        if not p.exists():
            sys.exit(f"Not found: {p}  (run preceding stages first — {name})")

    # Load supporting data
    results = {int(r["peak_id"]): r
               for r in csv.DictReader(open(init_csv))}

    seg: dict[int, tuple] = {}
    if seg_csv.exists():
        for row in csv.DictReader(open(seg_csv)):
            seg[int(row["frame_idx"])] = (
                float(row["cx"]), float(row["cy"]), float(row["radius_px"]))

    dye: dict[int, tuple] = {}
    if dye_csv.exists():
        for row in csv.DictReader(open(dye_csv)):
            dye[int(row["frame_idx"])] = (float(row["x"]), float(row["y"]))

    calib = load_calibration(CALIB_DIR)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    pulse_ids = list(results.keys()) if args.all_pulses else [args.pulse_id]

    for pid in pulse_ids:
        if pid not in results:
            print(f"[warn] Pulse {pid} not in initiation.csv — skipping")
            continue

        result     = results[pid]
        peak_frame = int(result["peak_frame"])
        print(f"\nPulse {pid}  frame={peak_frame}  "
              f"R{result['rhopalium_id']}  phi={result['phi_origin_body']}°")

        divfields = load_divfields(div_dir, peak_frame)
        if not divfields:
            print(f"  [warn] no divfields found for peak {peak_frame}")
            continue

        # Nearest seg / dye entries for this peak
        def nearest(d, k):
            if not d:
                return None
            return d.get(k) or d[min(d, key=lambda x: abs(x - k))]

        seg_row = nearest(seg, peak_frame) or (320, 256, 100)
        dye_row = nearest(dye, peak_frame)

        # Outputs
        bd_out   = OUTPUTS_DIR / f"{stem}_pulse_{pid:03d}_breakdown.png"
        anim_out = OUTPUTS_DIR / f"{stem}_pulse_{pid:03d}_divergence.mp4"

        make_breakdown_figure(
            result, divfields, frames_dir, calib, seg_row, dye_row,
            args.stride, bd_out,
        )
        if not args.no_anim:
            make_divergence_animation(
                result, divfields, frames_dir, calib, seg_row, dye_row,
                args.stride, args.anim_fps, anim_out,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
