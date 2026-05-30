"""
scripts/validate_tracking.py

Validation tool — overlays both tracked points on the source video.

Reads:
    *_track.csv  (CoTracker)  -- dye mark position per frame
    *_seg.csv    (SAM2)       -- bell centroid + radius per frame

Outputs:
    <stem>_annotated.mp4      -- full video with both points overlaid
    <stem>_validation.png     -- static mosaic of N sampled frames

Overlay legend:
    red dot + white circle  = bell centroid + equiv. radius (SAM2)
    green dot               = dye mark, visible (CoTracker)
    purple dot              = dye mark, low confidence (CoTracker)
    green line              = body axis (centroid -> dye)
    HUD                     = frame, timestamp, body-axis angle phi_dye

Usage:
    venv\Scripts\python scripts\validate_tracking.py
    venv\Scripts\python scripts\validate_tracking.py --no-video   # mosaic only
    venv\Scripts\python scripts\validate_tracking.py --no-mosaic  # video only
    venv\Scripts\python scripts\validate_tracking.py --stride 4   # faster render
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUTS_DIR, FPS

MOSAIC_PANELS   = 8      # frames sampled in the static mosaic
MOSAIC_PANEL_SZ = 512    # px per panel (square)
TRAIL_LEN       = 60     # frames of dye trail in the video

COLOR_CENTROID  = (  0,   0, 220)   # BGR red
COLOR_DYE_VIS  = (  0, 255,  50)   # BGR green
COLOR_DYE_OCC  = ( 80,  80, 200)   # BGR purple
COLOR_AXIS     = (  0, 200,  80)   # BGR green line
COLOR_BELL     = (220, 220, 220)   # BGR white circle
COLOR_TRAIL    = (  0, 160, 220)   # BGR amber trail


# ── CSV loaders ───────────────────────────────────────────────────────────────

def load_dye_csv(path: Path) -> dict[int, tuple[float, float, bool]]:
    """Returns {raw_frame_idx: (x, y, visible)} from CoTracker CSV."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["frame_idx"])] = (
                float(row["x"]), float(row["y"]), bool(int(row["visible"]))
            )
    return out


def load_seg_csv(path: Path) -> dict[int, tuple[float, float, float]]:
    """Returns {raw_frame_idx: (cx, cy, radius)} from SAM2 seg CSV."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["frame_idx"])] = (
                float(row["cx"]), float(row["cy"]), float(row["radius_px"])
            )
    return out


# ── Per-frame overlay ─────────────────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    raw_idx: int,
    fps: float,
    dye_track: dict,
    seg_track: dict,
    dye_trail: list[tuple[float, float]],   # recent (x,y) positions
) -> np.ndarray:
    """Draw centroid, bell circle, dye mark, body axis and HUD onto frame."""
    out = frame.copy()

    seg = seg_track.get(raw_idx)
    dye = dye_track.get(raw_idx)

    # Bell circle + centroid
    if seg is not None:
        cx, cy, r = seg
        if r > 0:
            cv2.circle(out, (round(cx), round(cy)), round(r),
                       COLOR_BELL, 1, cv2.LINE_AA)
        cv2.circle(out, (round(cx), round(cy)), 5, COLOR_CENTROID, -1, cv2.LINE_AA)
        cv2.circle(out, (round(cx), round(cy)), 6, (255, 255, 255), 1, cv2.LINE_AA)

    # Dye trail
    for i, (tx, ty) in enumerate(dye_trail):
        alpha = i / max(len(dye_trail), 1)
        c = tuple(round(v * alpha) for v in COLOR_TRAIL)
        cv2.circle(out, (round(tx), round(ty)), 1, c, -1)

    # Dye mark + body axis
    if dye is not None:
        dx, dy_, vis = dye
        dye_col = COLOR_DYE_VIS if vis else COLOR_DYE_OCC
        cv2.circle(out, (round(dx), round(dy_)), 4, dye_col, -1, cv2.LINE_AA)
        cv2.circle(out, (round(dx), round(dy_)), 5, (255, 255, 255), 1, cv2.LINE_AA)

        if seg is not None:
            cx, cy, _ = seg
            cv2.line(out, (round(cx), round(cy)), (round(dx), round(dy_)),
                     COLOR_AXIS, 1, cv2.LINE_AA)
            phi = math.degrees(math.atan2(dy_ - cy, dx - cx))
        else:
            phi = float("nan")
    else:
        phi = float("nan")

    # HUD
    ts = raw_idx / fps
    phi_str = f"{phi:+.1f}deg" if not math.isnan(phi) else "no dye"
    hud = (f"f{raw_idx:05d}  t={ts:.2f}s  phi={phi_str}  "
           + ("VIS" if (dye and dye[2]) else "occl"))
    cv2.putText(out, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (220, 220, 220), 1, cv2.LINE_AA)

    return out


# ── Annotated video ───────────────────────────────────────────────────────────

def render_annotated_video(
    video_path: Path,
    out_path: Path,
    dye_track: dict,
    seg_track: dict,
    fps: float,
    stride: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps / stride, (w, h))

    dye_trail: list[tuple[float, float]] = []
    raw_idx   = 0

    with tqdm(total=(total + stride - 1) // stride,
              desc="Rendering video", unit="fr") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if raw_idx % stride != 0:
                raw_idx += 1
                continue

            # Update trail
            dye = dye_track.get(raw_idx)
            if dye is not None:
                dye_trail.append((dye[0], dye[1]))
                if len(dye_trail) > TRAIL_LEN:
                    dye_trail.pop(0)

            annotated = draw_overlay(frame, raw_idx, fps,
                                     dye_track, seg_track, dye_trail)
            writer.write(annotated)
            raw_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()


# ── Static mosaic ─────────────────────────────────────────────────────────────

def _fit_panel(img: np.ndarray, size: int):
    h, w   = img.shape[:2]
    sc     = size / max(h, w)
    nh, nw = round(h * sc), round(w * sc)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0     = (size - nh) // 2
    x0     = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = cv2.resize(img, (nw, nh))
    return canvas, sc, x0, y0


def render_mosaic(
    video_path: Path,
    out_path: Path,
    dye_track: dict,
    seg_track: dict,
    fps: float,
    n_panels: int,
    panel_sz: int,
) -> None:
    # Sample frames that exist in at least the dye CSV
    all_idxs = sorted(dye_track.keys())
    if not all_idxs:
        print("[warn] No dye frames — skipping mosaic.")
        return
    sample_idxs = [all_idxs[i] for i in
                   np.linspace(0, len(all_idxs) - 1, n_panels, dtype=int)]

    cap    = cv2.VideoCapture(str(video_path))
    panels = []

    for raw_idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        annotated  = draw_overlay(frame, raw_idx, fps, dye_track, seg_track, [])
        panel, *_  = _fit_panel(annotated, panel_sz)

        # Label strip
        seg = seg_track.get(raw_idx)
        dye = dye_track.get(raw_idx)
        cx_str = f"c=({seg[0]:.0f},{seg[1]:.0f})" if seg else "c=n/a"
        dx_str = f"d=({dye[0]:.0f},{dye[1]:.0f})" if dye else "d=n/a"
        phi_str = ""
        if seg and dye:
            phi = math.degrees(math.atan2(dye[1] - seg[1], dye[0] - seg[0]))
            phi_str = f"  phi={phi:+.1f}d"
        strip = np.zeros((20, panel_sz, 3), dtype=np.uint8)
        cv2.putText(strip, f"f{raw_idx} t={raw_idx/fps:.1f}s  {cx_str}  {dx_str}{phi_str}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)
        panels.append(np.vstack([panel, strip]))

    cap.release()

    if not panels:
        print("[warn] No mosaic panels generated.")
        return

    ph = panels[0].shape[0]
    pw = panels[0].shape[1]
    n_cols = min(4, len(panels))
    n_rows = math.ceil(len(panels) / n_cols)
    grid   = np.zeros((n_rows * ph, n_cols * pw, 3), dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, n_cols)
        grid[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = p

    # Legend header
    has_seg = bool(seg_track)
    legend_items = [
        (COLOR_CENTROID, "centroid (SAM2)" if has_seg else "centroid (Hough est.)"),
        (COLOR_DYE_VIS,  "dye mark visible (CoTracker)"),
        (COLOR_DYE_OCC,  "dye mark low-conf"),
    ]
    header = np.zeros((28, grid.shape[1], 3), dtype=np.uint8)
    x = 8
    for color, text in legend_items:
        cv2.circle(header, (x + 5, 14), 5, color, -1)
        cv2.putText(header, text, (x + 14, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        x += len(text) * 8 + 30

    mosaic = np.vstack([header, grid])
    cv2.imwrite(str(out_path), mosaic)
    print(f"Mosaic saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Validate dye + centroid tracking")
    ap.add_argument("--video", default="data/test_clip_1min.mp4")
    ap.add_argument("--dye-csv",  default=None, help="CoTracker *_track.csv")
    ap.add_argument("--seg-csv",  default=None, help="SAM2 *_seg.csv")
    ap.add_argument("--stride",   type=int, default=4,
                    help="Frame stride for video render (default 4 = 30fps from 120fps)")
    ap.add_argument("--no-video",  action="store_true", help="Skip annotated video")
    ap.add_argument("--no-mosaic", action="store_true", help="Skip static mosaic")
    args = ap.parse_args()

    root       = Path(__file__).parent.parent
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = root / video_path
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    stem = video_path.stem
    cap  = cv2.VideoCapture(str(video_path))
    fps  = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Auto-find CSVs
    def _find(pattern, arg, label):
        if arg:
            p = Path(arg)
            if not p.is_absolute():
                p = root / p
            return p
        hits = sorted(OUTPUTS_DIR.glob(pattern))
        if not hits:
            print(f"[warn] No {label} found in outputs/ — skipping that overlay.")
            return None
        return hits[-1]

    dye_csv_path = _find(f"{stem}_track.csv", args.dye_csv, "CoTracker CSV")
    seg_csv_path = _find(f"{stem}_seg.csv",   args.seg_csv, "SAM2 seg CSV")

    dye_track = load_dye_csv(dye_csv_path) if dye_csv_path else {}
    seg_track = load_seg_csv(seg_csv_path) if seg_csv_path else {}

    print(f"Video      : {video_path.name}  ({fps:.0f} fps)")
    print(f"Dye CSV    : {dye_csv_path.name if dye_csv_path else 'none'}  "
          f"({len(dye_track)} frames)")
    print(f"Seg CSV    : {seg_csv_path.name if seg_csv_path else 'none'}  "
          f"({len(seg_track)} frames)")

    if not dye_track and not seg_track:
        sys.exit("No CSV data loaded — nothing to draw.")

    OUTPUTS_DIR.mkdir(exist_ok=True)

    if not args.no_video:
        vid_out = OUTPUTS_DIR / f"{stem}_annotated.mp4"
        print(f"\nRendering annotated video -> {vid_out}")
        render_annotated_video(video_path, vid_out, dye_track, seg_track,
                               fps, stride=args.stride)

    if not args.no_mosaic:
        mosaic_out = OUTPUTS_DIR / f"{stem}_validation.png"
        print(f"\nRendering mosaic -> {mosaic_out}")
        render_mosaic(video_path, mosaic_out, dye_track, seg_track,
                      fps, MOSAIC_PANELS, MOSAIC_PANEL_SZ)


if __name__ == "__main__":
    main()
