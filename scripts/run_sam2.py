"""
scripts/run_sam2.py

Stage 1 — Bell segmentation using SAM2 video predictor.

Usage:
    venv\Scripts\python scripts\run_sam2.py
    venv\Scripts\python scripts\run_sam2.py --video data/test_clip_1min.mp4
    venv\Scripts\python scripts\run_sam2.py --stride 2  --window-size 300

Controls (click window):
    Left-click    -- place / re-place prompt point on the jellyfish bell
    ENTER / SPACE -- confirm and start segmentation
    ESC           -- quit

Why windowed processing?
    SAM2 caches image features in inference_state as it processes frames.
    For long videos this grows unbounded.  We reinitialise inference_state
    every WINDOW_SIZE frames (fresh feature cache), handing off the last mask
    as the new prompt.  Memory is therefore bounded regardless of video length.

Outputs (in outputs/):
    <stem>_frames/              -- extracted JPEG frames (reused on re-runs)
    <stem>_masks/               -- per-frame binary mask PNGs (0=bg, 255=bell)
    <stem>_seg.csv              -- frame_idx, timestamp_s, cx, cy, radius_px
    <stem>_sam2_validation.png  -- mosaic: raw frame | mask overlay | body-axis
"""

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAM2_WEIGHTS, OUTPUTS_DIR, FPS

SAM2_CFG      = "configs/sam2.1/sam2.1_hiera_b+.yaml"
WINDOW_SIZE   = 200    # frames per SAM2 window — bounds feature-cache RAM (~2.4 GB at 1024px)
DISPLAY_SCALE = 2      # click-UI zoom factor
N_VALID       = 8      # frames sampled in validation mosaic
PANEL_SZ      = 480    # pixels per mosaic panel (square)
MASK_COLOR    = (180, 120, 0)   # BGR tint for mask overlay (amber)


# ── Click UI ──────────────────────────────────────────────────────────────────

class ClickSelector:
    def __init__(self, frame_bgr: np.ndarray, scale: float):
        h, w = frame_bgr.shape[:2]
        self.scale = scale
        self._orig = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        self.point = None

    def _on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (round(x / self.scale), round(y / self.scale))

    def run(self) -> tuple[int, int] | None:
        title = "Click the jellyfish bell  --  ENTER/SPACE confirm  ESC quit"
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, self._on_mouse)
        while True:
            disp = self._orig.copy()
            if self.point:
                px = round(self.point[0] * self.scale)
                py = round(self.point[1] * self.scale)
                cv2.drawMarker(disp, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                cv2.circle(disp, (px, py), 12, (0, 255, 0), 2)
                cv2.putText(disp, f"({self.point[0]}, {self.point[1]})",
                            (px + 16, py - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            else:
                cv2.putText(disp, "Click the jellyfish bell",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.imshow(title, disp)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32) and self.point:
                break
            if key == 27:
                self.point = None
                break
        cv2.destroyAllWindows()
        return self.point


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: Path, out_dir: Path, stride: int) -> int:
    """
    Extract video frames to out_dir as zero-padded JPEGs required by SAM2.
    Skips extraction if directory already contains the expected number of files.
    Returns the number of extracted frames.
    """
    cap  = cv2.VideoCapture(str(video_path))
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected  = (total_raw + stride - 1) // stride

    existing = list(out_dir.glob("*.jpg")) if out_dir.exists() else []
    if len(existing) == expected:
        print(f"  Frame cache hit: {expected} JPEGs already in {out_dir.name}/")
        cap.release()
        return expected

    out_dir.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with tqdm(total=expected, desc="Extracting frames", unit="fr") as pbar:
        for raw_idx in range(total_raw):
            ret, frame = cap.read()
            if not ret:
                break
            if raw_idx % stride == 0:
                cv2.imwrite(
                    str(out_dir / f"{n_out:06d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                n_out += 1
                pbar.update(1)
    cap.release()
    return n_out


# ── Mask utilities ────────────────────────────────────────────────────────────

def mask_to_stats(mask_bool: np.ndarray) -> tuple[float, float, float]:
    """Return (cx, cy, equiv_radius) from a boolean mask using image moments."""
    m = cv2.moments(mask_bool.astype(np.uint8))
    if m["m00"] == 0:
        return 0.0, 0.0, 0.0
    cx     = m["m10"] / m["m00"]
    cy     = m["m01"] / m["m00"]
    radius = math.sqrt(m["m00"] / math.pi)
    return cx, cy, radius


# ── SAM2 windowed propagation ─────────────────────────────────────────────────

def run_sam2(
    predictor,
    frames_dir: Path,
    click_point: tuple[int, int],
    mask_dir: Path,
    total_frames: int,
    window_size: int,
    save_indices: set[int],
) -> list[tuple[float, float, float]]:
    """
    Propagate SAM2 through all frames using non-overlapping windows.

    Each window reinitialises inference_state (fresh feature cache).
    The last mask of each window is passed as a mask prompt to the next.

    Only frames whose index is in save_indices are written to disk as PNGs.
    Centroid stats are computed and returned for every frame regardless.

    Returns list[(cx, cy, radius)] indexed by extracted-frame index.
    """
    if save_indices:
        mask_dir.mkdir(parents=True, exist_ok=True)

    # Temp directory holding only the current window's frames.
    # SAM2's init_state pre-allocates one tensor for ALL files in the directory
    # (num_frames × 3 × 1024 × 1024 × float32).  Pointing it at the full
    # frames_dir with 7200 files would require ~81 GB.  Using a per-window
    # subdir bounds that to window_size × ~12 MB ≈ 2.4 GB for window=200.
    win_dir = frames_dir.parent / "_sam2_win_tmp"

    all_stats: list[tuple[float, float, float]] = []
    prev_mask: np.ndarray | None = None
    all_jpgs  = sorted(frames_dir.glob("*.jpg"))
    n_windows = math.ceil(total_frames / window_size)

    try:
        for win_idx in range(n_windows):
            win_start = win_idx * window_size
            win_end   = min(win_start + window_size, total_frames)
            n         = win_end - win_start
            print(f"\n  Window {win_idx + 1}/{n_windows}  "
                  f"[frames {win_start}-{win_end - 1}]  ({n} frames)")

            # Populate win_dir with this window's frames using hardlinks.
            # Falls back to copying if hardlinks are unavailable (cross-device).
            if win_dir.exists():
                shutil.rmtree(win_dir)
            win_dir.mkdir()
            for local_idx, global_idx in enumerate(range(win_start, win_end)):
                dst = win_dir / f"{local_idx:06d}.jpg"
                try:
                    dst.hardlink_to(all_jpgs[global_idx])
                except (OSError, NotImplementedError):
                    shutil.copy2(all_jpgs[global_idx], dst)

            # init_state sees only window_size frames -> bounded RAM
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(
                    str(win_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )

                # Prompts always use local (0-based) indices within win_dir
                if prev_mask is None:
                    predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=1,
                        points=np.array([[float(click_point[0]),
                                          float(click_point[1])]], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32),
                    )
                else:
                    predictor.add_new_mask(
                        state, frame_idx=0, obj_id=1, mask=prev_mask,
                    )

                local_masks: dict[int, np.ndarray] = {}
                for local_idx, _, logits in tqdm(
                    predictor.propagate_in_video(state, max_frame_num_to_track=n),
                    total=n,
                    desc=f"  SAM2 win {win_idx + 1}",
                    unit="fr",
                    leave=False,
                ):
                    local_masks[local_idx] = (logits[0, 0] > 0.0).cpu().numpy()

            # Map local indices back to global; write PNGs and compute stats
            for local_idx in range(n):
                global_idx = win_start + local_idx
                mask = local_masks.get(local_idx)
                if mask is None:
                    all_stats.append((0.0, 0.0, 0.0))
                    continue
                if global_idx in save_indices:
                    cv2.imwrite(
                        str(mask_dir / f"{global_idx:06d}.png"),
                        (mask.astype(np.uint8) * 255),
                    )
                all_stats.append(mask_to_stats(mask))

            # Last mask of this window is the prompt for the next window
            if local_masks:
                prev_mask = local_masks[max(local_masks)]

    finally:
        if win_dir.exists():
            shutil.rmtree(win_dir)

    return all_stats


# ── Validation mosaic ─────────────────────────────────────────────────────────

def _square_panel(img_bgr: np.ndarray, size: int):
    h, w   = img_bgr.shape[:2]
    scale  = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    small  = cv2.resize(img_bgr, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0     = (size - nh) // 2
    x0     = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = small
    return canvas, scale, x0, y0


def make_validation_mosaic(
    video_path: Path,
    frames_dir: Path,
    mask_dir: Path,
    seg_stats: list,          # [(cx,cy,r), ...]  indexed by extracted frame
    dye_csv: Path | None,
    fps_eff: float,           # effective fps after stride
    n_panels: int,
    panel_sz: int,
    out_path: Path,
) -> None:
    """
    Build a mosaic of N frames, each showing three panels:
      LEFT   — raw video frame
      CENTRE — mask overlay (amber tint) + centroid (red) + equiv circle (white)
      RIGHT  — body-axis schematic (centroid + dye mark if available)
    """
    total = len(seg_stats)
    sample_idxs = np.linspace(0, total - 1, n_panels, dtype=int)

    # Load dye track CSV if present
    dye_track: dict[int, tuple[float, float, bool]] = {}
    if dye_csv and dye_csv.exists():
        with open(dye_csv) as f:
            for row in csv.DictReader(f):
                dye_track[int(row["frame_idx"])] = (
                    float(row["x"]), float(row["y"]), bool(int(row["visible"]))
                )

    all_jpgs = sorted(frames_dir.glob("*.jpg"))
    rows = []

    for idx in sample_idxs:
        if idx >= len(all_jpgs):
            continue
        frame_bgr = cv2.imread(str(all_jpgs[idx]))
        if frame_bgr is None:
            continue
        cx, cy, radius = seg_stats[idx]

        mask_path = mask_dir / f"{idx:06d}.png"
        mask_img  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None

        # ── LEFT: raw frame ──────────────────────────────────────────────────
        left, sc, ox, oy = _square_panel(frame_bgr, panel_sz)

        # ── CENTRE: mask overlay + centroid ──────────────────────────────────
        mid_base = frame_bgr.copy()
        if mask_img is not None:
            tint = np.zeros_like(mid_base)
            tint[mask_img > 127] = MASK_COLOR
            mid_base = cv2.addWeighted(mid_base, 0.6, tint, 0.4, 0)

        mid, sc2, ox2, oy2 = _square_panel(mid_base, panel_sz)

        if radius > 0:
            ccx = round(cx * sc2) + ox2
            ccy = round(cy * sc2) + oy2
            cr  = round(radius * sc2)
            cv2.circle(mid, (ccx, ccy), cr, (255, 255, 255), 1, cv2.LINE_AA)  # equiv circle
            cv2.circle(mid, (ccx, ccy), 5, (0, 0, 220), -1, cv2.LINE_AA)      # centroid red
            cv2.circle(mid, (ccx, ccy), 6, (255, 255, 255), 1, cv2.LINE_AA)

        # ── RIGHT: body-axis schematic (dimmed frame) ─────────────────────────
        bg = (frame_bgr.astype(np.float32) * 0.25).astype(np.uint8)
        right, sc3, ox3, oy3 = _square_panel(bg, panel_sz)

        raw_frame_idx = idx  # extracted-frame index == original if stride=1
        dye = dye_track.get(raw_frame_idx)
        if radius > 0 and dye is not None:
            dx  = round(dye[0] * sc3) + ox3
            dy_ = round(dye[1] * sc3) + oy3
            ccx = round(cx * sc3) + ox3
            ccy = round(cy * sc3) + oy3
            cr  = round(radius * sc3)
            cv2.circle(right, (ccx, ccy), cr, (40, 40, 40), 2, cv2.LINE_AA)
            cv2.line(right, (ccx, ccy), (dx, dy_), (0, 200, 80), 1, cv2.LINE_AA)
            cv2.circle(right, (ccx, ccy), 5, (0, 0, 220), -1, cv2.LINE_AA)
            cv2.circle(right, (ccx, ccy), 6, (255, 255, 255), 1, cv2.LINE_AA)
            dye_col = (0, 255, 50) if dye[2] else (80, 80, 200)
            cv2.circle(right, (dx, dy_), 5, dye_col, -1, cv2.LINE_AA)
            cv2.circle(right, (dx, dy_), 6, (255, 255, 255), 1, cv2.LINE_AA)
        elif radius > 0:
            ccx = round(cx * sc3) + ox3
            ccy = round(cy * sc3) + oy3
            cr  = round(radius * sc3)
            cv2.circle(right, (ccx, ccy), cr, (40, 40, 40), 2, cv2.LINE_AA)
            cv2.circle(right, (ccx, ccy), 5, (0, 0, 220), -1, cv2.LINE_AA)
            cv2.putText(right, "no dye track", (ox3 + 4, oy3 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        # Label strip
        ts  = idx / fps_eff
        lbl = f"frame {idx}  t={ts:.1f}s  c=({cx:.0f},{cy:.0f})  r={radius:.0f}px"
        strip = np.zeros((22, panel_sz * 3, 3), dtype=np.uint8)
        cv2.putText(strip, lbl, (6, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 200, 200), 1, cv2.LINE_AA)

        row = np.vstack([np.hstack([left, mid, right]), strip])
        rows.append(row)

    if not rows:
        print("[warn] No panels generated for validation mosaic.")
        return

    # Column headers
    header = np.zeros((28, rows[0].shape[1], 3), dtype=np.uint8)
    for i, label in enumerate(["Raw frame", "Mask overlay + centroid", "Body axis (centroid + dye)"]):
        x = i * panel_sz + 6
        cv2.putText(header, label, (x, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1, cv2.LINE_AA)

    mosaic = np.vstack([header] + rows)
    cv2.imwrite(str(out_path), mosaic)
    print(f"Validation mosaic saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="SAM2 bell segmentation — Stage 1")
    ap.add_argument("--video", default="data/test_clip_1min.mp4")
    ap.add_argument("--stride", type=int, default=1,
                    help="Frame stride for extraction (default 1 = all frames)")
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                    help=f"SAM2 window size in extracted frames (default {WINDOW_SIZE})")
    ap.add_argument("--dye-csv", default=None,
                    help="Optional CoTracker CSV to overlay body axis in validation")
    ap.add_argument("--save-n-masks", type=int, default=20,
                    help="Number of evenly-spaced sample masks to save as PNGs (default 20). "
                         "Pass 0 to save none, or use --save-all-masks for every frame.")
    ap.add_argument("--save-all-masks", action="store_true",
                    help="Save a PNG mask for every frame (needed later for RAFT). "
                         "Produces ~7200 files for a 1-min 120fps clip.")
    args = ap.parse_args()

    root       = Path(__file__).parent.parent
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = root / video_path
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    cap  = cv2.VideoCapture(str(video_path))
    fps_raw   = cap.get(cv2.CAP_PROP_FPS)
    fps_eff   = fps_raw / args.stride
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ret, first_bgr = cap.read()
    cap.release()

    if not ret:
        sys.exit("Cannot read first frame.")

    stem = video_path.stem
    OUTPUTS_DIR.mkdir(exist_ok=True)
    frames_dir = OUTPUTS_DIR / f"{stem}_frames"
    mask_dir   = OUTPUTS_DIR / f"{stem}_masks"
    seg_csv    = OUTPUTS_DIR / f"{stem}_seg.csv"
    val_img    = OUTPUTS_DIR / f"{stem}_sam2_validation.png"

    n_extracted = (total_raw + args.stride - 1) // args.stride
    print(f"\nVideo  : {video_path.name}")
    print(f"Size   : {w}x{h}  |  {fps_raw:.1f} fps  |  {total_raw} raw frames")
    print(f"Stride : {args.stride}  =>  {n_extracted} extracted frames  "
          f"({fps_eff:.1f} fps effective)")
    n_wins = math.ceil(n_extracted / args.window_size)
    print(f"Windows: {n_wins} x {args.window_size} frames\n")

    # ── 1. Extract frames ─────────────────────────────────────────────────────
    n_extracted = extract_frames(video_path, frames_dir, args.stride)

    # ── 2. Click UI ───────────────────────────────────────────────────────────
    point = ClickSelector(first_bgr, DISPLAY_SCALE).run()
    if point is None:
        sys.exit("No point selected.")
    print(f"Prompt point: x={point[0]}, y={point[1]}")

    # ── 3. Load SAM2 ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SAM2 on {device}...")
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(SAM2_CFG, str(SAM2_WEIGHTS), device=device)

    # ── 4. Determine which frames to save as PNGs ────────────────────────────
    if args.save_all_masks:
        save_indices = set(range(n_extracted))
        print(f"Mask mode: saving all {n_extracted} frames as PNGs")
    elif args.save_n_masks > 0:
        idxs = np.linspace(0, n_extracted - 1, args.save_n_masks, dtype=int)
        save_indices = set(idxs.tolist())
        print(f"Mask mode: saving {len(save_indices)} sample frames as PNGs")
    else:
        save_indices = set()
        print("Mask mode: no PNGs saved (CSV only)")

    # ── 5. Propagate ─────────────────────────────────────────────────────────
    print(f"\nRunning SAM2 propagation...")
    seg_stats = run_sam2(
        predictor, frames_dir, point, mask_dir,
        n_extracted, args.window_size, save_indices,
    )

    # ── 6. Save centroid CSV ──────────────────────────────────────────────────
    print(f"\nWriting centroid CSV: {seg_csv}")
    with open(seg_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_idx", "timestamp_s", "cx", "cy", "radius_px"])
        for i, (cx, cy, r) in enumerate(seg_stats):
            raw = i * args.stride
            wr.writerow([raw, f"{raw / fps_raw:.4f}", f"{cx:.2f}", f"{cy:.2f}", f"{r:.2f}"])

    # ── 7. Validation mosaic ──────────────────────────────────────────────────
    dye_csv_path = None
    if args.dye_csv:
        dye_csv_path = Path(args.dye_csv)
    else:
        # Auto-find CoTracker CSV for the same stem
        candidates = list(OUTPUTS_DIR.glob(f"{stem}_track.csv"))
        if candidates:
            dye_csv_path = candidates[0]
            print(f"Auto-found dye CSV: {dye_csv_path.name}")

    print(f"Building validation mosaic...")
    make_validation_mosaic(
        video_path, frames_dir, mask_dir, seg_stats,
        dye_csv_path, fps_eff, N_VALID, PANEL_SZ, val_img,
    )

    valid_pct = sum(1 for cx, cy, r in seg_stats if r > 0) / max(len(seg_stats), 1) * 100
    print(f"\nDone.  Valid masks: {valid_pct:.1f}%")
    print(f"  Masks  : {mask_dir}/")
    print(f"  CSV    : {seg_csv}")
    print(f"  Visual : {val_img}")


if __name__ == "__main__":
    main()
