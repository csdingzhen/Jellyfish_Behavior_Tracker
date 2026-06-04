"""
scripts/cotracker_test.py

Interactive dye-mark tracker using CoTracker3 (offline model).

Usage:
    venv\Scripts\python scripts\cotracker_test.py
    venv\Scripts\python scripts\cotracker_test.py --video data/test_clip_1min.mp4
    venv\Scripts\python scripts\cotracker_test.py --stride 4   # every 4th frame (30 fps)

Controls (click window):
    Left-click    — place / re-place the query point
    ENTER / SPACE — confirm and start tracking
    ESC           — quit without tracking

Outputs (written to outputs/):
    <stem>_tracked.mp4  — video with overlaid trajectory and visibility
    <stem>_track.csv    — frame_idx, timestamp_s, x, y, visible
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from cotracker.predictor import CoTrackerPredictor

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COTRACKER_WEIGHTS, OUTPUTS_DIR

# ── Constants ─────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 200   # frames per GPU pass — safe for 8 GB VRAM at 640×512
DISPLAY_SCALE = 2     # enlarge first frame for easier clicking
TRAIL_LEN     = 60    # past frames shown as motion trail (~0.5 s at 120 fps)

COLOR_VIS    = (  0, 255,  50)   # bright green — confirmed visible
COLOR_OCCL   = ( 80,  80, 200)   # muted purple — low confidence
COLOR_TRAIL  = (  0, 180, 255)   # amber trail


# ── Click UI ──────────────────────────────────────────────────────────────────

class ClickSelector:
    """Show the first frame enlarged; let the user click a point, confirm with ENTER."""

    def __init__(self, frame_bgr: np.ndarray, scale: float):
        h, w = frame_bgr.shape[:2]
        self.scale = scale
        self._disp_orig = cv2.resize(
            frame_bgr, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        self.point: tuple[int, int] | None = None

    def _on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (round(x / self.scale), round(y / self.scale))

    def run(self) -> tuple[int, int] | None:
        title = "Click the dye mark — ENTER/SPACE to confirm, ESC to quit"
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, self._on_mouse)

        while True:
            disp = self._disp_orig.copy()

            if self.point is not None:
                px = round(self.point[0] * self.scale)
                py = round(self.point[1] * self.scale)
                cv2.drawMarker(disp, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                cv2.circle(disp, (px, py), 12, (0, 255, 0), 2)
                cv2.putText(
                    disp, f"({self.point[0]}, {self.point[1]})",
                    (px + 16, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    disp, "Click the dye mark",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2,
                )

            cv2.imshow(title, disp)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32) and self.point is not None:   # ENTER or SPACE
                break
            if key == 27:                                      # ESC
                self.point = None
                break

        cv2.destroyAllWindows()
        return self.point


# ── Video helpers ─────────────────────────────────────────────────────────────

def read_chunk(cap: cv2.VideoCapture, start_raw: int, n_samples: int,
               stride: int) -> np.ndarray:
    """
    Read n_samples frames starting at raw frame start_raw, sampling every
    `stride` frames.  Returns uint8 RGB array (T, H, W, 3).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_raw)
    frames, raw_read = [], 0
    while len(frames) < n_samples:
        ret, frame = cap.read()
        if not ret:
            break
        if raw_read % stride == 0:
            frames.append(frame[:, :, ::-1].copy())  # BGR → RGB
        raw_read += 1
    return np.stack(frames) if frames else np.zeros((0, 1, 1, 3), dtype=np.uint8)


def frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """(T, H, W, 3) uint8 → (1, T, 3, H, W) float32 on device."""
    return (
        torch.from_numpy(frames)
        .permute(0, 3, 1, 2)
        .float()
        .unsqueeze(0)
        .to(device)
    )


# ── Chunked CoTracker inference ───────────────────────────────────────────────

def track_chunked(
    model: CoTrackerPredictor,
    cap: cv2.VideoCapture,
    query_xy: tuple[int, int],
    total_raw_frames: int,
    stride: int,
    chunk_size: int,
    progress_callback=None,   # (current, total, message) — for scheduler/UI
    cancel_event=None,        # threading.Event — checked between chunks
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track `query_xy` through the whole video using non-overlapping chunks.
    Each chunk hands off the last tracked position as the seed for the next.

    Returns
    -------
    tracks  : (N, 2)  float32 — (x, y) in original pixel coordinates
    visible : (N,)    bool    — CoTracker visibility flag
    """
    device = next(model.parameters()).device
    n_sampled = (total_raw_frames + stride - 1) // stride
    all_tracks  = np.zeros((n_sampled, 2), dtype=np.float32)
    all_visible = np.zeros(n_sampled, dtype=bool)

    qx, qy = float(query_xy[0]), float(query_xy[1])
    write_ptr  = 0
    raw_ptr    = 0

    with tqdm(total=n_sampled, desc="Tracking", unit="frames",
              disable=progress_callback is not None) as pbar:
        while raw_ptr < total_raw_frames:
            if cancel_event is not None and cancel_event.is_set():
                break

            frames = read_chunk(cap, raw_ptr, chunk_size, stride)
            n = len(frames)
            if n == 0:
                break

            vid = frames_to_tensor(frames, device)                # (1,T,3,H,W)
            q   = torch.tensor([[[0.0, qx, qy]]], device=device)  # (1,1,3)

            with torch.no_grad():
                pred_tracks, pred_vis = model(vid, queries=q)
            # pred_tracks: (1, T, 1, 2)   pred_vis: (1, T, 1)

            t_np = pred_tracks[0, :n, 0, :].cpu().numpy()        # (T, 2)
            v_np = pred_vis[0, :n, 0].cpu().numpy().astype(bool)  # (T,)

            all_tracks [write_ptr : write_ptr + n] = t_np
            all_visible[write_ptr : write_ptr + n] = v_np

            # Hand-off: seed next chunk from last tracked position
            qx, qy = float(t_np[-1, 0]), float(t_np[-1, 1])

            write_ptr += n
            raw_ptr   += n * stride
            pbar.update(n)
            if progress_callback is not None:
                progress_callback(write_ptr, n_sampled, f"chunk {raw_ptr // (chunk_size * stride)}")

    return all_tracks[:write_ptr], all_visible[:write_ptr]


# ── Output rendering ──────────────────────────────────────────────────────────

def render_video(
    cap: cv2.VideoCapture,
    tracks: np.ndarray,
    visible: np.ndarray,
    out_path: Path,
    fps: float,
    stride: int,
) -> None:
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps / stride, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    sample_idx = 0
    raw_idx    = 0

    with tqdm(total=len(tracks), desc="Rendering", unit="frames") as pbar:
        while sample_idx < len(tracks):
            ret, frame = cap.read()
            if not ret:
                break
            if raw_idx % stride != 0:
                raw_idx += 1
                continue

            # Motion trail
            trail_start = max(0, sample_idx - TRAIL_LEN)
            for i in range(trail_start, sample_idx):
                alpha = (i - trail_start) / TRAIL_LEN
                c = tuple(round(v * alpha) for v in COLOR_TRAIL)
                cv2.circle(frame, (round(tracks[i, 0]), round(tracks[i, 1])), 1, c, -1)

            # Current point
            cx, cy = round(tracks[sample_idx, 0]), round(tracks[sample_idx, 1])
            vis    = bool(visible[sample_idx])
            color  = COLOR_VIS if vis else COLOR_OCCL
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

            # HUD overlay
            label = f"f{raw_idx:05d}  ({cx},{cy})  {'VIS' if vis else 'occl'}"
            cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (220, 220, 220), 1, cv2.LINE_AA)

            writer.write(frame)
            sample_idx += 1
            raw_idx    += 1
            pbar.update(1)

    writer.release()


def write_csv(
    path: Path,
    tracks: np.ndarray,
    visible: np.ndarray,
    fps: float,
    stride: int,
) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp_s", "x", "y", "visible"])
        for i, ((x, y), v) in enumerate(zip(tracks, visible)):
            raw = i * stride
            w.writerow([raw, f"{raw / fps:.4f}", f"{x:.2f}", f"{y:.2f}", int(v)])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="CoTracker dye-mark test")
    ap.add_argument("--video", default="data/test_clip_1min.mp4",
                    help="Input video path (relative to project root or absolute)")
    ap.add_argument("--stride", type=int, default=1,
                    help="Frame stride: 1=all frames, 4=every 4th (30 fps from 120 fps)")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                    help=f"Frames per GPU pass (default {CHUNK_SIZE})")
    args = ap.parse_args()

    # Resolve video path
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = Path(__file__).parent.parent / video_path
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps       = cap.get(cv2.CAP_PROP_FPS)
    h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_sampled = (total_raw + args.stride - 1) // args.stride

    print(f"\nVideo  : {video_path.name}")
    print(f"Size   : {w}×{h}  |  {fps:.1f} fps  |  {total_raw} raw frames")
    print(f"Stride : {args.stride}  →  {n_sampled} sampled frames  "
          f"({fps / args.stride:.1f} fps effective)")
    print(f"Chunks : ~{(n_sampled + args.chunk_size - 1) // args.chunk_size} "
          f"× {args.chunk_size} frames\n")

    # Read first frame for click UI
    ret, first_bgr = cap.read()
    if not ret:
        sys.exit("Cannot read first frame.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Click UI
    point = ClickSelector(first_bgr, scale=DISPLAY_SCALE).run()
    if point is None:
        print("No point selected — exiting.")
        sys.exit(0)
    print(f"Query point: x={point[0]}, y={point[1]}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CoTracker weights from {COTRACKER_WEIGHTS} on {device}...")
    model = CoTrackerPredictor(checkpoint=str(COTRACKER_WEIGHTS)).to(device).eval()

    # Track
    tracks, visible = track_chunked(
        model, cap, point, total_raw,
        stride=args.stride, chunk_size=args.chunk_size,
    )

    vis_pct = visible.mean() * 100
    print(f"\nTracking complete: {len(tracks)} frames, {vis_pct:.1f}% marked visible")

    # Save outputs
    OUTPUTS_DIR.mkdir(exist_ok=True)
    stem    = video_path.stem
    vid_out = OUTPUTS_DIR / f"{stem}_tracked.mp4"
    csv_out = OUTPUTS_DIR / f"{stem}_track.csv"

    print(f"Rendering → {vid_out}")
    render_video(cap, tracks, visible, vid_out, fps, stride=args.stride)

    print(f"Writing CSV → {csv_out}")
    write_csv(csv_out, tracks, visible, fps, stride=args.stride)

    cap.release()
    print("\nDone.")
    print(f"  Video : {vid_out}")
    print(f"  CSV   : {csv_out}")


if __name__ == "__main__":
    main()
