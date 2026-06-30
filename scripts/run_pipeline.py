"""
scripts/run_pipeline.py

CLI entry point for the full Cassiopea pipeline.

Collects the two required click points (bell centre for SAM2, dye mark for
CoTracker), then hands off to src/pipeline.py which runs all stages with
the optimal parallel schedule for the available GPU.

Usage
-----
  venv\Scripts\python scripts\run_pipeline.py --video data\test_clip.mp4
  venv\Scripts\python scripts\run_pipeline.py --video data\test_clip.mp4 --stride 4
  venv\Scripts\python scripts\run_pipeline.py --help

The script prints a live progress table while the pipeline runs.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline
from src.scheduler import ProgressEvent, TaskStatus
from src.resources import HARDWARE


# ── CLI progress printer ──────────────────────────────────────────────────────

_BAR_WIDTH = 20
_last_lines = 0


def _clear_lines(n: int) -> None:
    """Move cursor up n lines and clear them (ANSI)."""
    if n > 0:
        print(f"\033[{n}A\033[J", end="", flush=True)


def _fmt_time(s: float) -> str:
    """Format elapsed seconds as a compact string: 45s / 2m34s / 1h02m."""
    if s < 1:
        return ""
    if s < 60:
        return f"{s:.0f}s"
    m = int(s // 60)
    sec = int(s % 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    return f"{m//60}h{m%60:02d}m"


def make_cli_progress(task_names: list[str], pipeline_start: float):
    """
    Returns a ProgressCallback that prints a live updating table with timers.

    Each task row shows: icon | name | progress bar | pct | message | elapsed
    The last row shows overall progress + total pipeline elapsed time.
    Thread-safe — multiple tasks may call this simultaneously.
    """
    _lock   = threading.Lock()
    _state: dict[str, ProgressEvent] = {}

    STATUS_ICON = {
        TaskStatus.PENDING:   "[ ]",
        TaskStatus.WAITING:   "[~]",
        TaskStatus.RUNNING:   "[>]",
        TaskStatus.DONE:      "[+]",
        TaskStatus.SKIPPED:   "[=]",
        TaskStatus.FAILED:    "[!]",
        TaskStatus.CANCELLED: "[-]",
    }

    def _render():
        global _last_lines
        lines = []
        for name in task_names:
            ev = _state.get(name)
            if ev is None:
                lines.append(f"  [ ]  {name}")
                continue
            icon    = STATUS_ICON.get(ev.status, "?")
            pct     = ev.fraction * 100
            filled  = round(ev.fraction * _BAR_WIDTH)
            bar     = "#" * filled + "." * (_BAR_WIDTH - filled)
            msg     = (ev.message or "")[:32]
            t_str   = _fmt_time(ev.elapsed_s)
            time_col = f"[{t_str:>6}]" if t_str else "        "
            lines.append(
                f"  {icon}  {name:<33s} [{bar}] {pct:5.1f}%  {time_col}  {msg}"
            )

        # Overall row with pipeline wall clock
        overall      = max((e.overall_fraction for e in _state.values()), default=0.0)
        filled       = round(overall * _BAR_WIDTH)
        bar          = "#" * filled + "." * (_BAR_WIDTH - filled)
        wall_elapsed = _fmt_time(time.time() - pipeline_start)
        lines.append(
            f"\n  Overall [{bar}] {overall*100:5.1f}%   wall: {wall_elapsed}"
        )

        _clear_lines(_last_lines)
        output = "\n".join(lines)
        print(output, flush=True)
        _last_lines = output.count("\n") + 1

    def callback(event: ProgressEvent) -> None:
        with _lock:
            _state[event.task_name] = event
            _render()

    return callback


# ── Click UI ──────────────────────────────────────────────────────────────────

class ClickSelector:
    """Reuse the same click UI pattern from the individual scripts."""
    SCALE = 2

    def __init__(self, frame_bgr, prompt: str):
        h, w = frame_bgr.shape[:2]
        self._disp = cv2.resize(frame_bgr,
                                (int(w * self.SCALE), int(h * self.SCALE)))
        self._scale = self.SCALE
        self._prompt = prompt
        self.point = None

    def _on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (round(x / self._scale), round(y / self._scale))

    def run(self) -> tuple[int, int] | None:
        title = f"{self._prompt}  —  ENTER/SPACE confirm  ESC quit"
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, self._on_mouse)
        while True:
            disp = self._disp.copy()
            if self.point:
                px = round(self.point[0] * self._scale)
                py = round(self.point[1] * self._scale)
                cv2.drawMarker(disp, (px, py), (0, 255, 0),
                               cv2.MARKER_CROSS, 30, 2)
                cv2.circle(disp, (px, py), 12, (0, 255, 0), 2)
                cv2.putText(disp, f"({self.point[0]}, {self.point[1]})",
                            (px + 16, py - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(disp, self._prompt, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.imshow(title, disp)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32) and self.point:
                break
            if key == 27:
                self.point = None
                break
        cv2.destroyAllWindows()
        return self.point


def collect_clicks(video_path: Path) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """
    Open two sequential click UIs on the first video frame.
    Returns (bell_click, dye_click) or None if user cancelled.
    """
    cap = cv2.VideoCapture(str(video_path))
    ret, first = cap.read()
    cap.release()
    if not ret:
        print(f"Cannot read first frame from {video_path}")
        return None

    print("\nStep 1/2 — Click the JELLYFISH BELL (anywhere on the bell body)")
    bell_click = ClickSelector(first, "Click the jellyfish BELL").run()
    if bell_click is None:
        print("Cancelled.")
        return None
    print(f"  Bell click: {bell_click}")

    print("\nStep 2/2 — Click the DYE MARK on the bell")
    dye_click = ClickSelector(first, "Click the DYE MARK").run()
    if dye_click is None:
        print("Cancelled.")
        return None
    print(f"  Dye click:  {dye_click}")

    return bell_click, dye_click


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cassiopea full analysis pipeline (parallelised)")
    ap.add_argument("--video",        default="data/test_clip_1min.mp4")
    ap.add_argument("--calib",        default=None,
                    help="Path to calibration JSON (auto-detected if omitted)")
    ap.add_argument("--stride",       type=int,   default=4,
                    help="SAM2 frame stride (default 4 = 30fps effective at 120fps)")
    ap.add_argument("--cotracker-stride", type=int, default=8,
                    help="CoTracker frame stride (default 8); can be larger than --stride "
                         "since nearest-frame interpolation handles the mismatch")
    ap.add_argument("--window-size",  type=int,   default=400,
                    help="SAM2 window size (default 400, larger = less overhead)")
    ap.add_argument("--inner-frac",   type=float, default=0.85)
    ap.add_argument("--outer-frac",   type=float, default=1.05)
    ap.add_argument("--pre-window",   type=int,   default=30)
    ap.add_argument("--min-distance", type=float, default=0.42)
    ap.add_argument("--prominence",   type=float, default=0.08)
    ap.add_argument("--save-n-masks", type=int,   default=20)
    # PERF branch options
    ap.add_argument("--image-size",   type=int,   default=512,
                    help="SAM2 internal ViT resolution (default 512 on perf branch)")
    ap.add_argument("--no-gpu-approach-b", action="store_true",
                    help="Disable GPU grid_sample for Approach B (fall back to CPU)")
    args = ap.parse_args()

    root       = Path(__file__).parent.parent
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = root / video_path
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    # Auto-find calibration
    calib_path = Path(args.calib) if args.calib else None
    if calib_path is None:
        cands = sorted((root / "calibration").glob("*.json"))
        if not cands:
            sys.exit("No calibration JSON found in calibration/. "
                     "Run scripts/calibrate_rhopalia.py first.")
        calib_path = cands[0]

    print(f"\nCassiopea Pipeline")
    print(f"  Video    : {video_path.name}")
    print(f"  Calib    : {calib_path.name}")
    print(f"  GPU      : {HARDWARE.gpu_name}  ({HARDWARE.gpu_vram_gb:.1f} GB)")
    print(f"  Parallel : up to {HARDWARE.max_gpu_concurrent} GPU tasks simultaneously")
    print(f"  SAM2 stride    : {args.stride}  ({120/args.stride:.0f} fps effective at 120fps)")
    print(f"  CoTrack stride : {args.cotracker_stride}  ({120/args.cotracker_stride:.0f} fps effective at 120fps)\n")

    # Collect click points
    clicks = collect_clicks(video_path)
    if clicks is None:
        sys.exit(0)
    bell_click, dye_click = clicks

    # PERF branch: CoTracker retained for accuracy — SAM2 dye tracking was tested
    # and found unreliable for faint dye marks. Other perf optimisations still apply.
    _sam2_tracks_dye = False

    task_names = [
        "SAM2 segmentation",
        "CoTracker tracking",
        "Margin diff (lab frame)",
        "Body-frame rotation",
        "Pulse initiation analysis",
    ]
    cancel_event   = threading.Event()
    pipeline_start = time.time()
    progress_cb    = make_cli_progress(task_names, pipeline_start)

    print("\nRunning pipeline...\n")

    try:
        result = run_pipeline(
            video_path       = video_path,
            bell_click       = bell_click,
            dye_click        = dye_click,
            calib_path       = calib_path,
            stride           = args.stride,
            cotracker_stride = args.cotracker_stride,
            window_size = args.window_size,
            inner_frac  = args.inner_frac,
            outer_frac  = args.outer_frac,
            pre_window  = args.pre_window,
            min_distance= args.min_distance,
            prominence  = args.prominence,
            save_n_masks= args.save_n_masks,
            # PERF branch: SAM2 tracks dye internally, GPU Approach B, 512px
            sam2_tracks_dye     = _sam2_tracks_dye,
            image_size          = args.image_size,
            use_gpu_approach_b  = not args.no_gpu_approach_b,
            progress_callback   = progress_cb,
            cancel_event        = cancel_event,
        )
    except KeyboardInterrupt:
        cancel_event.set()
        print("\nCancelled by user.")
        sys.exit(1)

    wall = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(result.summary())
    print(f"\nTask timings:")
    for name, status in result.task_status.items():
        t = result.task_elapsed.get(name, 0.0)
        t_str = _fmt_time(t) if t > 0 else "-"
        print(f"  {name:<35s} {status:<10s} {t_str:>8}")
    print(f"\n  Total wall-clock: {_fmt_time(wall)}")
    print(f"{'='*60}")

    # run_pipeline() now writes the provenance log itself (shared by CLI + UI).
    from src.tasks import run_dir
    log_path = run_dir(video_path) / f"{video_path.stem}_run_log.json"
    print(f"\n  Run log: {log_path}")

    if result.success:
        print("\nOutputs:")
        for label, path in [
            ("Seg CSV",         result.seg_csv),
            ("Track CSV",       result.track_csv),
            ("Initiation CSV",  result.initiation_csv),
            ("Summary plot",    result.initiation_plot),
            ("Annotated video", result.annotated_video),
        ]:
            if path:
                print(f"  {label:20s} {path}")


if __name__ == "__main__":
    main()
