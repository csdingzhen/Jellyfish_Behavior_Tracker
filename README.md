# Cassiopea Behavior Analysis Pipeline

Automated video analysis pipeline for *Cassiopea* (upside-down jellyfish) recordings. Given a top-view video with a small dye mark applied to the bell, the pipeline extracts:

- **Bell orientation over time** — which way the animal is facing at each moment
- **Contraction timing** — when the bell pulses and how strongly
- **Pulse initiation site** — which rhopalium (marginal neurocluster) triggered each contraction

---

## Background

*Cassiopea* has 16 rhopalia distributed around the bell margin. These act as independent pacemakers — any one of them can trigger a contraction wave that spreads across the whole bell. The key scientific question is: which rhopalium fires most often, and do they follow a pattern?

Because the animal has no obvious anatomical landmark for orientation, we apply a small **dye mark** to the bell surface. This mark provides the body-fixed reference direction from which all angular measurements are made.

---

## How it works

```text
Your video
    │
    ├─► Track the dye mark through every frame            (CoTracker)
    │       → bell orientation at each moment
    │
    ├─► Outline the jellyfish bell in every frame         (SAM2)
    │       → bell centre, size, and boundary
    │       → contour radii r(θ, t): bell-edge position per angle
    │
    ├─► [One-time] From a hi-res photo, click each rhopalium
    │       → fixed body-frame angles saved to calibration file
    │
    └─► For each frame pair, measure optical change at the bell margin
            → decomposed by angle (polar unwrap)
            → peaks in total margin activity = contraction events
            → first angle to show elevated activity = initiation site
            → match to nearest rhopalium from calibration
```

---

## How the initiation detection works (Approach B)

### Polar unwrapping

`cv2.warpPolar` remaps the circular bell image into a rectangle where rows = angle
(0–360°) and columns = radius, "unrolling" the bell margin into a flat band:

```text
Original frame           After polar unwrap
                         0°  ──────────────────── 360°
   ┌───────────┐         │  inner  [margin] outer  │
   │     ○     │  ──►    │  ░░░░░░ █████████ ░░░░  │
   └───────────┘         └─────────────────────────┘
                                   ▲
                             strip we extract
```

We keep only the outer annular band (default: 75–105% of bell radius).
Setting the outer edge beyond 100% captures tissue outside the bell — where an
expansion wave first appears when a rhopalium fires.

### Frame-to-frame difference per angle

```text
margin_diff[θ, t] = mean( |strip(t+1, θ) − strip(t, θ)| )
```

This 2D signal detects tissue deformation, texture change, and boundary movement —
sensitive to events before the macroscopic contraction is visible.

### Body-frame rotation

The signal is rotated by −φ\_dye(t) so that angle 0 always points toward the dye
mark, making all frames directly comparable regardless of animal rotation.

### Pulse detection and initiation

Summing across all angles gives total margin activity. Peaks = contraction events.
For each peak, the algorithm scans backward through the pre-window to find the
earliest frame where a localised angle exceeds 25% of the peak — that angle is the
initiation site, matched to the nearest rhopalium from the calibration file.

The result is a **space-time plot** (angle × time heatmap) showing the wave
spreading from the initiation rhopalium:

```text
angle
360°│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    │░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓░░░  ← wave spreading
    │░░░░░░░░░░░░▓▓░░░░░░░░░░░░░░░░  ← initiation point
    │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0°└─────────────────────────────► time → peak
```

---

## Recommended configuration

Based on validation testing, the following settings give the best accuracy:

| Setting | Value | Notes |
| --- | --- | --- |
| SAM2 stride | **1** | Process every frame for accurate contour radii |
| CoTracker stride | **4** | Track dye every 4th frame (30 fps effective at 120 fps) |
| SAM2 model | `tiny` | Fastest; no measurable accuracy loss on Cassiopea's high-contrast bell |
| Inner frac | 0.75 | Inner edge of margin ring |
| Outer frac | 1.05 | Extends slightly past bell edge to capture outward expansion |

> **Speed vs accuracy trade-off:** stride=1 SAM2 is ~4× slower than stride=4 but eliminates
> interpolation artifacts in the contour radii that can distort initiation angle estimates.
> For exploratory runs use stride=4; for publication-quality results use stride=1.

---

## Requirements

### Hardware

- NVIDIA GPU with ≥ 6 GB VRAM (tested on RTX 4060 8 GB)
- ≥ 16 GB system RAM (32 GB recommended)
- CUDA 12.x driver installed

### Video format

- Top-view recording, camera fixed relative to the chamber
- Recommended: ≥ 30 fps (120 fps tested and preferred)
- Grayscale or colour; resolution ≥ 640 × 512

### Dye mark

- A small visible mark applied to the bell surface
- Fluorescent dye under UV illumination gives best contrast
- Place the mark off-centre so it can define an orientation angle

---

## Installation

All commands are run in **PowerShell** from the project folder.
See [INSTALLATION.md](INSTALLATION.md) for full instructions including cross-platform setup.

### 1. Clone the repository

```powershell
cd C:\Projects
git clone <repo-url> Jellyfish
cd Jellyfish
```

### 2. Run the setup script

```powershell
.\setup.ps1
```

Creates a virtual environment, installs all packages, downloads model weights, and
runs a GPU smoke test. After it finishes you should see:

```text
Setup complete.
```

### 3. Configure your video folder

Open [config.py](config.py) and set `VIDEO_DIR` to wherever your recordings live:

```python
VIDEO_DIR = Path(r"C:\Users\YourName\Videos\Cassiopea")
```

---

## Graphical interface (UI)

A napari-based desktop application covers both calibration and video processing without the command line.

**Double-click `Cassiopea Pipeline.exe`** in the project folder. It's a small
(~7 MB) launcher with the app icon — no console window, just a loading splash
then the main window, and it can be pinned to the taskbar. `setup.ps1` builds it
automatically; rebuild any time with `.\packaging\build_launcher.ps1`. It starts
the UI through the venv (so keep it next to `venv\` and `scripts\`); it does
**not** bundle Python/torch.

To launch from a terminal instead — useful for seeing startup errors, since the
`.exe` shows no console:

```powershell
.\venv\Scripts\python scripts\run_ui.py
```

The window has two tabs:

| Tab | Purpose |
|-----|---------|
| **Calibrate** | One-time annotation of rhopalia positions from a still photo |
| **Process** | Run the full analysis pipeline on a video |

See [UI_GUIDE.md](UI_GUIDE.md) for step-by-step instructions.

---

## Running the pipeline (CLI)

### Option A — One command (recommended)

`run_pipeline.py` runs all stages automatically with the optimal parallel schedule
for your GPU.

```powershell
# Recommended (best accuracy):
.\venv\Scripts\python.exe scripts\run_pipeline.py --video data\test_clip.mp4 --stride 1 --cotracker-stride 4

# Faster exploratory run:
.\venv\Scripts\python.exe scripts\run_pipeline.py --video data\test_clip.mp4 --stride 4 --cotracker-stride 4
```

The script opens two click windows in sequence — one for the bell, one for the dye
mark — then runs everything unattended. Outputs land in `outputs\<video_stem>\`.

**Key options:**

| Flag | Default | Effect |
| --- | --- | --- |
| `--stride` | 4 | SAM2 frame subsampling (1 = every frame; 4 = 30fps from 120fps) |
| `--cotracker-stride` | 4 | CoTracker frame interval (recommend matching `--stride`) |
| `--sam2-model` | `tiny` | SAM2 variant: `tiny`, `small`, `base_plus`, `large` |
| `--calib` | auto | Path to calibration JSON (auto-detected from `calibration/`) |
| `--inner-frac` | 0.75 | Inner edge of polar ring (fraction of bell radius) |
| `--outer-frac` | 1.05 | Outer edge — values > 1.0 capture outside-bell expansion signal |
| `--prominence` | 0.08 | Peak detection threshold (fraction of signal range) |

**Checkpoint / resume:** if the run is interrupted, restarting the same command
skips stages whose outputs already exist. Pass `--recompute` to force all stages to re-run.

---

### Option B — Individual scripts (standalone / debugging)

Each stage can be run independently.

#### Prepare a test clip

```powershell
& "C:\path\to\ffmpeg.exe" -i "C:\path\to\recording.mp4" `
    -ss 00:00:00 -t 60 -c copy data\test_clip.mp4
```

#### Step 1 — Calibrate rhopalium positions (one-time per animal)

Take a high-resolution still photo of the jellyfish with the dye mark visible.

```powershell
.\venv\Scripts\python.exe scripts\calibrate_rhopalia.py "C:\path\to\photo.jpg"
```

Three-step click process: bell centre → dye mark (0° reference) → each rhopalium.
BACKSPACE to undo, ENTER when done.

**Outputs:**

- `calibration\photo_name.json` — permanent rhopalium body-frame angle record
- `calibration\photo_name_annotated.png` — labelled verification diagram

#### Step 2 — Segment the bell (SAM2)

```powershell
.\venv\Scripts\python.exe scripts\run_sam2.py --video data\test_clip.mp4 --stride 1
```

Click anywhere on the bell body, press ENTER.

#### Step 3 — Track the dye mark (CoTracker)

```powershell
.\venv\Scripts\python.exe scripts\cotracker_test.py --video data\test_clip.mp4 --stride 4
```

Click the dye mark, press ENTER.

#### Step 4 — Pulse initiation analysis (Approach B)

```powershell
.\venv\Scripts\python.exe scripts\run_approach_b.py
```

Phase 1 computes `_margin_diff_lab.npy` (lab-frame) then `_margin_diff.npy`
(body-frame). Both are cached and reused on re-runs.

---

## SAM2 model selection

Four model sizes are available. `tiny` is recommended:

| Model | Weights | Speed (RTX 4060, stride=1) | Use when |
| --- | --- | --- | --- |
| `tiny` | 148 MB | ~3.5 fps | **Recommended default** |
| `small` | 185 MB | ~2.5 fps | Marginal accuracy improvement |
| `base_plus` | 308 MB | ~2.2 fps | If tiny shows mask drift |
| `large` | ~900 MB | ~0.8–1 fps | Maximum accuracy, short recordings only |

---

## Performance

### Processing time estimates

For a **10-minute recording at 120 fps** on an RTX 4060:

| Configuration | SAM2 | CoTracker | Approach B | Total |
| --- | --- | --- | --- | --- |
| stride=1 SAM2, stride=4 CT *(recommended)* | ~120 min | ~28 min | ~5 min | **~2.5 hrs** |
| stride=4 SAM2, stride=4 CT *(fast)* | ~30 min | ~28 min | ~5 min | **~1 hr** |
| stride=4 SAM2, stride=8 CT *(legacy default)* | ~30 min | ~110 min | ~5 min | **~2.5 hrs** |

### Active performance optimisations

- **bfloat16 inference** — CoTracker runs in mixed precision (~1.5–2× faster than float32)
- **`torch.compile`** — both CoTracker and the SAM2 image encoder are JIT-compiled on
  the first run; subsequent chunks benefit from fused kernels (~15–25% faster)
- **Parallel task scheduler** — `src/scheduler.py` runs SAM2, CoTracker, and
  Approach B Phase 1 concurrently, gated by available VRAM
- **Streaming video decode** — Approach B reads directly from the source video;
  no JPEG extraction needed
- **Automatic frame cleanup** — extracted JPEG frames (needed by SAM2) are deleted
  after segmentation completes, saving 5+ GB per recording

### Stride guidelines

| Stride | Effective fps (at 120 fps) | Accuracy |
| --- | --- | --- |
| 1 | 120 fps | Maximum — **recommended for analysis** |
| 4 | 30 fps | Good for exploration; slight interpolation artifacts |
| 8 | 15 fps | Fast; noticeable accuracy loss on initiation angles |

---

## Reading the initiation CSV

| Column | Meaning |
| --- | --- |
| `peak_frame` | Video frame at peak contraction |
| `timestamp_s` | Time in seconds |
| `init_angle_deg` | Body-frame angle of initiation site (0° = dye mark) |
| `rhopalium_id` | Matched rhopalium (0–15) |
| `angular_dist_deg` | Arc distance to matched rhopalium — below 11.25° = confident |
| `signal_confident` | 1 = strong signal AND angular_dist ≤ 11.25°, 0 = uncertain |

---

## Output file reference

Outputs are **scoped by project**: when a project named `proj1` is active, all of
its recordings are grouped under `outputs/proj1/`, with one subfolder per video
and project-level summaries alongside them. Runs with no project open fall back
to `outputs/<stem>/` directly.

```text
outputs\
  <project>\                             one folder per project (e.g. "proj1")
    videos.csv                           one row per recording — batch summary
                                         (n_pulses, confident %, dominant rhopalium, …)
    <stem>\                              one folder per video
      # ── Manifest ──────────────────────────────────────────
      <stem>_summary.json                metadata + params + provenance + results
                                         (the machine-readable "read me first")
      # ── Results ───────────────────────────────────────────
      <stem>_initiation_b.csv            Per-pulse rhopalium assignments
      <stem>_initiation_b_plot.png       Activity signal + firing histogram
      <stem>_initiation_b_annotated.mp4  Annotated video (first 60 s)
      # ── Intermediate data (reused on re-runs / re-analysis) ─
      <stem>_seg.csv                     Bell centre + radius per frame  (SAM2)
      <stem>_contour_radii.npy           Bell boundary r(θ, t)  (SAM2)
      <stem>_track.csv                   Dye mark position per frame  (CoTracker)
      <stem>_margin_diff_lab.npy         Lab-frame margin diff (cache for body-frame stage)
      <stem>_margin_diff.npy             Body-frame margin activity cache  (Approach B)
      # ── QC / diagnostics ──────────────────────────────────
      <stem>_tracked.mp4                 Dye mark overlay video
      <stem>_masks\                      20 sample PNG masks for visual spot-check
      # ── State / provenance ────────────────────────────────
      <stem>_sam2.complete               Checkpoint sentinel (SAM2 stage finished)
      <stem>_cotrack.complete            Checkpoint sentinel (CoTracker stage finished)
      <stem>_run_log.json                Config + timing for every run (appended; CLI and UI)

calibration\
  <animal>.json                  Rhopalium body-frame angles  (permanent record)
  <animal>_annotated.png         Labelled calibration diagram
```

> Notes:
>
> - With no project open, the `<project>\` level is omitted — outputs go straight
>   to `outputs\<stem>\` (and `videos.csv` lands at `outputs\videos.csv`).
> - The SAM2 frame-extraction is streamed, so no `<stem>_frames/` directory is
>   created.
> - `<stem>_margin_diff_lab.npy` is retained (not deleted) because the body-frame
>   stage's checkpoint depends on it.
> - `summary.json` and `videos.csv` are regenerated after every run, so they
>   always reflect the current results.

---

## Project structure

```text
src\
  calibration_core.py  Shared calibration math (body_angle, build_calibration, etc.)
  resources.py         GPU detection, GpuGate semaphore
  scheduler.py         DAG task runner with parallel execution and UI-ready callbacks
  tasks.py             Task factory functions wrapping each pipeline stage
  pipeline.py          run_pipeline() — assembles and executes the full pipeline

scripts\
  run_ui.py             Launch the napari graphical interface
  run_pipeline.py       Full pipeline CLI (primary entry point)
  run_sam2.py           SAM2 bell segmentation (standalone)
  cotracker_test.py     CoTracker dye tracking (standalone)
  run_approach_b.py     Approach B pulse initiation analysis (standalone)
  calibrate_rhopalia.py One-time rhopalium calibration from hi-res photo
  validate_tracking.py  Side-by-side validation video generator

ui\
  app.py           napari viewer setup, dock widgets, and cross-widget signal wiring
  widget.py        Two-tab container (CalibrationTab + ProcessingTab) + hardware/project bars
  calibration.py   Calibration workflow UI
  processing.py    Video processing workflow UI
  sidebar.py       Video browser / batch queue (left dock) — see UI_GUIDE.md
  watcher.py       FolderWatcher — detects recordings finishing on disk
  project.py       ProjectState persistence + continuity-click extraction
  hardware.py      GPU status bar + auto-queue toggle
  style.py         Shared dark theme (STYLESHEET) and card/badge widget helpers
  workers.py       Background thread worker and progress relay
  parameters.py    PipelineParams dataclass
  thumbnails.py    Video thumbnail cache helper

packaging\
  launcher.py        Source for the no-console launcher .exe
  make_ico.py        Generate assets/app_icon.ico from the SVG
  build_launcher.ps1 Build "Cassiopea Pipeline.exe" (PyInstaller)

calibration\      Rhopalium calibration files (tracked in git)
data\             Short test clips (gitignored)
outputs\          All pipeline outputs (gitignored)
weights\          Model weights (gitignored)
docs\             LLM-accessible project knowledge (architecture, design decisions)
```

---

## Known limitations

### Initiation detection accuracy

The polar margin approach works best with frame rate ≥ 60 fps, visible texture at
the bell margin, and consistent camera position. Some pulses return
`signal_confident = 0` — use only confident assignments for statistics.
The `angular_dist_deg` column quantifies confidence; values below 11.25°
(half of 360°/16 rhopalia spacing) indicate reliable assignment.

### Long recordings

The pipeline is designed for segment-by-segment processing. SAM2 uses 200-frame
windows internally (memory-bounded). For recordings longer than ~10 minutes,
run `run_pipeline.py` on each segment and concatenate the resulting CSVs.

---

## License notes

- **SAM2** (bell segmentation): Apache 2.0
- **CoTracker** (dye tracking): CC-BY-NC 4.0 — research use only. For commercial
  use, replace with TAPIR (Apache 2.0).
- **RAFT** (optical flow, via torchvision): BSD — installed but not used in the
  primary pipeline

---

## Citation

Models used:

- SAM2: Ravi et al., Meta AI (2024)
- CoTracker: Karaev et al., Meta AI + Oxford VGG (2023)
