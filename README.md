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
All checks passed. Environment is ready.
```

### 3. Configure your video folder

Open [config.py](config.py) and set `VIDEO_DIR` to wherever your recordings live:

```python
VIDEO_DIR = Path(r"C:\Users\YourName\Videos\Cassiopea")
```

---

## Running the pipeline

### Option A — One command (recommended)

`run_pipeline.py` runs all stages automatically with the optimal parallel schedule
for your GPU. SAM2 and CoTracker start simultaneously on GPUs with ≥ 20 GB VRAM;
Approach B Phase 1 overlaps with CoTracker on all hardware.

```powershell
.\venv\Scripts\python.exe scripts\run_pipeline.py --video data\test_clip.mp4
```

The script opens two click windows in sequence — one for the bell, one for the dye
mark — then runs everything unattended. Outputs land in
`outputs\<video_stem>\`.

**Key options:**

| Flag | Default | Effect |
| --- | --- | --- |
| `--stride` | 4 | Frame subsampling (4 = 30 fps from 120 fps). Higher = faster, less temporal resolution. |
| `--sam2-model` | base | SAM2 variant: `tiny`, `small`, `base`, `large` (see model comparison below) |
| `--calib` | auto | Path to calibration JSON (auto-detected from `calibration/`) |

**Checkpoint / resume:** if the run is interrupted, restarting the same command
skips stages whose outputs already exist.

---

### Option B — Individual scripts (standalone / debugging)

Each stage can be run independently if you need to re-run one step without
repeating the others.

#### Prepare a test clip

```powershell
& "C:\path\to\ffmpeg.exe" -i "C:\path\to\recording.mp4" `
    -ss 00:00:00 -t 60 -c copy data\test_clip.mp4
```

#### Step 1 — Segment the bell (SAM2)

```powershell
.\venv\Scripts\python.exe scripts\run_sam2.py --video data\test_clip.mp4 --stride 4
```

Click anywhere on the bell body, press ENTER.

#### Step 2 — Track the dye mark (CoTracker)

```powershell
.\venv\Scripts\python.exe scripts\cotracker_test.py --video data\test_clip.mp4 --stride 4
```

Click the dye mark, press ENTER.

#### Step 3 — Validate tracking

```powershell
.\venv\Scripts\python.exe scripts\validate_tracking.py
```

Generates `outputs\<stem>\<stem>_annotated.mp4`. The red dot should stay on the
bell centre; the green dot should follow the dye mark throughout.

#### Step 4 — Calibrate rhopalium positions (one-time per animal)

Take a high-resolution still photo of the jellyfish with the dye mark visible.

```powershell
.\venv\Scripts\python.exe scripts\calibrate_rhopalia.py "C:\path\to\photo.jpg"
```

Three-step click process: bell centre → dye mark (0° reference) → each rhopalium.
BACKSPACE to undo, ENTER when done.

**Outputs:**

- `calibration\photo_name.json` — permanent rhopalium body-frame angle record
- `calibration\photo_name_annotated.png` — labelled verification diagram

#### Step 5 — Pulse initiation analysis (Approach B)

```powershell
.\venv\Scripts\python.exe scripts\run_approach_b.py
```

Phase 1 computes `_margin_diff_lab.npy` (lab-frame, streams from video — no JPEG
extraction needed) then `_margin_diff.npy` (body-frame, requires CoTracker output).
Phase 1 is cached and reused on re-runs unless `--recompute` is passed.

**Tunable parameters:**

| Parameter | Default | Effect |
| --- | --- | --- |
| `--inner-frac` | 0.75 | Inner edge of margin ring (fraction of bell radius) |
| `--outer-frac` | 1.05 | Outer edge — values > 1.0 include outside-bell expansion signal |
| `--pre-window` | 30 | Frames before peak to scan for initiation |
| `--min-distance` | 0.42 s | Minimum time between pulses |
| `--prominence` | 0.05 | Peak detection threshold (fraction of signal range) |
| `--recompute` | off | Force recompute margin_diff (needed when changing inner/outer-frac) |

---

## SAM2 model selection

Four model sizes are available. The `tiny` model is recommended for most
recordings — it is ~2× faster than `base` with no measurable quality difference
on the high-contrast Cassiopea bell:

| Model | Weights | Speed (RTX 4060) | Use when |
| --- | --- | --- | --- |
| `tiny` | 148 MB | ~8.5 fps | **Recommended default** |
| `small` | 185 MB | ~6 fps | Marginal accuracy improvement |
| `base` | 308 MB | ~5.6 fps | If tiny shows mask drift |
| `large` | ~900 MB | ~2–3 fps | Maximum accuracy, long recordings |

Select with `--sam2-model tiny` (or set as default in
[config.py](config.py)):

```python
SAM2_WEIGHTS = WEIGHTS_DIR / "sam2" / "sam2.1_hiera_tiny.pt"
SAM2_CONFIG  = "configs/sam2.1/sam2.1_hiera_t.yaml"
```

---

## Performance

### Processing time estimates

For a **10-minute recording at 120 fps**, stride = 4 (30 fps effective):

| Hardware | SAM2 tiny | CoTracker | Approach B | Total |
| --- | --- | --- | --- | --- |
| RTX 4060 laptop (sequential) | ~45 min | ~110 min | ~5 min | **~2.5–3 hrs** |
| RTX 4090 desktop (parallel) | ~15 min | ~25 min | ~3 min | **~30–40 min** |

### Active performance optimisations

- **bfloat16 inference** — CoTracker runs in mixed precision, using the GPU's tensor
  cores (~1.5–2× faster than float32)
- **`torch.compile`** — both CoTracker and the SAM2 image encoder are JIT-compiled on
  the first run; subsequent chunks benefit from fused kernels (~15–25% faster)
- **Parallel task scheduler** — `src/scheduler.py` runs SAM2, CoTracker, and
  Approach B Phase 1 concurrently, gated by available VRAM
- **Streaming video decode** — Approach B reads directly from the source video;
  no JPEG extraction needed
- **Automatic frame cleanup** — extracted JPEG frames (needed by SAM2) are deleted
  after segmentation completes, saving 5+ GB per recording

### Stride guidelines

| Stride | Effective fps | Processing time | Temporal resolution |
| --- | --- | --- | --- |
| 1 | 120 fps | ~15 hrs (10 min clip, RTX 4060) | Maximum |
| 4 | 30 fps | ~2.5–3 hrs | Recommended — pulses are ~1 s long |
| 8 | 15 fps | ~1.5 hrs | Faster; marginal accuracy tradeoff |

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

All outputs for a recording land in their own subfolder:

```text
outputs\
  <stem>\
    <stem>_frames\               TEMP: JPEG frames for SAM2, auto-deleted after use
    <stem>_masks\                20 sample PNG masks for visual spot-check
    <stem>_seg.csv               Bell centre + radius per frame  (SAM2)
    <stem>_contour_radii.npy     Bell boundary r(θ, t)  (SAM2)
    <stem>_track.csv             Dye mark position per frame  (CoTracker)
    <stem>_tracked.mp4           Dye mark overlay video
    <stem>_margin_diff_lab.npy   TEMP: lab-frame margin diff (deleted after Phase 1b)
    <stem>_margin_diff.npy       Body-frame margin activity cache  (Approach B)
    <stem>_initiation_b.csv      Per-pulse rhopalium assignments
    <stem>_initiation_b_plot.png Activity signal + firing histogram
    <stem>_initiation_b_annotated.mp4  Full video with pulse labels
    <stem>_spacetime_pulse_b\    Per-pulse angle × time heatmaps
      pulse_000.png
      pulse_001.png
      ...
    <stem>_annotated.mp4         Validation video (dye + centroid overlay)
    <stem>_sam2_validation.png   SAM2 segmentation spot-check

calibration\
  <animal>.json                  Rhopalium body-frame angles  (permanent record)
  <animal>_annotated.png         Labelled calibration diagram
```

---

## Project structure

```text
src\
  resources.py     GPU detection, GpuGate semaphore
  scheduler.py     DAG task runner with parallel execution and UI-ready callbacks
  tasks.py         Task factory functions wrapping each pipeline stage
  pipeline.py      run_pipeline() — assembles and executes the full pipeline

scripts\
  run_pipeline.py       Full pipeline CLI (primary entry point)
  run_sam2.py           SAM2 bell segmentation (standalone)
  cotracker_test.py     CoTracker dye tracking (standalone)
  run_approach_b.py     Approach B pulse initiation analysis (standalone)
  calibrate_rhopalia.py One-time rhopalium calibration from hi-res photo
  validate_tracking.py  Side-by-side validation video generator

calibration\            Rhopalium calibration files (tracked in git)
data\                   Short test clips (gitignored)
outputs\                All pipeline outputs (gitignored)
weights\                Model weights (gitignored)
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
