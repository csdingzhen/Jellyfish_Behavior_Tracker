# Cassiopea Behavior Analysis Pipeline

Automated video analysis pipeline for *Cassiopea* (upside-down jellyfish) recordings. Given a top-view video of the animal with a small dye mark applied to its bell, this pipeline extracts:

- **Bell orientation over time** — which way the animal is facing at each moment
- **Contraction signal over time** — when the bell pulses and how strongly
- **Pulse initiation site** — which rhopalium (marginal neurocluster) triggered each contraction

---

## Background

*Cassiopea* has 16 rhopalia distributed around its bell margin. These act as independent pacemakers — any one of them can trigger a contraction wave that spreads across the whole bell. A key scientific question is: which rhopalium fires most often, and do they follow a pattern?

Because the animal has no obvious anatomical landmark for orientation (unlike *Aurelia*, which has a visible gonad cross), we apply a small **dye mark** to the bell surface. This mark serves as a reference point from which all angular measurements are made.

---

## How it works

```text
Your video
    │
    ├─► Track the dye mark through every frame
    │       → gives the bell orientation at each moment
    │
    ├─► Outline the jellyfish bell in every frame
    │       → gives the bell centre, size, and boundary
    │
    ├─► [One-time] From a hi-res photo, record where each rhopalium sits
    │       → gives fixed angular positions relative to the dye mark
    │
    ├─► Measure how pixels move between consecutive frames
    │       → contracting regions show pixels moving inward
    │       → summing this over the bell gives a contraction signal over time
    │
    └─► For each contraction peak, find where the wave first appeared
            → convert that location to an angle relative to the dye mark
            → match to the nearest rhopalium
            → that rhopalium initiated the pulse
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
- Fluorescent dye under UV illumination gives best contrast, but standard dye also works
- Place the mark off-centre so it can define an orientation angle

---

## Installation

All commands are run in **PowerShell** from the project folder.

### 1. Clone or download this repository

```powershell
cd C:\Projects
git clone <repo-url> Jellyfish
cd Jellyfish
```

### 2. Run the setup script

Creates a virtual environment, installs all packages, downloads model weights (~400 MB), and runs a GPU check:

```powershell
.\setup.ps1
```

After it finishes you should see:

```text
All checks passed. Environment is ready.
```

### 3. Set your video folder

Open [config.py](config.py) and change `VIDEO_DIR` to wherever your recordings live:

```python
VIDEO_DIR = Path(r"C:\Users\YourName\Videos\Cassiopea")
```

---

## Step-by-step guide

### Prepare a test clip

Working on the full recording (often 1+ hours) is slow during development. Crop a 60-second clip first using ffmpeg:

```powershell
& "C:\path\to\ffmpeg.exe" -i "C:\Users\YourName\Videos\recording.mp4" `
    -ss 00:00:00 -t 60 -c copy data\test_clip.mp4
```

---

### Step 1 — Segment the bell (SAM2)

This step outlines the jellyfish bell in every frame using an AI segmentation model. It produces a per-frame centre point and size estimate.

```powershell
.\venv\Scripts\python.exe scripts\run_sam2.py --video data\test_clip.mp4
```

A window opens showing the first frame. **Click anywhere on the bell body** (not the edge, not the tentacles), then press **ENTER**. The model tracks the bell outline through the full clip automatically.

For long recordings, use `--window-size 200` to limit memory use per batch.

**Outputs:**

- `outputs\test_clip_seg.csv` — bell centre (cx, cy) and radius for each frame
- `outputs\test_clip_sam2_validation.png` — spot-check image showing segmentation quality

---

### Step 2 — Track the dye mark (CoTracker)

This step tracks the dye mark through every frame using a point-tracking AI model. Even faint or low-contrast marks can be followed using temporal context across many frames.

```powershell
.\venv\Scripts\python.exe scripts\cotracker_test.py --video data\test_clip.mp4
```

A window opens. **Click on the dye mark** in the first frame, then press **ENTER**.

For a faster first test run, process every 4th frame (30 fps effective):

```powershell
.\venv\Scripts\python.exe scripts\cotracker_test.py --stride 4
```

**Outputs:**

- `outputs\test_clip_track.csv` — dye mark position (x, y) and confidence for each frame
- `outputs\test_clip_tracked.mp4` — video with dye mark overlaid

---

### Step 3 — Validate tracking

Generate a combined annotated video showing both the bell outline and dye mark on every frame. This is the key quality check before proceeding.

```powershell
.\venv\Scripts\python.exe scripts\validate_tracking.py
```

**What to look for:**

- The red dot (bell centre) should stay on the animal throughout
- The green dot (dye mark) should follow the mark on the bell
- The body-axis angle (`phi`) shown in the HUD should be stable during resting

**Output:** `outputs\test_clip_annotated.mp4`

---

### Step 4 — Calibrate rhopalium positions (one-time per animal)

Take a **high-resolution still photo** of the jellyfish with the dye mark visible. Run the calibration tool to record where each rhopalium sits relative to the dye mark. This only needs to be done once — rhopalium positions are fixed to the bell.

```powershell
.\venv\Scripts\python.exe scripts\calibrate_rhopalia.py "C:\path\to\photo.jpg"
```

**Three-step process in the window:**

| Step | Action | Key |
| --- | --- | --- |
| 1 | Click the bell centre of mass | ENTER |
| 2 | Click the dye mark (becomes the 0° reference) | ENTER |
| 3 | Click each rhopalium one by one | BACKSPACE to undo, ENTER when done |

**Outputs:**

- `calibration\photo_name.json` — rhopalium body-frame angles (used by Step 6)
- `calibration\photo_name_annotated.png` — labelled diagram for verification

---

### Step 5 — Compute the contraction signal (RAFT)

This step measures pixel motion between consecutive frames and quantifies how much the bell is contracting at each moment. The sum of inward motion across the bell gives a contraction signal over time.

```powershell
.\venv\Scripts\python.exe scripts\run_raft.py
```

Processing takes roughly 2–5 minutes for a 1-minute clip at 120 fps.

**Outputs:**

- `outputs\test_clip_contraction.csv` — contraction strength per frame
- `outputs\test_clip_peaks.csv` — detected pulse events (frame, time, strength)
- `outputs\test_clip_contraction_plot.png` — signal plot with peaks marked — **check this first**

Tuning peak detection if the plot looks wrong:

```powershell
# Adjust minimum gap between pulses (default 0.42 s):
.\venv\Scripts\python.exe scripts\run_raft.py --min-distance 0.5

# Raise the detection threshold to reduce false peaks:
.\venv\Scripts\python.exe scripts\run_raft.py --prominence 0.1
```

---

### Step 6 — Identify pulse initiators

For each detected contraction peak, this step finds where the contractile wave first appeared, converts that location to a body-frame angle, and matches it to the nearest rhopalium.

```powershell
.\venv\Scripts\python.exe scripts\run_stage6.py
```

**Outputs:**

- `outputs\test_clip_initiation.csv` — per-pulse rhopalium assignments
- `outputs\test_clip_initiation_plot.png` — which rhopalium fires most + firing timeline
- `outputs\test_clip_initiation_annotated.mp4` — full video with pulse initiation sites labelled

**Reading the initiation CSV:**

| Column | Meaning |
| --- | --- |
| `peak_frame` | Video frame at peak contraction |
| `timestamp_s` | Time in seconds |
| `rhopalium_id` | Which rhopalium initiated (0–15) |
| `phi_origin_body` | Angle of initiation site in body frame (degrees) |
| `angular_dist_deg` | Gap to matched rhopalium — smaller = more confident |

A well-functioning result has `angular_dist_deg` consistently below ~11° (half the 22.5° spacing between 16 evenly distributed rhopalia).

---

## Known limitations

### Pulse initiation detection

The initiation analysis depends on detecting where the contractile wave first appears in the optical flow field. This works best with high frame rate (≥ 60 fps recommended), good image contrast in the bell texture, and a clear bell-background boundary. In recordings with low contrast or motion blur, the `angular_dist_deg` column indicates reliability — values above 20° suggest the match is uncertain and should be reviewed using the annotated video.

### Dye mark contrast

Very faint dye marks can be tracked reliably by CoTracker if they are visible in at least the first few frames. Adding a fluorescent dye with UV illumination dramatically improves contrast and tracking confidence.

### Long recordings

For recordings longer than a few minutes:

- SAM2 processes in batches of 200 frames to stay within memory limits
- CoTracker processes in chunks of 200 frames
- RAFT streams frame-by-frame so length is not a memory constraint

---

## Output file reference

```text
outputs\
  <clip>_seg.csv                   bell centre and radius per frame  (SAM2)
  <clip>_track.csv                 dye mark position per frame  (CoTracker)
  <clip>_contraction.csv           contraction signal per frame  (RAFT)
  <clip>_peaks.csv                 detected pulse events
  <clip>_initiation.csv            per-pulse rhopalium assignments
  <clip>_annotated.mp4             validation video  (dye + centroid)
  <clip>_sam2_validation.png       SAM2 segmentation spot-check
  <clip>_contraction_plot.png      contraction signal plot
  <clip>_initiation_plot.png       which rhopalium fires when
  <clip>_initiation_annotated.mp4  full video with pulse labels

calibration\
  <animal>.json                    rhopalium body-frame angles  (permanent record)
  <animal>_annotated.png           labelled calibration diagram
```

---

## License notes

- **SAM2** (bell segmentation): Apache 2.0
- **CoTracker** (dye tracking): CC-BY-NC 4.0 — research use only. For commercial use, replace with TAPIR (Apache 2.0).
- **RAFT** (optical flow, via torchvision): BSD

---

## Citation

Models used:

- SAM2: Ravi et al., Meta AI (2024)
- CoTracker: Karaev et al., Meta AI + Oxford VGG (2023)
- RAFT: Teed & Deng, Princeton (2020)
