# Jellyfish Behavior Analysis Pipeline

Video analysis pipeline for extracting orientation, contraction dynamics, and pulse-initiation timing from top-view recordings of *Cassiopea* (upside-down jellyfish).

---

## Species context

*Cassiopea* sits bell-down on the substrate and pulses from a fixed orientation. Unlike *Aurelia*, it has **no high-contrast anatomical landmark** (no gonad cross) that can serve as a rotational reference. All orientation tracking is anchored to an **artificial dye mark** applied to the bell.

---

## Hardware requirements

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4060 (8 GB VRAM) — tested |
| RAM | 32 GB |
| CUDA | 12.x driver |
| Python | 3.10+ |

---

## Pipeline overview

```
Video recording (120 fps, 640×512, H.264)
        │
        ▼
Stage 1 ── SAM2 bell segmentation
        │   per-frame binary mask + centroid C = (cx, cy)
        │
        ▼
Stage 2 ── CoTracker dye tracking   ← [IMPLEMENTED]
        │   per-frame dye position D = (dx, dy)
        │   body axis: φ_dye = atan2(dy−cy, dx−cx)
        │
        ▼
Stage 3 ── Rhopalium detection (polar unwrap)
        │   8 body-frame angles {φ_rho_0 … φ_rho_7}
        │
        ▼
Stage 4 ── Rhopalium identity assignment
        │   identity ← angular position relative to dye axis (persistent)
        │
        ▼
Stage 5 ── RAFT optical flow + divergence
        │   div(x,y,t) = ∂u/∂x + ∂v/∂y  over bell mask
        │   contraction_signal(t) = ΣΣ div(x,y,t)
        │
        ▼
Stage 6 ── Pulse initiation analysis
            for each pulse peak t*:
              pre_pulse = Σ div[:,:,t*−k]  k=1..5
              origin_xy = argmax(pre_pulse · mask)
              φ_origin  = atan2(origin_y−cy, origin_x−cx) − φ_dye
              initiator = argmin_i |φ_rho_i − φ_origin| mod 2π
```

---

## Stage 1 — Bell segmentation (SAM2)

**Model**: SAM2 (`sam2.1_hiera_b+`, ~308 MB)
**Approach**: single click-prompt on frame 0; mask propagated through entire recording using SAM2's video predictor.

**Outputs**:
- Per-frame binary mask `M(x,y,t)`
- Centroid `C(t) = (cx, cy)` from mask moments
- Bell radius estimate `R(t)` from mask area

**Why SAM2**: *Cassiopea* is a bright disc on dark water — high contrast makes prompt-based segmentation reliable. SAM2's temporal propagation handles the occasional frame where contrast drops.

---

## Stage 2 — Dye mark tracking (CoTracker)

**Model**: CoTracker3 offline (`scaled_offline.pth`, ~97 MB, CC-BY-NC)
**Approach**: user clicks dye mark in frame 0; CoTracker propagates the point through the video using a sliding temporal window, exploiting multi-frame context to track low-contrast features.

**Key math**:
```
body axis:   φ_dye(t) = atan2(D_y(t) − C_y(t),  D_x(t) − C_x(t))

body-frame transform for any point P:
  φ_body(P, t) = atan2(P_y − C_y,  P_x − C_x) − φ_dye(t)
```
This transform is invariant to both **translation** and **rotation** of the animal.

**Outputs**:
- `outputs/<stem>_track.csv` — `frame_idx, timestamp_s, x, y, visible`
- `outputs/<stem>_tracked.mp4` — annotated video

**License note**: CoTracker3 is CC-BY-NC. For commercial use, swap to TAPIR (Apache 2.0).

---

## Stage 3 — Rhopalium detection

**Approach**: polar unwrap of the bell region around centroid C, then 1D peak detection on the angular intensity profile at r ≈ R_bell.

```python
polar = cv2.warpPolar(masked_frame, (360, R_bell), (cx, cy),
                      R_bell, cv2.WARP_POLAR_LINEAR)
ring    = polar[:, int(0.85 * R_bell):]   # outer annulus
profile = ring.mean(axis=1)               # 1D angular intensity
peaks   = scipy.signal.find_peaks(profile, distance=30, prominence=5)
```

**Identity assignment**: rhopalium *i* is identified by its body-frame angle:
```
φ_rho_i = φ_lab_i − φ_dye(t)
```
These body-frame angles are constant across time (rhopalia are fixed to the bell), enabling persistent identity without re-detection every frame.

**Fallback**: if peak detection is insufficient, a small 2D detector on DINOv2 features trained on ~100–200 labeled frames.

---

## Stage 4 — Rhopalium identity (angular)

8 rhopalia are labeled R0–R7 by ascending body-frame angle from the dye axis. Labels are consistent across:
- All frames within a recording (rotation-invariant)
- Across recordings if dye placement is consistent

---

## Stage 5 — Contraction via optical flow divergence

**Model**: RAFT-small (torchvision)
**Why divergence over area**: flow divergence `div = ∂u/∂x + ∂v/∂y` is a **physical quantity** (local volume change rate) and is insensitive to rigid-body translation/rotation. Silhouette area conflates contraction with tilt, depth change, and tentacle motion.

**Processing**:
```
for each consecutive frame pair (t, t+1):
    (u, v) = RAFT(frame_t, frame_{t+1})
    div     = ∂u/∂x + ∂v/∂y          # finite difference
    div_masked = div * M(x,y,t)        # restrict to bell
    contraction_signal[t] = ΣΣ div_masked
```

**Performance note**: RAFT is the most compute-intensive stage (one full network pass per frame pair). For 1-hour recordings at 120 fps, use:
- RAFT-small (3–5× faster than RAFT-large)
- Stride 4 (30 fps effective — sufficient for pulse timescales of ~1–2 s)
- Stream-and-discard: compute divergence immediately, store only the scalar signal + keyframe divergence maps

---

## Stage 6 — Pulse initiation analysis

For each contraction peak at time `t*`:

```python
# 1. Accumulate divergence in pre-pulse window (before wave has spread)
pre_pulse = sum(div_field[:, :, t* - k] for k in range(1, 6))

# 2. Find spatial origin (within bell mask)
origin_xy = np.unravel_index(np.argmax(pre_pulse * mask), pre_pulse.shape)

# 3. Convert to body frame
phi_origin = atan2(origin_y - cy, origin_x - cx) - phi_dye

# 4. Assign to nearest rhopalium
angular_dist = [(phi_origin - phi_rho_i) % (2*pi) for i in range(8)]
initiator = argmin(angular_dist)
```

**Statistical outputs** across all pulses in a recording:
- Histogram of initiating rhopalium → dominant pacemaker identification
- Inter-pulse interval per rhopalium → individual firing rates
- Sequential patterns → do rhopalia alternate? suppress each other?

---

## Body-frame coordinate system

```
                  φ_dye (dye mark, body-frame 0°)
                    •  ← D(t)
                   /
                  /  ← body axis
                 /
                •  ← C(t)  (centroid)
               / \
              /   \
      φ_rho_7     φ_rho_1   (body-frame rhopalium angles, fixed)
```

All angular quantities are expressed relative to `φ_dye`, making them invariant to the animal's rotation in the lab frame.

---

## Setup

```powershell
cd d:\Jellyfish
.\setup.ps1           # creates venv, installs all deps, downloads weights, runs smoke test
```

Or step by step:
```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
.\venv\Scripts\pip.exe install -r requirements.txt
.\venv\Scripts\pip.exe install "git+https://github.com/facebookresearch/sam2.git"
.\venv\Scripts\pip.exe install "git+https://github.com/facebookresearch/co-tracker.git"
.\venv\Scripts\python.exe scripts\test_cuda.py
```

Edit [config.py](config.py) to set `VIDEO_DIR` to wherever your recordings live.

---

## Usage

### Crop a test clip
```powershell
& "C:\Users\yhhua\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe" `
    -i "D:\path\to\recording.mp4" -ss 00:00:00 -t 60 -c copy data\test_clip_1min.mp4
```

### Stage 2 — Track dye mark
```powershell
# Full resolution (120 fps) — ~43 min on laptop RTX 4060
.\venv\Scripts\python.exe scripts\cotracker_test.py

# Stride 4 (30 fps effective) — ~11 min, sufficient for go/no-go
.\venv\Scripts\python.exe scripts\cotracker_test.py --stride 4
```

### Validate tracking
```powershell
.\venv\Scripts\python.exe scripts\validate_tracking.py
# outputs/test_clip_1min_validation.png
```

---

## Output files

| File | Contents |
|---|---|
| `outputs/<stem>_track.csv` | `frame_idx, timestamp_s, x, y, visible` — dye mark position per frame |
| `outputs/<stem>_tracked.mp4` | Annotated video: green dot=visible, purple=occluded, amber trail |
| `outputs/<stem>_validation.png` | Mosaic: raw frame + schematic overlay for spot-checking |

---

## Open questions

1. **Recording FPS confirmed**: 120 fps (from `ffprobe`).
2. Is camera position fixed across recording sessions? (needed for chamber-wall external reference)
3. How visually distinct are rhopalia in analysis-grade footage? (determines Stage 3 approach)
4. Is cross-recording rhopalium identity needed? (determines whether dye placement must be consistent across days)
5. Out-of-plane tilt: add a mask aspect-ratio sanity check to flag tilted frames.

---

## License

- **SAM2**: Apache 2.0
- **CoTracker3**: CC-BY-NC 4.0 — research use only; swap to TAPIR (Apache 2.0) if commercial use is needed
- **RAFT** (torchvision): BSD
