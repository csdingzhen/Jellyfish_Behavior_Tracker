# Jellyfish Behavior Analysis — Project Memory

## Project goal

Build a video analysis pipeline that, from top-view recordings of **Cassiopea** (upside-down jellyfish), extracts:

1. **Orientation over time** — absolute rotational angle of the bell, derived against a stable body-frame reference.
2. **Contraction over time** — quantitative measure of bell pulsing, with magnitude, and ideally spatial resolution (uniform vs local contractions).
3. **Pulse initiation** — which rhopalium (neurocluster on the bell margin) initiated each contraction wave. Rhopalia act as pacemakers, so this is a central scientific question.
4. *(Optional)* **Per-frame state labels** — e.g. pulsing / resting / drifting / escape-contracting.

## Species-specific context (important)

The species is **Cassiopea**, not *Aurelia*. This matters:
- **No gonad cross.** Cassiopea does not have *Aurelia*'s four-leaf gonad pattern, so we have no high-contrast internal anatomical landmark for orientation.
- **Bell appears as a relatively uniform pale disc** with subtle internal texture (zooxanthellae symbiont distribution). This texture is helpful for optical flow but not for landmark detection.
- **Mostly sessile.** Cassiopea typically pulses from a fixed orientation rather than free-swimming, so identity tracking through rapid rotation is less of a concern than for free-swimming species.
- **Rhopalia (8, at the bell margin in clefts between marginal lappets) are low-contrast** in current footage. They may or may not be reliably detectable depending on image quality.

## Hardware

- 32 GB system RAM
- NVIDIA RTX 4060 (8 GB VRAM)
- Local development, no cloud GPU planned.

Implications: prefer ViT-B/14 or smaller backbones; mind batch sizes (8 GB is the ceiling); CoTracker and SAM2 both fit comfortably; disk I/O may bottleneck long recordings before GPU does.

## What has been tried and failed — do not re-propose

### DeepLabCut (keypoint regression on rhopalia)
- **Permutation symmetry**: rhopalia are interchangeable; they have no semantic identity like "left eye." DLC's per-keypoint identity assumption produces inconsistent training labels.
- **No rotation equivariance**: standard backbones must relearn each rotation independently.
- **Translucency / low contrast**: weak local features for the CNN heatmap regressor to lock onto.
- **Deformation**: tissue around each rhopalium stretches during pulses, so "local appearance" is not stable.

### Classical CV for contraction (thresholding + ellipse fit + area)
- Translucency → no hard silhouette edge; thresholding brittle to lighting.
- Ellipse-fit assumption breaks during contraction.
- Single scalar metrics (area, diameter) conflate true contraction with tilt, depth changes, and tentacle motion.

### CBAS (jones-lab-tamu/CBAS)
- Evaluated. CBAS = frozen DINOv2/DINOv3 ViT + LSTM head for video **behavior classification**.
- Not suitable as the primary tool: it outputs class labels, not coordinates or angles.
- Possibly useful as a *downstream* binary/multi-class classifier on top of the geometric pipeline.

### Anatomy-as-fiducial (gonad cross)
- Considered for Aurelia where the cross is striking. **Does not apply to Cassiopea** — no equivalent landmark exists.

## Fiducial strategy: dye mark + CoTracker

Because Cassiopea lacks an anatomical reference frame, we use an **artificial fiducial: a dye mark applied to the bell**. The mark provides the rotational reference that gonads provided for Aurelia.

### Current state
- A dye mark is being applied experimentally. Contrast is currently **faint**, and wet-lab improvements are in progress.
- However, we should not assume the dye must be high-contrast — modern point trackers can lock onto low-contrast features given temporal context.

### Wet-lab improvements worth pursuing (in priority order)
1. **Fluorescent labeling** with a violet/UV excitation light and emission filter on the camera. Single biggest possible contrast win (10–100x). Cassiopea zooxanthellae autofluoresce in red, so a green/yellow fluorescent dye is spectrally separable.
2. **Larger / higher-contrast single mark** beats multiple small marks. One fiducial is sufficient to break rotational symmetry.
3. **Mark placement near bell center** is more stable; near bell edge gives better angular precision but is easier to lose.
4. **Periodic color reference frames** even if the primary recording is grayscale — preserves spectral separability for re-identification.

### Detection strategy: CoTracker

**CoTracker** (`facebookresearch/co-tracker`, Meta AI + Oxford VGG, CC-BY-NC license) is a transformer-based video point tracker. It tracks arbitrary pixels through video using temporal context, and is specifically robust to low-contrast features because it aggregates evidence across many frames.

Workflow:
1. Manually click the dye mark in 1–2 frames where it's most visible.
2. CoTracker propagates the point through the entire video, including frames where the dye is barely visible to a human.
3. Output: per-frame (x, y) of the dye mark, with confidence.

**License note**: CC-BY-NC is fine for academic research but blocks commercial use. If commercialization becomes relevant, swap to **TAPIR** (Apache 2.0) — similar capability, more permissive.

Why CoTracker before further wet-lab improvements: if it works on existing faint-dye footage, we save the protocol-change effort. Test this *first*.

### Fallback / hybrid: temporal averaging in body frame
If CoTracker alone is insufficient:
- Use SAM2 mask + centroid to co-register all frames into a common bell-centered coordinate system.
- Average many co-registered frames → dye signal accumulates, random noise cancels.
- Variance map across registered frames also highlights the dye location.

## Proposed architecture

A segmentation-first pipeline with a dye-anchored body coordinate frame.

### Stage 1 — Bell segmentation
- **SAM2** with click-prompt on first frame; propagate the mask through video.
- Output: per-frame binary mask of the bell.
- For Cassiopea this is cleaner than for Aurelia — bright disc on dark water, high contrast.

### Stage 2 — Centroid & body axis
- Centroid from the bell mask.
- **Dye position from CoTracker.**
- **Body axis** = unit vector from centroid to dye position.
- This gives an absolute body-frame orientation per frame, with no anatomical landmark needed.

### Stage 3 — Rhopalia detection
- Detect rhopalia as a **single anonymous class** (not 8 identified keypoints).
- Approach options, in order:
  1. **Polar unwrap** around centroid + classical peak detection on the angular intensity profile.
  2. If peak detection is insufficient: small 1D CNN on the polar strip.
  3. If image contrast is the bottleneck: train a small 2D detector (DINOv2 features + tiny regression head) on ~100–200 labeled frames. Single-class task means small data requirements.
- Restrict search to within the SAM2 mask to suppress chamber-wall reflections.

### Stage 4 — Rhopalia identity assignment
- Each rhopalium identified by its **angular position relative to the body axis** (dye-anchored).
- Identity is then persistent across the recording, across rotations, and meaningfully comparable across recordings if dye placement is consistent.

### Stage 5 — Contraction
- **RAFT optical flow** between consecutive frames, masked by the Stage 1 mask.
- **Divergence of the flow field** = local contraction/expansion rate (physical quantity).
- Integrate divergence over the bell mask → per-frame scalar contraction signal.
- Keep the 2D divergence field → spatially resolved contraction maps.
- Mask-area time series as cheap secondary sanity-check signal.
- Cassiopea's zooxanthellae texture gives optical flow good features to track.

### Stage 6 — Pulse initiation analysis
- For each pulse (peak in integrated divergence):
  1. Examine the divergence field a few frames *before* the peak.
  2. Locate the spatial origin of the divergence wave.
  3. Map to the nearest rhopalium (using Stage 3 positions).
  4. Convert to body-frame angle (via Stage 2 dye axis) for cross-pulse/cross-recording statistics.
- **Note**: this analysis works per-pulse independently — does not require persistent identity tracking across the recording. The dye-anchored body axis lets us label initiators by body-frame angle directly.

### Stage 7 — *(Optional)* state classification
- If discrete labels are needed (pulsing / resting / drifting), train a CBAS-style classifier (frozen DINOv2 + LSTM head).
- **Window-size warning**: Cassiopea pulses are sub-second. Shrink LSTM window much shorter than CBAS defaults.

## External reference frame: chamber walls

The recording chamber has a segmented transparent wall with gaps at fixed angular positions. If the camera is fixed relative to the chamber (it appears to be), the chamber provides an **external lab-frame reference**. This is independent of the dye fiducial and useful for:
- Sanity-checking orientation estimates.
- Detecting camera drift or chamber rotation.
- Providing a backup reference if the dye mark is temporarily occluded or fades.

A simple template-match against the chamber wall pattern at recording start gives a fixed lab-frame coordinate system.

## Design principles

1. **Don't fight symmetries.** Radial structure is exploitable; polar coordinates and dye-anchored angles leverage it.
2. **Don't track identity from appearance.** Identity comes from geometry (angular position relative to fiducial), not local features.
3. **Prefer physical quantities to geometric proxies.** Flow divergence > silhouette-area change.
4. **Segment first, derive everything from the mask.**
5. **Use the dye as the body-frame anchor**, not as a per-rhopalium identifier.
6. **Per-pulse analyses are easier than persistent-identity analyses** — frame the science around per-pulse questions when possible.

## Tech stack

- Python 3.10+
- PyTorch with CUDA 12.x (matched to RTX 4060 driver)
- **SAM2** — `facebookresearch/sam2`
- **CoTracker** — `facebookresearch/co-tracker` (CC-BY-NC — research use only)
  - Alternative if commercial relevance: **TAPIR** (Apache 2.0)
- **RAFT** optical flow — torchvision has a port, or `princeton-vl/RAFT`
- OpenCV — polar unwrap (`cv2.warpPolar`), contour ops, template matching
- scipy — Hungarian (`scipy.optimize.linear_sum_assignment`), peak finding (`scipy.signal.find_peaks`)
- *(Optional, Stage 7)* CBAS v3 branch — `jones-lab-tamu/CBAS`

## Open questions to resolve early

1. **Recording FPS**: pulses are sub-second; need ≥30 FPS, ideally higher.
2. **Is the camera position fixed across recordings?** Looks like yes — needed for chamber-as-external-reference to work.
3. **How visually distinct are the rhopalia in analysis-grade footage?** Determines Stage 3 implementation (peak-finder vs learned detector).
4. **How faint is the dye really?** Run CoTracker on existing footage with a manually-clicked dye location in frame 1 — does it track reliably? This decides whether fluorescent labeling is worth the protocol change.
5. **Cross-recording identity needed?** I.e., does "rhopalium A on day 1 = rhopalium A on day 5" matter scientifically? If yes, dye placement must be consistent across recordings; if no, per-recording angular labeling suffices.
6. **Out-of-plane tilt**: top-view assumes horizontal bell. Add a sanity check (mask aspect ratio variance) to flag tilted frames.

## Concrete first-week milestones

1. Set up environment; verify CUDA works with SAM2, CoTracker, and RAFT on the 4060.
2. Run SAM2 on a sample clip; visually inspect mask quality across a full pulse cycle.
3. **Run CoTracker on the existing faint-dye footage**: manually click the dye in frame 1, verify it propagates reliably through the recording. This is the critical go/no-go test for the wet-lab side.
4. Implement polar unwrap and visualize unwrapped strips at several phases of contraction.
5. Sanity-check rhopalia peak detection (classical first, learned only if needed).
6. Run RAFT on a sample sequence; plot divergence-integrated-over-mask vs time and confirm peaks align with visible pulses.
7. Implement Stage 6 (pulse initiation): for one identified pulse, locate divergence origin and map to nearest rhopalium. Confirm the answer is plausible.
8. Decide based on results whether to invest in fluorescent dye improvements or whether current contrast is sufficient.

## How to work on this project

- Prefer minimal, debuggable building blocks over end-to-end systems.
- Plot intermediate signals at every stage — masks, dye trajectories, polar strips, divergence fields, peak detections overlaid on frames.
- Keep raw + intermediate outputs cached on disk; recomputing SAM2, CoTracker, and RAFT is expensive.
- When in doubt about a design choice, choose the option that better respects the radial symmetry of the animal and uses the dye-anchored body frame.
- The single highest-priority empirical test is whether CoTracker can track the existing faint dye. Run that test before doing anything else expensive.
