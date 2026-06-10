# Project Overview

## Goal

Build a video analysis pipeline that, given top-view recordings of **Cassiopea** (upside-down jellyfish), automatically extracts:

1. **Orientation over time** — absolute rotational angle of the bell in a body-fixed coordinate frame.
2. **Contraction timing** — per-frame magnitude of bell pulsing.
3. **Pulse initiation site** — which of the 16 rhopalia (marginal neuroclusters) triggered each contraction wave.

## Why this matters scientifically

Cassiopea rhopalia are independent pacemakers. Any one of them can initiate a contraction wave. The central question is whether initiation is random, biased toward specific rhopalia, or modulated by sensory state. Answering this requires a body-frame coordinate system so that "rhopalium 5 initiated this pulse" means the same thing across recordings and across animals.

## Species context

The species is **Cassiopea** (upside-down jellyfish), not *Aurelia* (moon jellyfish). Key differences that constrain the technical approach:

- **No anatomical landmark.** Aurelia has a four-leaf gonad cross that can serve as an orientation reference. Cassiopea has no equivalent. We use an **artificial dye mark** applied to the bell surface as the body-frame anchor.
- **Bell is a pale uniform disc** with subtle zooxanthellae texture. Good for optical flow; poor for keypoint regression.
- **16 rhopalia** (not 8) distributed at the clefts between marginal lappets. They are low-contrast in current footage.
- **Mostly sessile** — pulses from a near-fixed position. Rotation happens but is slow. Rapid translation and identity loss (as in free-swimming medusae) is not a concern.

## Recording setup

- Camera is **fixed** relative to the recording chamber.
- 120 fps at 640×512, grayscale or colour.
- Chamber has a segmented transparent wall with gaps at fixed angles — usable as an external lab-frame reference if needed.
- A small dye mark is applied to the bell. Fluorescent dye + UV illumination is the highest-contrast option.

## What has been tried and ruled out

See [03_design_decisions.md](03_design_decisions.md) for the full list. Short summary:

- **DeepLabCut** — per-keypoint identity assumption incompatible with radially symmetric, interchangeable rhopalia.
- **Classical CV** (thresholding + ellipse fit) — translucency makes thresholding brittle.
- **CBAS** — outputs class labels, not coordinates; useful only as a downstream classifier.
- **Gonad cross as fiducial** — does not exist in Cassiopea.

## Recommended reading order for new developers

1. [00_overview.md](00_overview.md) — this file
2. [01_pipeline_architecture.md](01_pipeline_architecture.md) — what each stage does
3. [02_codebase_map.md](02_codebase_map.md) — which file to edit for what
4. [03_design_decisions.md](03_design_decisions.md) — why things are the way they are
5. [04_performance.md](04_performance.md) — hardware, timing, optimization knobs
6. [05_flower_comparison.md](05_flower_comparison.md) — mentor's independent script
