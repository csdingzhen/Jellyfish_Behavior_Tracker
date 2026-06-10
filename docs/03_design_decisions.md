# Design Decisions and What Has Been Tried

This document records why the pipeline is built the way it is, including approaches that were considered and rejected. Understanding these decisions prevents future developers from re-proposing solutions that have already been ruled out.

---

## Core design principles

1. **Don't fight symmetries.** The bell is radially symmetric. Polar coordinates and dye-anchored angles exploit this; Cartesian representations fight it.
2. **Don't track identity from appearance.** Rhopalia look alike; their identity comes from their angular position relative to the dye fiducial, not from local visual features.
3. **Prefer physical quantities over geometric proxies.** Margin frame-difference (and divergence of optical flow) captures tissue motion directly. Silhouette area is a weak proxy that conflates contraction with tilt and camera angle.
4. **Segment first, derive everything from the mask.** The SAM2 mask provides the centroid, radius, and contour — all downstream stages use these rather than re-running their own segmentation.
5. **Dye mark = body-frame anchor.** The dye is the only source of rotational reference. Everything else (rhopalium IDs, initiation angles, cross-recording comparisons) depends on this anchor being tracked accurately.
6. **Per-pulse analyses are easier than persistent-identity analyses.** Each pulse can be analysed independently given the body-frame angle at the time of the pulse. Cross-recording rhopalium identity is an additional question, not a prerequisite.

---

## Fiducial strategy: why a dye mark

Cassiopea has no anatomical landmark equivalent to the gonad cross in Aurelia. Without an external reference, the body-frame orientation is undefined: you can detect pulses but cannot assign them to a specific rhopalium.

Options considered:

| Option | Why rejected |
| --- | --- |
| Gonad cross (Aurelia approach) | Cassiopea has no gonad cross |
| Chamber wall as reference | Provides a lab-frame reference but not a body-frame one; the animal can rotate within the chamber |
| Texture matching across frames | Feasible with optical flow but requires solving for rotation globally; dye is simpler and more robust |
| No reference at all | Can detect pulses and compute angles, but angles have no consistent meaning across frames |

The dye mark was chosen because it is the simplest and most robust: one artificial point tracked by CoTracker gives a per-frame body-frame angle with sub-degree precision.

### Wet-lab guidance for future improvements

If the dye is currently hard to see, priority order for improving contrast:

1. **Fluorescent dye + UV/violet illumination + emission filter on camera.** Single biggest win (10–100× contrast improvement). Cassiopea zooxanthellae autofluoresce red; a green/yellow fluorescent dye is spectrally separable.
2. **Larger mark** — one large spot beats several small ones.
3. **Mark placement off-centre but not at the bell edge** — edge marks have higher angular sensitivity but are more likely to be occluded by tentacles.
4. **Test CoTracker on existing footage first** before changing the wet-lab protocol — it can track low-contrast features using temporal context.

---

## Why CoTracker, not a classical tracker

Classical trackers (Lucas-Kanade, template matching, CSRT) fail on the faint dye because:
- The dye has low contrast relative to the background bell texture.
- Classical trackers have no temporal memory — a single occluded frame causes drift.

CoTracker uses a transformer with a 100+ frame context window. It has been shown to maintain tracks through complete occlusions. One manually clicked seed in frame 0 is all it needs.

Alternative: **TAPIR** (DeepMind, Apache 2.0). Equivalent capability to CoTracker but more permissive license. Use TAPIR if commercialisation becomes relevant.

---

## Why SAM2, not classical segmentation

Cassiopea's bell is a pale disc on dark water — high contrast at the boundary, but:
- The bell is translucent; thresholding on intensity picks up internal structure.
- During contraction the bell shape changes; an ellipse fit breaks.
- Tentacles extend beyond the bell and confuse area-based methods.

SAM2 handles all of these because it is a learned mask propagator with a memory bank — it tracks "this specific object through time" rather than "pixels above threshold." One click in frame 0 is all it needs.

---

## Why polar unwrapping for margin activity, not optical flow divergence

Both approaches detect contraction. The current pipeline uses polar unwrapping (frame difference along the margin) rather than full optical flow divergence because:

1. **Speed.** Frame difference on the polar strip is ~50× faster than computing dense optical flow and its divergence.
2. **Sufficiency.** Pulse detection accuracy is comparable; the margin strip already isolates where contraction events are most visible.
3. **Robustness.** Optical flow divergence requires accurate flow estimation; noisy flow (from low-texture regions) introduces false initiations.

RAFT optical flow (`scripts/run_raft.py`) is implemented but not used in the primary pipeline. It remains available for experiments.

---

## What was tried and rejected

### DeepLabCut (keypoint regression)

Proposed as a way to detect rhopalia positions directly.

**Problems:**
- Rhopalia are interchangeable — they have no semantic identity like "left eye." Per-keypoint identity assignment requires consistent labelling, which is undefined for radially symmetric structures.
- Standard CNN backbones have no rotation equivariance; the model has to relearn every rotation.
- Translucency and deformation make the local features around each rhopalium unstable.
- Label noise is high: annotators cannot consistently agree which rhopalium is "number 3."

**Why it's still not worth trying with rotation-equivariant architectures:** the underlying problem is identity assignment, not detection. Even if you can detect all 16 rhopalium positions, you still need to assign IDs — and that requires an external reference (the dye). The dye-anchored pipeline solves the assignment problem directly without per-keypoint detection.

### Classical CV (thresholding + ellipse fit + area)

Proposed as a cheap contraction signal.

**Problems:**
- Translucency: the bell has no hard edge; thresholding is brittle to lighting changes.
- Ellipse fit assumption breaks during contraction (bell deforms asymmetrically).
- Area/diameter conflates true contraction with tilt, depth changes, and tentacle motion.

### CBAS (jones-lab-tamu/CBAS)

Evaluated as a primary analysis tool.

**Finding:** CBAS = frozen DINOv2 backbone + LSTM head → outputs class labels (pulsing / resting / etc.), not coordinates or angles. Useful as a downstream classifier if discrete state labels are needed (optional Stage 7 in the architecture), but cannot replace the geometric pipeline.

---

## Current open questions

1. **Cross-recording identity.** Does "rhopalium A on day 1 = rhopalium A on day 5" matter scientifically? If yes, dye placement must be consistent across recordings. If no, per-recording angular labelling is sufficient.
2. **Out-of-plane tilt.** The pipeline assumes the bell is horizontal and the camera is orthogonal. Large tilts will distort apparent rhopalium angles. A sanity check on mask ellipse aspect ratio can flag tilted frames.
3. **Rhopalium visibility.** For recordings with high-quality optics, a polar-unwrap peak detector may be able to locate rhopalia automatically (removing the need for manual calibration per animal). This would require testing on high-resolution footage.
4. **Long recordings (> 10 min).** The pipeline currently processes segments and requires manual concatenation of output CSVs. A wrapper that automatically segments long recordings would be valuable.
