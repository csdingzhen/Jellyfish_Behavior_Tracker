# Performance Reference

## Hardware baseline

All timings below are measured on the development machine:

- GPU: NVIDIA RTX 4060 (8 GB VRAM)
- CPU: (unspecified)
- RAM: 32 GB
- OS: Windows 11
- PyTorch with CUDA 12.4

---

## Recommended configuration (accuracy-first)

| Parameter | Value | Rationale |
| --- | --- | --- |
| SAM2 stride | 1 | Every frame; eliminates interpolation artifacts in contour radii |
| CoTracker stride | 4 | Every 4th frame (30 fps); sufficient for dye angle accuracy |
| SAM2 model | tiny | Fastest; no measurable mask quality difference for Cassiopea's high-contrast bell |
| Image size | 512 px (ViT internal) | Balances speed and feature resolution |

This configuration was validated by the lab mentor as producing the most accurate initiation site assignments.

---

## Processing time (10-minute clip at 120 fps)

### Recommended config (SAM2 stride=1, CT stride=4)

| Stage | Time |
| --- | --- |
| SAM2 segmentation | ~120 min |
| CoTracker tracking | ~28 min |
| Approach B (margin diff + initiation) | ~5 min |
| **Total (parallel)** | **~2.5 hrs** |

### Fast exploratory config (SAM2 stride=4, CT stride=4)

| Stage | Time |
| --- | --- |
| SAM2 segmentation | ~30 min |
| CoTracker tracking | ~28 min |
| Approach B | ~5 min |
| **Total (parallel)** | **~60 min** |

### Legacy config (SAM2 stride=4, CT stride=8) — not recommended

| Stage | Time |
| --- | --- |
| SAM2 segmentation | ~30 min |
| CoTracker tracking | ~110 min |
| Approach B | ~5 min |
| **Total (parallel)** | **~2.5 hrs** |

Note: stride=8 CoTracker is slower than stride=4 because stride=8 creates more chunks at a larger effective context length. Always prefer stride=4 over stride=8.

---

## SAM2 model comparison (RTX 4060)

| Model | Weights | Speed at stride=1 | Speed at stride=4 |
| --- | --- | --- | --- |
| tiny | 148 MB | ~3.5 fps | ~8.5 fps |
| small | 185 MB | ~2.5 fps | ~6 fps |
| base_plus | 308 MB | ~2.2 fps | ~5.6 fps |
| large | ~900 MB | ~0.8 fps | ~2–3 fps |

`tiny` is recommended for all standard recordings. Switch to `base_plus` only if you observe mask drift on difficult footage (e.g., heavy tentacle motion or camera shake).

---

## Active optimisations in the codebase

### bfloat16 inference (CoTracker)
CoTracker runs in mixed precision (`torch.bfloat16`). On Ampere+ GPUs this uses tensor cores and gives ~1.5–2× speedup with no accuracy loss for tracking.

### torch.compile
Both the SAM2 image encoder and the CoTracker model are compiled with `torch.compile` on first run. This triggers kernel fusion and layout optimisation. The first chunk is ~30 s slower; all subsequent chunks run ~15–25% faster. The compiled model is cached for the session.

### Parallel task scheduler
`src/scheduler.py` runs SAM2, CoTracker, and Approach B Phase 1 concurrently subject to GPU gate constraints. On an 8 GB GPU: SAM2 ~3 GB + CoTracker ~3 GB = ~6 GB, leaving ~2 GB headroom. Approach B Phase 1 is CPU-bound and can overlap with both.

### Streaming video decode
Approach B reads directly from the source MP4 using OpenCV's `VideoCapture`, then seeks to each required frame. No JPEG extraction is needed; this saves ~5+ GB disk I/O per 10-minute recording at 120 fps.

### Automatic frame cleanup
SAM2 requires JPEG frames on disk (it uses its own file loader). These are written to `<stem>_frames/` and deleted automatically after segmentation completes to recover disk space (~5–8 GB per 10-minute clip at 120 fps).

### NVENC video encoding
Annotated output videos are encoded with NVENC (NVIDIA hardware H.264 encoder) when available. `_probe_nvenc()` does a functional test (encodes a 16×16 dummy frame) before committing to NVENC — this prevents broken-pipe crashes if the GPU driver exposes NVENC in the encoder list but the hardware does not actually support it (common on some laptop configurations).

---

## Future optimisation opportunities

1. **CoTracker chunk size.** The chunk size parameter controls how many frames are processed in one forward pass. Larger chunks use more VRAM but reduce overhead. Current default is not tuned; a systematic sweep could find a faster setting.
2. **SAM2 window overlap.** SAM2 processes overlapping windows of ~200 frames. The overlap percentage could be reduced without loss of quality (most overlap is redundant for a slow-moving, sessile animal).
3. **ONNX export.** SAM2 and CoTracker could be exported to ONNX for deployment on machines without a PyTorch installation.
4. **Batched Approach B.** The margin diff computation is currently frame-sequential; it could be vectorised into batched GPU operations.
