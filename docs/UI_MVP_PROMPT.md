# MVP UI build brief — Cassiopea pipeline Napari frontend

## Context

This task is building the first UI for a Cassiopea jellyfish behavior analysis pipeline. The Python pipeline is already complete and working. The job here is purely to wrap a usable interface around it.

Before writing any code, read:

1. `CLAUDE.md` — the project memory file with overall architecture and design principles.
2. `README.md` — the detailed pipeline spec.
3. `src/scheduler.py` — pay close attention to the existing UI-ready callbacks; the progress reporting goes through here.
4. `src/pipeline.py` and `src/tasks.py` — understand how `run_pipeline()` is composed.
5. `scripts/run_pipeline.py` — the CLI entry point for processing.
6. `scripts/calibrate_rhopalia.py` — the existing CLI calibration tool. The MVP needs to replace this with a Napari-native version, so read it carefully to understand the logic (clicks → angles → JSON output).

Do not restructure existing pipeline code. The UI sits on top of it. The one piece of logic that may need extraction is the angle-computation / JSON-writing core of `calibrate_rhopalia.py` — pull that into a helper module so both the CLI script and the new UI can call it. Don't change its output format.

## MVP scope — what to build

A single-file or small-package Napari application (not yet a published plugin — that comes later) supporting two workflows in the same dock widget, switched by tabs or radio buttons:

### Workflow A: Calibrate animal (one-time per animal)

1. **Image picker.** User selects a high-resolution still photo of the jellyfish with the dye mark visible. Photo loads into the Napari viewer as an Image layer.
2. **Animal name.** A text field for the animal identifier. The output JSON will be saved as `calibration/<name>.json`. Warn if a file with that name already exists.
3. **Sequential click workflow.** Three-stage Points-layer annotation, in this fixed order:
   - **Stage 1: bell centre** — single click, yellow point, labelled "C".
   - **Stage 2: dye mark** — single click, green point, labelled "D". This defines the 0° reference direction.
   - **Stage 3: rhopalia** — sequential clicks for each rhopalium (Cassiopea has 16). Red points, labelled with their index (R0, R1, R2, …) in click order around the bell. As each rhopalium is added, display its computed body-frame angle (angle relative to the dye direction, measured around the bell centre) in a small live-updating table in the sidebar.
4. **Undo support.** BACKSPACE or a "Remove last" button removes the most recently added point from the current stage. The UI does not advance to the next stage until the current stage is signalled complete (a "Next" button per stage, or auto-advance for the single-click stages).
5. **Save.** "Save calibration" button writes `calibration/<name>.json` in the existing format and saves `calibration/<name>_annotated.png` as a verification image (the photo with all labelled points overlaid — render this from the Napari viewer export, not by reinventing the drawing code).

### Workflow B: Process video

1. **Folder picker → video list.** User selects a folder. The UI lists all video files (mp4, avi, mov) in that folder. Show file names and a small first-frame thumbnail next to each.
2. **Video selection → first frame display.** Clicking a video loads its first frame into the Napari viewer as an Image layer.
3. **Calibration selection.** A dropdown lists all existing `calibration/*.json` files. User picks which calibration to use for this video. Warn if no calibrations exist and direct the user to Workflow A.
4. **Two-click annotation.** A "Mark bell" button puts the viewer in a mode where the next click adds a point to a "bell" Points layer (red). A "Mark dye" button does the same for a "dye" Points layer (green). User can re-click to update the position. Show the current coordinates in the sidebar.
5. **Parameter panel.** Use `magicgui` to generate a form from a typed parameter dataclass. At minimum expose: `stride` (int, default 4), `sam2_model` (str enum: tiny/small/base/large, default tiny), `pre_window` (int, default 30), `inner_frac` (float, default 0.75), `outer_frac` (float, default 1.05), `prominence` (float, default 0.05).
6. **Run button.** Calls the existing `run_pipeline()` in a background thread using `napari.qt.threading.thread_worker`. Disable the button while running. Pass the bell click, dye click, video path, selected calibration path, and parameters into the pipeline.
7. **Progress display.** Hook into the scheduler's existing callbacks to display per-stage progress bars (SAM2, CoTracker, Approach B). A simple log text area showing scheduler events is also fine.
8. **Results display.** When the pipeline finishes, automatically:
   - Load `<stem>_annotated.mp4` (or the underlying frames) as a Napari Image layer at the original time axis.
   - Load the dye track CSV as a Tracks layer.
   - Load the bell centre CSV as a Points layer.
   - Load `<stem>_initiation_b.csv` and display it as a small table in the sidebar.
   - Show `<stem>_initiation_b_plot.png` in a side panel (just embed as an image, no need for interactive plotting in MVP).

## Explicit non-goals — do not build these yet

- Batch processing of multiple videos in sequence
- Editing existing calibrations (delete + redo is acceptable in MVP)
- Polished results dashboard with interactive plots (the static PNG is enough for MVP)
- Cancellation of in-flight GPU work (a "stop after current stage" toggle is acceptable but optional)
- Plugin packaging, `pyproject.toml` for distribution, PyPI publication
- Configuration presets, save/load parameter sets

These can come later. The MVP is "one biologist can calibrate one animal and process one video end-to-end through the UI."

## Technical guidance

- **Threading**: always use `napari.qt.threading.thread_worker` for the pipeline call. Never touch Napari layers from the worker thread — yield results back to the main thread and update layers in `yielded` callbacks.
- **magicgui**: import as `from magicgui import magicgui` and use the `@magicgui` decorator on a function that returns a dict of parameters, or use `magicgui.widgets.Container` for fuller control.
- **Layers convention**:
  - Processing: bell mask → Labels layer; bell centre / dye click → Points layers with distinct colours; dye trajectory → Tracks layer.
  - Calibration: centre (yellow), dye (green), rhopalia (red with index labels) — each a separate Points layer for easy toggling.
- **Click-mode UX in Napari**: prefer adding a Points layer with `mode='add'` so the user just clicks to place points, rather than building a custom event handler. To enforce click order (centre → dye → rhopalia), only enable one Points layer's add-mode at a time and disable the others.
- **Angle computation**: extract the existing logic from `calibrate_rhopalia.py` into a helper (e.g. `src/calibration_core.py`) that both the CLI and the UI call. Same input/output, same JSON format.
- **File paths**: read `VIDEO_DIR` from `config.py` as the default folder for the browser, but let the user override.
- **Thumbnails**: extract first frame with `cv2.VideoCapture` (already a dependency); cache thumbnails as small PNGs in a hidden `.thumbnails/` folder inside the video directory so repeated browsing is fast.
- **Scheduler integration**: read `src/scheduler.py` first and find where the callbacks are emitted. The UI subscribes to those callbacks via a thread-safe queue. Do not modify the scheduler — just consume its events.

## Suggested file layout

```
ui/
  __init__.py
  app.py             # main entry point, instantiates napari and the dock widget
  widget.py          # the dock widget with the two workflow tabs
  calibration.py     # Workflow A: calibration tab
  processing.py      # Workflow B: video processing tab
  parameters.py      # dataclass + magicgui binding for pipeline parameters
  workers.py         # thread_worker wrapping run_pipeline + progress callback adapter
  thumbnails.py      # first-frame extraction and caching

src/
  calibration_core.py  # NEW: extracted angle-computation + JSON writer (shared by CLI and UI)
```

Add an entry point script at `scripts/run_ui.py` that just launches `ui.app.main()`.

## Development order — please follow this sequence

Build incrementally and verify each step works before moving on. Do not jump ahead.

1. **Skeleton**: get a Napari window open with an empty dock widget showing two tabs (Calibrate / Process) and placeholder labels in each. Confirm it launches.
2. **Extract `calibration_core.py`**: move the angle-computation and JSON-writing logic out of `scripts/calibrate_rhopalia.py` into a shared module. Verify the existing CLI script still works after the refactor.
3. **Calibration tab — image loading and centre/dye clicks**: image picker, load into viewer, two-stage click for centre then dye. Display coordinates in the sidebar.
4. **Calibration tab — rhopalia clicks and live angle table**: add rhopalia stage with iterative clicking, live-updating angle table.
5. **Calibration tab — save**: write JSON via `calibration_core.py` and save the annotated PNG. Verify the JSON is identical in format to one produced by the CLI script.
6. **Process tab — folder picker + video list (no thumbnails yet)**: just file names. Selecting one loads first frame into the viewer.
7. **Add thumbnails** to the video list once selection-and-load works.
8. **Process tab — calibration dropdown and two-click annotation**.
9. **Parameter panel** via magicgui. Verify values flow correctly into a parameters dataclass.
10. **Run button → background pipeline call**, no progress bar yet. Just block the UI with a spinner and confirm the pipeline runs to completion with the UI-supplied inputs.
11. **Progress bars and log** hooked into the scheduler callbacks.
12. **Results loading and display** after pipeline completes.

After each step, stop, run it, and report what you've verified before proceeding. Do not chain steps into one large implementation pass.

## Things to ask the user about

If you encounter ambiguity in any of the following, stop and ask before guessing:

- Whether the bell-click in Workflow B should immediately preview a SAM2 mask (nice to have, but adds latency) or just store the coordinates for later use by the pipeline.
- Whether the calibration UI should enforce a fixed count of 16 rhopalia or allow any number (the existing CLI script's behaviour should be the source of truth — check it).
- Whether the UI should support the `--recompute` flag for Approach B, or always use cached margin_diff if available.
- Whether OS is Windows-only (per the README's PowerShell instructions) or cross-platform is needed.
- The exact angular convention used in `calibration_core.py` (counter-clockwise from dye? clockwise? zero direction?) — match it exactly; do not assume.

## Final note

The pipeline already works end-to-end from the CLI. If something seems harder to build than expected, it's almost always because the UI is trying to replicate logic that already exists in `src/` or `scripts/`. Read first, then wrap — don't reimplement.
