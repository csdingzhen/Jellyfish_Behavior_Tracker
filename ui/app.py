"""
ui/app.py

Entry point: create the napari viewer and attach the Cassiopea dock widget.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_ICON = Path(__file__).parent.parent / "assets" / "app_icon.svg"

# Windows: set the App User Model ID before QApplication is created so the
# taskbar groups this process under our own icon rather than Python's.
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "Jellyfish.Cassiopea.1"
    )


def _simplify_viewer(viewer) -> None:
    """Hide the photoshop-like layer controls and layer list panels."""
    try:
        from qtpy.QtWidgets import QDockWidget
        hide_names = {"layer controls", "layer list"}
        for dock in viewer.window._qt_window.findChildren(QDockWidget):
            if dock.objectName() in hide_names:
                dock.hide()
    except Exception:
        pass  # graceful fallback if napari internals change


def _make_icon(svg_path: Path):
    """Rasterize SVG → multi-resolution QIcon (needed for Windows taskbar)."""
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QIcon, QPixmap, QPainter
    from qtpy.QtSvg import QSvgRenderer
    renderer = QSvgRenderer(str(svg_path))
    icon = QIcon()
    for size in (16, 32, 48, 64, 128, 256):
        px = QPixmap(size, size)
        px.fill(Qt.transparent)
        p = QPainter(px)
        renderer.render(p)
        p.end()
        icon.addPixmap(px)
    return icon


def main():
    import threading
    import napari
    from qtpy.QtWidgets import QApplication
    from .widget import CassiopeaWidget
    from .sidebar import VideoSidebarWidget
    from .project import extract_continuity_clicks

    viewer = napari.Viewer(title="Cassiopea Pipeline")
    _simplify_viewer(viewer)

    if _ICON.exists():
        icon = _make_icon(_ICON)
        QApplication.instance().setWindowIcon(icon)
        viewer.window._qt_window.setWindowIcon(icon)

    # Right dock — main workflow tabs + hardware panel
    widget = CassiopeaWidget(viewer)
    viewer.window.add_dock_widget(
        widget,
        name="Cassiopea",
        area="right",
        allowed_areas=["right", "left"],
    )

    # Left dock — video browser / queue
    sidebar = VideoSidebarWidget()
    viewer.window.add_dock_widget(
        sidebar,
        name="Videos",
        area="left",
        allowed_areas=["left", "right"],
    )

    # ── Signal wiring ─────────────────────────────────────────────────────────

    # Clicking a video in sidebar → load into viewer
    sidebar.video_selected.connect(widget.on_video_selected)

    # Project opened/created → load video folder into sidebar + notify process tab
    def _on_project_changed(state):
        if state.video_folder:
            sidebar.load_folder(Path(state.video_folder))

    widget.project_bar.project_changed.connect(_on_project_changed)

    # Hardware widget auto-queue toggle → sidebar
    widget.hw_widget.auto_queue_changed.connect(sidebar.set_auto_queue)

    # ── Auto-queue pipeline runner ────────────────────────────────────────────
    # When sidebar decides a video should start, this function is called.
    # It resolves bell/dye clicks (shared annotation + continuity) and
    # starts the pipeline worker.

    _active_worker = [None]   # mutable cell so inner closures can replace it

    def _start_queued_video(path: Path):
        state = widget.project_bar.project
        if state is None:
            sidebar.mark_failed(path)
            return

        # Resolve annotation: per-video overrides shared, shared is updated
        # after each completed run for continuity.
        bell_raw, dye_raw = state.get_clicks(path)
        if bell_raw is None:
            bell_raw = state.shared_bell_click
        if dye_raw is None:
            dye_raw = state.shared_dye_click

        if bell_raw is None or dye_raw is None:
            # No annotation available — fall back to manual (load video for user)
            sidebar.mark_failed(path)
            widget.on_video_selected(path)
            return

        calib_path = Path(state.calibration) if state.calibration else None
        if calib_path is None or not calib_path.exists():
            sidebar.mark_failed(path)
            return

        bell_click = (int(bell_raw[0]), int(bell_raw[1]))
        dye_click  = (int(dye_raw[0]),  int(dye_raw[1]))

        # Build params from process tab (if built) or defaults
        from .parameters import PipelineParams
        params = (widget.process_tab._params
                  if widget.process_tab is not None
                  else PipelineParams())

        sidebar.mark_processing(path)

        from .workers import run_pipeline_worker
        cancel_ev = threading.Event()
        worker = run_pipeline_worker(
            video_path   = path,
            bell_click   = bell_click,
            dye_click    = dye_click,
            calib_path   = calib_path,
            params       = params,
            cancel_event = cancel_ev,
        )
        _active_worker[0] = worker

        def _on_done(result):
            if result.success:
                sidebar.mark_done(path)
                # Continuity: update shared annotation from last-frame outputs
                bell_new, dye_new = extract_continuity_clicks(
                    result.seg_csv, result.track_csv
                )
                if bell_new:
                    state.shared_bell_click = bell_new
                if dye_new:
                    state.shared_dye_click = dye_new
                if state._path:
                    state.save()
            else:
                sidebar.mark_failed(path)

        def _on_error(_):
            sidebar.mark_failed(path)

        worker.returned.connect(_on_done)
        worker.errored.connect(_on_error)

        # Forward progress to sidebar per-video bar and to Process tab (if open)
        _path = path   # capture for lambda
        worker.yielded.connect(
            lambda ev: sidebar.update_video_progress(
                _path, ev.overall_fraction, ev.task_name
            )
        )
        if widget.process_tab is not None:
            worker.yielded.connect(widget.process_tab._on_progress_event)

        worker.start()

    sidebar.queue_start.connect(_start_queued_video)

    napari.run()


if __name__ == "__main__":
    main()
