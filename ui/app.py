"""
ui/app.py

Entry point: create the napari viewer and attach the Cassiopea dock widget.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_ICON = Path(__file__).parent.parent / "assets" / "app_icon.svg"
_REPO_URL = "https://github.com/csdingzhen/Jellyfish_Behavior_Tracker"

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


def _setup_menu(viewer) -> None:
    """Replace napari's default menu bar with a minimal, app-relevant one.

    napari's stock menus (File/View/Layers/Plugins/Window/Help) expose
    layer, plugin, and preferences actions that don't map to this app and
    only create ambiguity. We clear them and add a small set of entries
    that are actually useful here.
    """
    try:
        from qtpy.QtWidgets import QAction, QMessageBox
        from qtpy.QtGui import QDesktopServices
        from qtpy.QtCore import QUrl

        win  = viewer.window._qt_window
        root = Path(__file__).parent.parent

        def _open(path: Path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

        def _open_url(url: str):
            QDesktopServices.openUrl(QUrl(url))

        mb = win.menuBar()
        mb.clear()

        # ── File ──────────────────────────────────────────────────────────
        file_menu = mb.addMenu("&File")

        act_outputs = QAction("Open Outputs Folder", win)
        def _open_outputs():
            from src.tasks import get_output_root
            d = get_output_root()
            d.mkdir(parents=True, exist_ok=True)
            _open(d)
        act_outputs.triggered.connect(_open_outputs)
        file_menu.addAction(act_outputs)

        act_projects = QAction("Open Projects Folder", win)
        act_projects.triggered.connect(lambda: _open(root / "project_folder"))
        file_menu.addAction(act_projects)

        file_menu.addSeparator()
        act_quit = QAction("Quit", win)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(win.close)
        file_menu.addAction(act_quit)

        # ── Help ──────────────────────────────────────────────────────────
        help_menu = mb.addMenu("&Help")

        act_guide = QAction("UI Guide", win)
        act_guide.triggered.connect(
            lambda: _open_url(f"{_REPO_URL}/blob/main/UI_GUIDE.md")
        )
        help_menu.addAction(act_guide)

        act_readme = QAction("README (GitHub)", win)
        act_readme.triggered.connect(lambda: _open_url(_REPO_URL))
        help_menu.addAction(act_readme)

        help_menu.addSeparator()
        act_about = QAction("About", win)
        act_about.triggered.connect(lambda: QMessageBox.about(
            win, "About Cassiopea Pipeline",
            "<b>Cassiopea Behavior Analysis Pipeline</b><br><br>"
            "Bell orientation, contraction timing, and pulse-initiation "
            "analysis for top-view <i>Cassiopea</i> recordings.<br><br>"
            "License: MIT<br>"
            f'<a href="{_REPO_URL}">{_REPO_URL}</a>'
        ))
        help_menu.addAction(act_about)
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
    _setup_menu(viewer)

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

    # ── Continuity annotation ──────────────────────────────────────────────────
    # The bell/dye clicks that seed each queued video can come from a project,
    # but batch processing must also work with NO project open (the common case:
    # sidebar Browse → Queue all, annotating only the first video by hand). So
    # we keep an app-level holder that survives without a project; the project's
    # shared_bell_click/dye_click are still updated when a project IS open.
    _continuity = {"bell": None, "dye": None}

    def _record_continuity(result):
        """After a successful run, capture the end-of-video bell/dye position
        as the starting guess for the next queued video. Always updates the
        app-level holder; also persists to the project if one is open."""
        bell_new, dye_new = extract_continuity_clicks(
            result.seg_csv, result.track_csv
        )
        if bell_new:
            _continuity["bell"] = bell_new
        if dye_new:
            _continuity["dye"] = dye_new
        state = widget.project_bar.project
        if state is not None:
            if bell_new:
                state.shared_bell_click = bell_new
            if dye_new:
                state.shared_dye_click = dye_new
            if state._path:
                state.save()

    def _resolve_annotation(path: Path):
        """Resolve (bell, dye, calib) for a queued video from, in priority
        order: per-video project clicks → project shared clicks → app-level
        continuity holder → whatever the user last annotated manually in the
        Process tab. Any of the three may be None if unresolved."""
        bell = dye = None
        calib = None
        state = widget.project_bar.project
        if state is not None:
            bell, dye = state.get_clicks(path)
            if bell is None:
                bell = state.shared_bell_click
            if dye is None:
                dye = state.shared_dye_click
            if state.calibration:
                calib = Path(state.calibration)
        # App-level continuity (covers the no-project batch case).
        if bell is None:
            bell = _continuity["bell"]
        if dye is None:
            dye = _continuity["dye"]
        # Last manual annotation still held by the Process tab — lets the
        # first hand-annotated video seed the rest of the batch even with no
        # project and before any run has produced continuity CSVs.
        pt = widget.process_tab
        if pt is not None:
            if bell is None and pt._bell_click is not None:
                bell = pt._bell_click
            if dye is None and pt._dye_click is not None:
                dye = pt._dye_click
            if (calib is None or not calib.exists()) and pt._calib_path is not None:
                calib = pt._calib_path
        return bell, dye, calib

    # Manual runs from the Process tab's own "Run pipeline" button bypass the
    # sidebar queue entirely (e.g. annotating the first video of a batch by
    # hand). Without this, the sidebar's status dot for that video would stay
    # stuck on whatever it showed before, and continuity would never advance —
    # so every subsequent queued video would hit the same "needs annotation".
    def _on_manual_pipeline_finished(path: Path, result):
        if result is not None and result.success:
            _record_continuity(result)   # seed continuity BEFORE advancing
            sidebar.mark_done(path)      # mark_done() advances the queue
        else:
            sidebar.mark_failed(path)

    widget.pipeline_finished.connect(_on_manual_pipeline_finished)

    # ── Auto-queue pipeline runner ────────────────────────────────────────────
    # When sidebar decides a video should start, this function is called.
    # It resolves bell/dye clicks (shared annotation + continuity) and
    # starts the pipeline worker.

    _active_worker = [None]   # mutable cell so inner closures can replace it

    def _start_queued_video(path: Path):
        # Resolve annotation from project → continuity holder → Process tab.
        # Works with or without a project open.
        bell_raw, dye_raw, calib_path = _resolve_annotation(path)

        if bell_raw is None or dye_raw is None:
            # No annotation available yet — pause (don't fail) and load the
            # video so the user can click bell+dye in the Process tab. The
            # same cause would hit every remaining queued video, so we
            # deliberately do NOT advance the queue here; once the user runs
            # this one manually, _record_continuity() seeds the rest and
            # mark_done() resumes it. Pausing avoids cascading one missing
            # annotation into "every video failed".
            sidebar.mark_needs_attention(path)
            widget.on_video_selected(path)
            return

        if calib_path is None or not calib_path.exists():
            # Missing calibration would recur identically for every video —
            # pause rather than cascade-fail, and load the video so the user
            # can pick a calibration in the Process tab.
            sidebar.mark_needs_attention(path)
            widget.on_video_selected(path)
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
                _record_continuity(result)   # seed continuity BEFORE advancing
                sidebar.mark_done(path)      # mark_done() advances the queue
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
