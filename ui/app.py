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
    import napari
    from qtpy.QtWidgets import QApplication
    from .widget import CassiopeaWidget
    from .sidebar import VideoSidebarWidget

    viewer = napari.Viewer(title="Cassiopea Pipeline")
    _simplify_viewer(viewer)

    if _ICON.exists():
        icon = _make_icon(_ICON)
        QApplication.instance().setWindowIcon(icon)
        viewer.window._qt_window.setWindowIcon(icon)

    # Right dock — main workflow tabs
    widget = CassiopeaWidget(viewer)
    viewer.window.add_dock_widget(
        widget,
        name="Cassiopea",
        area="right",
        allowed_areas=["right", "left"],
    )

    # Left dock — video browser sidebar
    sidebar = VideoSidebarWidget()
    viewer.window.add_dock_widget(
        sidebar,
        name="Videos",
        area="left",
        allowed_areas=["left", "right"],
    )

    # Connect sidebar → process tab
    sidebar.video_selected.connect(widget.on_video_selected)

    # Connect project bar → sidebar (auto-load folder when project opens/creates)
    def _on_project_for_sidebar(state):
        if state.video_folder:
            from pathlib import Path
            sidebar.load_folder(Path(state.video_folder))

    widget.project_bar.project_changed.connect(_on_project_for_sidebar)

    napari.run()


if __name__ == "__main__":
    main()
