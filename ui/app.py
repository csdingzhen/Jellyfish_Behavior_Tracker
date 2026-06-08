"""
ui/app.py

Entry point: create the napari viewer and attach the Cassiopea dock widget.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


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


def main():
    import napari
    from .widget import CassiopeaWidget

    viewer = napari.Viewer(title="Cassiopea Pipeline")
    _simplify_viewer(viewer)

    widget = CassiopeaWidget(viewer)
    viewer.window.add_dock_widget(
        widget,
        name="Cassiopea",
        area="right",
        allowed_areas=["right", "left"],
    )
    napari.run()


if __name__ == "__main__":
    main()
