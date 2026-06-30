"""
packaging/make_ico.py

Generate assets/app_icon.ico (multi-size) from assets/app_icon.svg.
Windows shortcuts and .exe icons need .ico; we only keep the .svg in source.
Run standalone or via build_launcher.ps1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = Path(__file__).resolve().parent.parent
_SVG  = _ROOT / "assets" / "app_icon.svg"
_ICO  = _ROOT / "assets" / "app_icon.ico"
_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def main() -> int:
    from qtpy.QtWidgets import QApplication
    from qtpy.QtCore import QRectF, Qt
    from qtpy.QtGui import QImage, QPainter
    from qtpy.QtSvg import QSvgRenderer
    from PIL import Image

    if not _SVG.exists():
        print(f"SVG not found: {_SVG}")
        return 1

    _ = QApplication.instance() or QApplication(sys.argv)

    img = QImage(256, 256, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    QSvgRenderer(str(_SVG)).render(p, QRectF(0, 0, 256, 256))
    p.end()

    tmp_png = _ROOT / "assets" / "_icon_256.png"
    if not img.save(str(tmp_png), "PNG"):
        print("Failed to rasterize SVG.")
        return 1

    Image.open(tmp_png).convert("RGBA").save(_ICO, format="ICO", sizes=_SIZES)
    tmp_png.unlink(missing_ok=True)
    print(f"Wrote {_ICO} ({_ICO.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
