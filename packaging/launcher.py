"""
packaging/launcher.py

Source for "Cassiopea Pipeline.exe" — a tiny launcher compiled with
PyInstaller (--onefile --windowed). It simply starts the UI via the project's
own ``pythonw.exe`` so the executable carries our icon and shows no console.

It does NOT bundle Python/torch/napari — the .exe must sit in the project
root, next to the ``venv\\`` and ``scripts\\`` folders (which setup.ps1 creates).
setup.ps1 builds it automatically; rebuild it any time with
``packaging/build_launcher.ps1``.

To debug a startup failure (the .exe shows no console), run the UI directly:
``venv\\Scripts\\python scripts\\run_ui.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    """Project root. When frozen the .exe lives there; in source, go up one."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def _error(msg: str) -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, msg, "Cassiopea Pipeline", 0x30)
            return
        except Exception:
            pass
    print(msg)


def main() -> int:
    root   = _root()
    pyw    = root / "venv" / "Scripts" / "pythonw.exe"
    py     = root / "venv" / "Scripts" / "python.exe"
    script = root / "scripts" / "run_ui.py"

    if not script.exists():
        _error(f"Could not find scripts\\run_ui.py next to the executable:\n{root}")
        return 1

    exe = pyw if pyw.exists() else (py if py.exists() else None)
    if exe is None:
        _error("Could not find the virtual environment (venv\\).\n"
               "Run setup.ps1 first to create it.")
        return 1

    # Detached: the launcher exits immediately and the UI owns its own window.
    subprocess.Popen([str(exe), str(script)], cwd=str(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
