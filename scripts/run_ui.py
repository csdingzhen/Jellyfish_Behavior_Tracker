"""
scripts/run_ui.py

Launch the Cassiopea napari UI.

Usage
-----
  venv\Scripts\python scripts\run_ui.py
"""

import os
import sys
from pathlib import Path


def _ensure_standard_streams() -> None:
    """Give console-oriented dependencies writable streams under pythonw.

    Windows GUI launches use ``pythonw.exe``, where ``sys.stdout`` and
    ``sys.stderr`` may be ``None``. SAM2 uses tqdm internally while loading
    video frames, and tqdm expects stderr to provide ``write`` and ``flush``.
    Route missing streams to the null device so progress output is discarded
    instead of crashing the pipeline.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


_ensure_standard_streams()

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.app import main

if __name__ == "__main__":
    main()
