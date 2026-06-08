"""
scripts/run_ui.py

Launch the Cassiopea napari UI.

Usage
-----
  venv\Scripts\python scripts\run_ui.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.app import main

if __name__ == "__main__":
    main()
