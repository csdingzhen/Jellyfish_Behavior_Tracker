"""
ui/thumbnails.py

First-frame thumbnail extraction and disk caching.
Thumbnails stored as small PNGs in <video_dir>/.thumbnails/.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

THUMB_SIZE = (120, 90)   # (width, height)


def thumb_path(video_path: Path) -> Path:
    cache_dir = video_path.parent / ".thumbnails"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{video_path.stem}.png"


def get_thumbnail(video_path: Path) -> np.ndarray | None:
    """
    Return a (H, W, 3) RGB thumbnail for *video_path*.
    Uses disk cache; extracts from video on first call.
    Returns None if the video cannot be read.
    """
    tp = thumb_path(video_path)
    if tp.exists():
        img = cv2.imread(str(tp))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None

    thumb = cv2.resize(frame, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(tp), thumb)
    return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)


def read_first_frame(video_path: Path) -> np.ndarray | None:
    """
    Return the full-resolution first frame as (H, W, 3) RGB, or None.
    """
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
