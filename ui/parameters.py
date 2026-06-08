"""
ui/parameters.py

Typed dataclass for pipeline parameters exposed in the Process tab.
magicgui binds directly to this dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Sam2Model(str, Enum):
    tiny   = "tiny"
    small  = "small"
    base   = "base_plus"
    large  = "large"


@dataclass
class PipelineParams:
    stride:      int   = 4
    sam2_model:  Sam2Model = Sam2Model.tiny
    pre_window:  int   = 30
    inner_frac:  float = 0.75
    outer_frac:  float = 1.05
    prominence:  float = 0.05
    cotracker_stride: int = 8
    force_recompute:  bool = False
