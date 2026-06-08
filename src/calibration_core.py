"""
src/calibration_core.py

Shared calibration logic used by both the CLI (scripts/calibrate_rhopalia.py)
and the Napari UI (ui/calibration.py).

Exports
-------
phi_deg(p1, p2)                        → float   lab-frame angle p1→p2
body_angle(centroid, dye, point)       → float   body-frame angle of a point
build_calibration(centroid, dye, rhopalia) → dict
save_annotated_image(img_bgr, calib, out_path)
write_calibration_json(calib, json_path, img_path, img_bgr)

JSON format is identical to what calibrate_rhopalia.py previously produced.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_CENTROID = (  0,   0, 210)
C_DYE      = (  0, 220,  50)
C_RHOP     = (210,  80,   0)
C_AXIS     = (  0, 180,  60)
C_TEXT     = (230, 230, 230)
C_BELL     = (180, 180, 180)

N_RHOPALIA_EXPECTED = 16   # Cassiopea — shown in UI guide text, not enforced


# ── Geometry helpers ──────────────────────────────────────────────────────────

def phi_deg(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Lab-frame angle from p1 to p2, degrees, range (-180, 180]."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def body_angle(
    centroid: tuple[int, int],
    dye: tuple[int, int],
    point: tuple[int, int],
) -> float:
    """
    Body-frame angle of *point* relative to the dye axis.

    0° = toward dye mark.  Range: (-180, 180].
    Convention matches calibrate_rhopalia.py exactly.
    """
    phi_lab  = phi_deg(centroid, point)
    phi_dye  = phi_deg(centroid, dye)
    phi_body = phi_lab - phi_dye
    return (phi_body + 180) % 360 - 180


# ── Calibration builder ───────────────────────────────────────────────────────

def build_calibration(
    centroid: tuple[int, int],
    dye: tuple[int, int],
    rhopalia: list[tuple[int, int]],
) -> dict:
    """
    Compute body-frame angles for each rhopalium and build the calib dict.
    Output format is identical to calibrate_rhopalia.py.
    """
    phi_dye_lab = phi_deg(centroid, dye)
    rho_list = []
    for i, rp in enumerate(rhopalia):
        phi_lab  = phi_deg(centroid, rp)
        phi_body = body_angle(centroid, dye, rp)
        rho_list.append({
            "id":           i,
            "px":           list(rp),
            "phi_lab_deg":  round(phi_lab,  3),
            "phi_body_deg": round(phi_body, 3),
        })
    # Sort by body-frame angle for consistent ordering
    rho_list.sort(key=lambda r: r["phi_body_deg"])
    for new_id, r in enumerate(rho_list):
        r["id"] = new_id

    return {
        "centroid_px":     list(centroid),
        "dye_px":          list(dye),
        "phi_dye_lab_deg": round(phi_dye_lab, 3),
        "n_rhopalia":      len(rho_list),
        "rhopalia":        rho_list,
    }


# ── JSON writer ───────────────────────────────────────────────────────────────

def write_calibration_json(
    calib: dict,
    json_path: Path,
    img_path: Path | None = None,
    img_bgr: np.ndarray | None = None,
) -> None:
    """
    Write calib dict to json_path, adding source_image and image_size_px
    metadata when available.
    """
    out = dict(calib)
    if img_path is not None:
        out["source_image"] = str(img_path)
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        out["image_size_px"] = [w, h]

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)


# ── Annotated image writer ────────────────────────────────────────────────────

def _draw_arrow(img, p1, p2, color, thickness=2, tip=12):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    cv2.arrowedLine(img, p1, p2, color, thickness,
                    cv2.LINE_AA, tipLength=tip / max(dist, 1))


def save_annotated_image(
    img_bgr: np.ndarray,
    calib: dict,
    out_path: Path,
) -> None:
    """
    Write a publication-quality annotated diagram to out_path.
    Identical output to the version previously in calibrate_rhopalia.py.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    cx, cy = calib["centroid_px"]
    dx, dy = calib["dye_px"]

    r_bell = round(math.hypot(dx - cx, dy - cy) * 2.5)
    cv2.circle(out, (cx, cy), r_bell, C_BELL, 2, cv2.LINE_AA)

    _draw_arrow(out, (cx, cy), (dx, dy), C_AXIS, thickness=2, tip=20)

    cv2.circle(out, (cx, cy), 12, C_CENTROID, -1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), 13, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "C", (cx + 16, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_CENTROID, 2, cv2.LINE_AA)

    cv2.circle(out, (dx, dy), 12, C_DYE, -1, cv2.LINE_AA)
    cv2.circle(out, (dx, dy), 13, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "D  (phi=0)", (dx + 16, dy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_DYE, 2, cv2.LINE_AA)

    for r in calib["rhopalia"]:
        rx, ry   = r["px"]
        phi_body = r["phi_body_deg"]
        rid      = r["id"]
        cv2.circle(out, (rx, ry), 12, C_RHOP, -1, cv2.LINE_AA)
        cv2.circle(out, (rx, ry), 13, (255, 255, 255), 2, cv2.LINE_AA)
        label = f"R{rid}  {phi_body:+.1f}d"
        cv2.putText(out, label, (rx + 16, ry + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_RHOP, 2, cv2.LINE_AA)
        cv2.line(out, (cx, cy), (rx, ry), (*C_RHOP, 120), 1, cv2.LINE_AA)

    table_w = 260
    table_h = 30 + len(calib["rhopalia"]) * 24 + 10
    tx0 = w - table_w - 10
    ty0 = 10
    cv2.rectangle(out, (tx0, ty0), (tx0 + table_w, ty0 + table_h), (20, 20, 20), -1)
    cv2.rectangle(out, (tx0, ty0), (tx0 + table_w, ty0 + table_h), (80, 80, 80), 1)
    cv2.putText(out, "Rhopalium  phi_body", (tx0 + 8, ty0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv2.LINE_AA)
    for i, r in enumerate(calib["rhopalia"]):
        y = ty0 + 42 + i * 24
        cv2.putText(out, f"  R{r['id']}        {r['phi_body_deg']:+7.2f} deg",
                    (tx0 + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_RHOP, 1, cv2.LINE_AA)

    max_save = 2400
    sh, sw = out.shape[:2]
    if max(sh, sw) > max_save:
        sc = max_save / max(sh, sw)
        out = cv2.resize(out, (round(sw * sc), round(sh * sc)),
                         interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
