"""
scripts/calibrate_rhopalia.py

One-time calibration tool: identify rhopalium body-frame angles from a
high-resolution still image of the jellyfish with dye mark visible.

Because rhopalia are fixed to the bell, their body-frame angles are
CONSTANTS for the animal.  This replaces Stage 3 (automated polar-unwrap
detection) entirely.

Three-step click workflow
--------------------------
  Step 1 — Centroid
      Click the bell centre of mass.
      An auto-detected suggestion is shown; click anywhere to override.
      ENTER / SPACE to confirm.

  Step 2 — Dye mark
      Click the dye mark on the bell surface.
      This becomes the phi = 0 deg body-frame reference.
      ENTER / SPACE to confirm.

  Step 3 — Rhopalia
      Click each visible rhopalium in any order.
      Points are numbered as you place them.
      BACKSPACE or right-click to undo the last point.
      ENTER / SPACE when done.

Keys (all steps)
-----------------
  Left-click    place / replace point
  ENTER / SPACE confirm current step and advance
  BACKSPACE     (step 3) undo last rhopalium
  Right-click   (step 3) undo last rhopalium
  ESC           quit without saving

Outputs  (written to calibration/)
------------------------------------
  <stem>.json   body-frame angles + pixel positions
  <stem>.png    annotated diagram for verification / publication
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration_core import (
    build_calibration,
    save_annotated_image,
    write_calibration_json,
    phi_deg as _phi_deg,
    C_CENTROID, C_DYE, C_RHOP, C_AXIS, C_TEXT, C_BELL,
    N_RHOPALIA_EXPECTED,
)

CALIB_DIR   = Path(__file__).parent.parent / "calibration"
MAX_DISPLAY = 1400

N_RHOPALIA = N_RHOPALIA_EXPECTED   # used in guide text only


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scale_for_display(img: np.ndarray, max_dim: int):
    h, w  = img.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        disp = cv2.resize(img, (round(w * scale), round(h * scale)),
                          interpolation=cv2.INTER_AREA)
    else:
        disp  = img.copy()
        scale = 1.0
    return disp, scale


def _to_img(px: tuple, scale: float) -> tuple[int, int]:
    return (round(px[0] / scale), round(px[1] / scale))


def _to_disp(px: tuple, scale: float) -> tuple[int, int]:
    return (round(px[0] * scale), round(px[1] * scale))


def _auto_centroid(img_bgr: np.ndarray) -> tuple[int, int] | None:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 4)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return (round(M["m10"] / M["m00"]), round(M["m01"] / M["m00"]))


def _draw_arrow(img, p1, p2, color, thickness=2, tip=12):
    cv2.arrowedLine(img, p1, p2, color, thickness,
                    cv2.LINE_AA, tipLength=tip / max(math.hypot(
                        p2[0]-p1[0], p2[1]-p1[1]), 1))


# ── Step renderers ────────────────────────────────────────────────────────────

def _render(base: np.ndarray, scale: float,
            centroid, dye, rhopalia, step: int) -> np.ndarray:
    out = base.copy()
    h, w = out.shape[:2]

    guides = [
        "Step 1/3  CENTROID  —  click bell centre  |  ENTER confirm",
        "Step 2/3  DYE MARK  —  click dye mark  |  ENTER confirm",
        f"Step 3/3  RHOPALIA  —  click each rhopalium ({N_RHOPALIA} expected)"
        f"  |  BACKSPACE undo  |  ENTER done",
    ]
    cv2.rectangle(out, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(out, guides[step], (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1, cv2.LINE_AA)

    if centroid and dye:
        cc = _to_disp(centroid, scale)
        dd = _to_disp(dye, scale)
        r_est = round(math.hypot(dd[0]-cc[0], dd[1]-cc[1]) * 2.5)
        cv2.circle(out, cc, r_est, C_BELL, 1, cv2.LINE_AA)
        _draw_arrow(out, cc, dd, C_AXIS, thickness=1, tip=15)

    if centroid:
        cc = _to_disp(centroid, scale)
        cv2.circle(out, cc, 9, C_CENTROID, -1, cv2.LINE_AA)
        cv2.circle(out, cc, 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "C", (cc[0]+12, cc[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CENTROID, 2, cv2.LINE_AA)

    if dye:
        dd = _to_disp(dye, scale)
        cv2.circle(out, dd, 9, C_DYE, -1, cv2.LINE_AA)
        cv2.circle(out, dd, 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "D  phi=0", (dd[0]+12, dd[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_DYE, 1, cv2.LINE_AA)

    for i, rp in enumerate(rhopalia):
        rr = _to_disp(rp, scale)
        phi_body = 0.0
        if centroid and dye:
            phi_lab  = _phi_deg(centroid, rp)
            phi_dye  = _phi_deg(centroid, dye)
            phi_body = phi_lab - phi_dye
            phi_body = (phi_body + 180) % 360 - 180
        cv2.circle(out, rr, 9, C_RHOP, -1, cv2.LINE_AA)
        cv2.circle(out, rr, 10, (255, 255, 255), 1, cv2.LINE_AA)
        label = f"R{i}  {phi_body:+.1f}d"
        cv2.putText(out, label, (rr[0]+12, rr[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_RHOP, 1, cv2.LINE_AA)

    if step == 2:
        badge = f"{len(rhopalia)} placed"
        cv2.putText(out, badge, (w - 140, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_RHOP, 1, cv2.LINE_AA)

    return out


# ── Click UI ──────────────────────────────────────────────────────────────────

class CalibrationUI:
    WIN = "Rhopalium calibration  —  ESC to quit"

    def __init__(self, img_path: Path):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            sys.exit(f"Cannot load image: {img_path}")
        self.img_path = img_path
        self.orig     = img_bgr
        self.disp, self.scale = _scale_for_display(img_bgr, MAX_DISPLAY)
        self._suggest = _auto_centroid(img_bgr)

        self.step     = 0
        self.centroid = self._suggest
        self.dye      = None
        self.rhopalia = []
        self._click   = None

    def _on_mouse(self, event, x, y, flags, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click = (x, y)
        if event == cv2.EVENT_RBUTTONDOWN and self.step == 2:
            if self.rhopalia:
                self.rhopalia.pop()

    def run(self):
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WIN, self._on_mouse)

        while True:
            frame = _render(self.disp, self.scale,
                            self.centroid, self.dye, self.rhopalia, self.step)

            if self.step == 0 and self._suggest:
                C_SUGGEST = (120, 120, 120)
                sd = _to_disp(self._suggest, self.scale)
                cv2.drawMarker(frame, sd, C_SUGGEST, cv2.MARKER_CROSS, 20, 1)
                cv2.putText(frame, "auto-detected (click to override)",
                            (sd[0]+14, sd[1]+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_SUGGEST, 1, cv2.LINE_AA)

            cv2.imshow(self.WIN, frame)
            key = cv2.waitKey(20) & 0xFF

            if self._click is not None:
                pt_img = _to_img(self._click, self.scale)
                if self.step == 0:
                    self.centroid = pt_img
                elif self.step == 1:
                    self.dye = pt_img
                elif self.step == 2:
                    self.rhopalia.append(pt_img)
                self._click = None

            if key in (13, 32):
                if self.step == 0 and self.centroid:
                    self.step = 1
                elif self.step == 1 and self.dye:
                    self.step = 2
                elif self.step == 2:
                    if len(self.rhopalia) == 0:
                        print("[warn] No rhopalia clicked — add at least one.")
                    else:
                        break
            elif key == 8 and self.step == 2:
                if self.rhopalia:
                    self.rhopalia.pop()
            elif key == 27:
                cv2.destroyAllWindows()
                print("Cancelled.")
                sys.exit(0)

        cv2.destroyAllWindows()
        return self.centroid, self.dye, self.rhopalia


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate rhopalium body-frame angles from a hi-res still image")
    ap.add_argument("image", help="Path to the high-resolution jellyfish image")
    ap.add_argument("--out", default=None,
                    help="Output stem name (default: image filename stem)")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"Image not found: {img_path}")

    stem     = args.out or img_path.stem
    CALIB_DIR.mkdir(exist_ok=True)
    json_out = CALIB_DIR / f"{stem}.json"
    png_out  = CALIB_DIR / f"{stem}_annotated.png"

    print(f"Image  : {img_path}")
    print(f"Output : {json_out}")
    print()
    print("Controls:")
    print("  Step 1 — click centroid  (ENTER to confirm)")
    print("  Step 2 — click dye mark  (ENTER to confirm)")
    print(f"  Step 3 — click each rhopalium  (BACKSPACE to undo, ENTER when done)")
    print()

    ui = CalibrationUI(img_path)
    centroid, dye, rhopalia = ui.run()

    calib = build_calibration(centroid, dye, rhopalia)
    img_bgr = cv2.imread(str(img_path))

    write_calibration_json(calib, json_out, img_path=img_path, img_bgr=img_bgr)
    print(f"Calibration JSON saved: {json_out}")

    save_annotated_image(img_bgr, calib, png_out)

    print(f"\nCalibration summary  ({calib['n_rhopalia']} rhopalia):")
    print(f"  Centroid : {centroid}")
    print(f"  Dye mark : {dye}  (phi_dye = {calib['phi_dye_lab_deg']:.1f} deg lab)")
    print()
    print(f"  {'ID':<4}  {'phi_body (deg)':>16}  {'px'}")
    for r in calib["rhopalia"]:
        print(f"  R{r['id']:<3}  {r['phi_body_deg']:>+14.2f}  {r['px']}")


if __name__ == "__main__":
    main()
