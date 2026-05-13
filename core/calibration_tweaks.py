"""Persistent calibration tweaks (loaded once at startup, saved on '+'/'-' etc.).

The bull detector finds the dark central disc reliably, but its diameter is
biased by morphological filtering and JPEG compression artefacts — typically
1-5% larger than the true 59.5 mm bull. Rather than try to perfectly tune
the detector for every camera/lighting combination, we expose three small
operator-adjustable corrections that get applied on top of the detected bull:

    scale_factor   — multiplies the bull's semi-axes before drawing/scoring
    offset_x_mm    — shifts the calibrated centre in paper-mm
    offset_y_mm    — shifts the calibrated centre in paper-mm

They persist to ``calibration_tweaks.json`` next to ``app.py`` and are
loaded on every startup. Defaults: 1.0, 0.0, 0.0.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


_DEFAULT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "calibration_tweaks.json",
)


@dataclass
class CalibrationTweaks:
    scale_factor: float = 1.0   # uniform multiplier for bull semi-axes
    offset_x_mm: float = 0.0    # paper-mm horizontal shift
    offset_y_mm: float = 0.0    # paper-mm vertical shift
    rotation_deg: float = 0.0   # extra rotation applied on top of bull angle
    aspect_ratio: float = 1.0   # extra Y / X size ratio (1.0 = circular)
    # Keystone correction simulates camera tilt. Each keystone value is the
    # fraction of one paper-half by which the OPPOSITE edges shrink/grow.
    #   keystone_h > 0  -> top edge wider than bottom (camera tilted DOWN)
    #   keystone_h < 0  -> top narrower (camera tilted UP)
    #   keystone_v > 0  -> right edge taller than left (camera tilted LEFT)
    #   keystone_v < 0  -> right shorter (camera tilted RIGHT)
    # Diagonal keystones move ONE corner along its diagonal direction so we
    # can correct asymmetric perspective (e.g. paper rotated AND tilted).
    #   keystone_d1 > 0  -> TL pulled inward, BR pushed outward (TL↔BR axis)
    #   keystone_d2 > 0  -> TR pulled inward, BL pushed outward (TR↔BL axis)
    keystone_h: float = 0.0
    keystone_v: float = 0.0
    keystone_d1: float = 0.0
    keystone_d2: float = 0.0
    # Independent rotation/scale for the BLUE PAPER OUTLINE only. Useful when
    # the printed rings are square-on but the paper itself is mounted
    # crooked: lets the operator align the visual rectangle without
    # touching the scoring geometry.
    paper_rotation_deg: float = 0.0
    paper_scale: float = 1.0

    def clamp(self) -> "CalibrationTweaks":
        return CalibrationTweaks(
            scale_factor=max(0.5, min(2.0, float(self.scale_factor))),
            offset_x_mm=max(-50.0, min(50.0, float(self.offset_x_mm))),
            offset_y_mm=max(-50.0, min(50.0, float(self.offset_y_mm))),
            rotation_deg=max(-45.0, min(45.0, float(self.rotation_deg))),
            aspect_ratio=max(0.5, min(2.0, float(self.aspect_ratio))),
            keystone_h=max(-0.5, min(0.5, float(self.keystone_h))),
            keystone_v=max(-0.5, min(0.5, float(self.keystone_v))),
            keystone_d1=max(-0.5, min(0.5, float(self.keystone_d1))),
            keystone_d2=max(-0.5, min(0.5, float(self.keystone_d2))),
            paper_rotation_deg=max(-45.0, min(45.0, float(self.paper_rotation_deg))),
            paper_scale=max(0.5, min(2.0, float(self.paper_scale))),
        )


def load_tweaks(path: str = _DEFAULT_FILE) -> CalibrationTweaks:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Drop unknown keys so old files don't blow up after a schema change.
        allowed = {"scale_factor", "offset_x_mm", "offset_y_mm",
                   "rotation_deg", "aspect_ratio",
                   "keystone_h", "keystone_v",
                   "keystone_d1", "keystone_d2",
                   "paper_rotation_deg", "paper_scale"}
        filtered = {k: v for k, v in raw.items() if k in allowed}
        return CalibrationTweaks(**filtered).clamp()
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return CalibrationTweaks()


def save_tweaks(tweaks: CalibrationTweaks, path: str = _DEFAULT_FILE) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(tweaks), f, indent=2)
    except OSError as e:
        print(f"[calibration_tweaks] failed to save: {e}")
