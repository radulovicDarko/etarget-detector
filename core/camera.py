"""Camera backend selector.

Returns either a :class:`PiCamStream` or an :class:`FFmpegStream` based on
``CAMERA_BACKEND`` in :mod:`config` (which itself can be overridden by the
``SHOOTERRANGE_CAMERA_BACKEND`` env var).

In ``"auto"`` mode we probe for ``picamera2`` + an actual CSI/USB camera
visible to libcamera, and fall back to the RTSP/FFmpeg pipeline otherwise.
That means the same code path runs both:
  - on a Raspberry Pi 5 with Camera Module 3 attached → PiCamStream
  - on a developer Mac/Linux box → FFmpegStream reading the iPhone RTSP feed

Returning a duck-typed ``open()/read()/release()`` object keeps ``app.py``
agnostic — the rest of the pipeline (detector, scoring, control server)
just consumes ``frame`` ndarrays.
"""
from __future__ import annotations

import os
import sys
from typing import Any


def _maybe_add_system_dist_packages() -> None:
    """On Raspberry Pi OS, picamera2 is typically installed via apt into
    /usr/lib/python3/dist-packages which is NOT visible inside a venv by
    default. Add it opportunistically so CAMERA_BACKEND='auto' can detect
    the Pi camera even when running under systemd with a venv.
    """
    candidates = [
        "/usr/lib/python3/dist-packages",
        "/usr/local/lib/python3/dist-packages",
    ]
    for p in candidates:
        if p not in sys.path and os.path.isdir(p):
            sys.path.append(p)


def _picam_available() -> bool:
    """True if picamera2 is importable AND at least one camera is visible."""
    try:
        from picamera2 import Picamera2  # type: ignore
    except Exception:  # noqa: BLE001
        _maybe_add_system_dist_packages()
        try:
            from picamera2 import Picamera2  # type: ignore
        except Exception:  # noqa: BLE001
            return False
    try:
        cams = Picamera2.global_camera_info()  # type: ignore[attr-defined]
        return bool(cams)
    except Exception:  # noqa: BLE001
        return False


def create_camera_stream(config: Any) -> Any:
    """Build the right stream for the current host.

    ``config`` is the :mod:`config` module so this stays a single import for
    callers (matches how :mod:`app` already pulls its constants).
    """
    backend = (getattr(config, "CAMERA_BACKEND", "auto") or "auto").lower()

    if backend == "auto":
        backend = "picam" if _picam_available() else "rtsp"
        print(f"[camera] auto-detected backend: {backend}")

    if backend == "picam":
        from .picam_stream import PiCamStream
        return PiCamStream(
            width=getattr(config, "PICAM_WIDTH", 960),
            height=getattr(config, "PICAM_HEIGHT", 540),
            fps=getattr(config, "PICAM_FPS", 60),
        )

    if backend == "rtsp":
        from .ffmpeg_stream import FFmpegStream
        return FFmpegStream(
            ffmpeg_path=config.FFMPEG_PATH,
            rtsp_url=config.RTSP_URL,
            transport=config.RTSP_TRANSPORT,
            scale=config.JPEG_SCALE,
        )

    raise ValueError(
        f"Unknown CAMERA_BACKEND={backend!r} (expected 'auto', 'picam', or 'rtsp')"
    )
