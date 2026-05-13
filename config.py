RTSP_URL = "rtsp://192.168.1.191:8554/stream"

WINDOW_NAME = "iPhone FFmpeg Low-Latency Stream"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540
CHANNELS = 3

DISPLAY_MAX_WIDTH = 1000
DISPLAY_MAX_HEIGHT = 700

SHOW_FPS = True

# ---------------------------------------------------------------------------
# Camera backend selection.
#
#   "auto" (default): try picamera2 first (Raspberry Pi + Camera Module),
#                     fall back to the FFmpeg/RTSP pipeline (iPhone NDI on Mac).
#   "picam":          force libcamera/picamera2. Fails fast if not available.
#   "rtsp":           force the FFmpeg/RTSP pipeline.
#
# Override at runtime with the SHOOTERRANGE_CAMERA_BACKEND env var.
# ---------------------------------------------------------------------------
import os as _os
CAMERA_BACKEND = _os.environ.get("SHOOTERRANGE_CAMERA_BACKEND", "auto").lower()

# PiCam (Camera Module 3 etc.) parameters. Used only when the picam backend
# is selected. Lower the framerate if CPU saturates on a Pi 4.
PICAM_WIDTH = int(_os.environ.get("SHOOTERRANGE_PICAM_WIDTH", "960"))
PICAM_HEIGHT = int(_os.environ.get("SHOOTERRANGE_PICAM_HEIGHT", "540"))
PICAM_FPS = int(_os.environ.get("SHOOTERRANGE_PICAM_FPS", "60"))

# ---------------------------------------------------------------------------
# Headless mode — no cv2.imshow / cv2.waitKey windows.
#
#   "auto" (default): headless when DISPLAY/WAYLAND_DISPLAY env vars are
#                     unset (typical for a Pi with no monitor attached).
#   "1" / "true":     force headless. Use on a Pi running as a service.
#   "0" / "false":    force GUI. Useful when developing on a Mac/Linux box.
#
# Override at runtime with the SHOOTERRANGE_HEADLESS env var.
# ---------------------------------------------------------------------------
HEADLESS = _os.environ.get("SHOOTERRANGE_HEADLESS", "auto").lower()


def is_headless() -> bool:
    """Resolve the HEADLESS flag to a concrete bool."""
    if HEADLESS in ("1", "true", "yes", "on"):
        return True
    if HEADLESS in ("0", "false", "no", "off"):
        return False
    # auto: assume headless when no display server is reachable.
    return not (_os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY"))

# Path to ffmpeg binary. By default uses whatever is on PATH (works on macOS/Linux
# after `brew install ffmpeg` or `apt install ffmpeg`, and on Windows if ffmpeg.exe
# is on PATH). Override with a full path if needed, e.g.:
#   macOS Homebrew (Apple Silicon): "/opt/homebrew/bin/ffmpeg"
#   macOS Homebrew (Intel):         "/usr/local/bin/ffmpeg"
#   Windows:                        r"C:\path\to\ffmpeg.exe"
import shutil as _shutil
FFMPEG_PATH = _shutil.which("ffmpeg") or "ffmpeg"
RTSP_TRANSPORT = "udp"
JPEG_SCALE = "960:-1"

# HSV color ranges for laser dot detection.
# Red wraps around hue 0/180 → uses two ranges that get OR-ed together.
# Tune these for your specific laser + camera combo. Use the test script
# in README to print the actual HSV value the camera sees.
#
# RED_1/RED_2  = traditional 650 nm red laser (e.g. cheap pointers)
# PURPLE/BLUE  = bright laser dots that overexpose CM3's sensor and read
#                as violet/blue around the saturated white core.
LOWER_RED_1 = (0, 120, 120)
UPPER_RED_1 = (10, 255, 255)

LOWER_RED_2 = (163, 120, 120)
UPPER_RED_2 = (179, 255, 255)

# Purple / violet (laser core that overexposed the sensor → halo reads
# as deep blue-violet). Some cameras blow the core out to near-white which
# lowers saturation, so keep S/V thresholds a bit more permissive.
LOWER_PURPLE = (105, 40, 80)
UPPER_PURPLE = (165, 255, 255)

MIN_AREA = 2
MAX_AREA = 5000

# Minimum gap between two accepted hits, in milliseconds. Acts as a
# debouncer so a single laser pulse that lights up multiple consecutive
# camera frames isn't counted twice. Lower → more rapid-fire throughput,
# but raises false-double risk if your laser pulse is long.
#  - 300 ms = max ~3 hits/s (very conservative, original default)
#  - 120 ms = max ~8 hits/s, comfortable for fast trigger work
#  - 80 ms  = max ~12 hits/s (close to the human limit)
SHOT_COOLDOWN_MS = 120

# Feature flag: master switch for the whole Unity UDP sender.
# When False, no UnitySender is created, no socket is opened, and no packets are sent.
UNITY_FF = False

UNITY_UDP_ENABLED = True
UNITY_HOST = "127.0.0.1"
UNITY_PORT = 5055
UNITY_INVERT_Y = True

CALIBRATION_ENABLED = True

# Square ROI representing the 170x170 mm ISSF 10m air pistol target paper
TARGET_X1 = 270
TARGET_Y1 = 130
TARGET_X2 = 630
TARGET_Y2 = 490

# ISSF 10m air pistol target (mm)
TARGET_PAPER_MM = 170.0
# Ring diameters mm, index 0 = ring 1 (outer), index 9 = ring 10 (bullseye)
RING_DIAMETERS_MM = (155.5, 139.5, 123.5, 107.5, 91.5, 75.5, 59.5, 43.5, 27.5, 11.5)
INNER_TEN_DIAMETER_MM = 5.0  # X ring
PELLET_DIAMETER_MM = 4.5
# Black bull covers rings 7..10 (diameter 59.5 mm)

# ---------------------------------------------------------------------------
# Mobile control server (HTTP API + MJPEG preview consumed by the React Native
# app). On Raspberry Pi 5 acting as a Wi-Fi AP, the mobile app connects to
# http://<pi-ap-ip>:CONTROL_SERVER_PORT (e.g. http://192.168.4.1:8080).
# ---------------------------------------------------------------------------
CONTROL_SERVER_ENABLED = True
CONTROL_SERVER_HOST = "0.0.0.0"
CONTROL_SERVER_PORT = 8080
DEVICE_NAME = "ShooterRange"
DEVICE_ID = "shooterrange-pi-001"
# Replace with a per-device random secret in production.
AUTH_TOKEN = "dev-token"
APP_VERSION = "0.1.0"
# JPEG quality for the mobile preview stream (1-100). Lower = less bandwidth.
MJPEG_QUALITY = 70
