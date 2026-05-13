import math
import os
import time

import cv2
import numpy as np

# Sound playback used to live here (pygame.mixer) but the mobile app now
# owns shot/reset audio so it can be muted per-user via Settings → Sound
# and so users without speakers on the Pi still get audio feedback.
# Kept as a no-op for backward compatibility with existing call sites.
def play_shoot():
    return

import config
from config import (
    RTSP_URL,
    WINDOW_NAME,
    DISPLAY_MAX_WIDTH,
    DISPLAY_MAX_HEIGHT,
    SHOW_FPS,
    FFMPEG_PATH,
    RTSP_TRANSPORT,
    JPEG_SCALE,
    LOWER_RED_1,
    UPPER_RED_1,
    LOWER_RED_2,
    UPPER_RED_2,
    LOWER_PURPLE,
    UPPER_PURPLE,
    MIN_AREA,
    MAX_AREA,
    SHOT_COOLDOWN_MS,
    UNITY_FF,
    UNITY_UDP_ENABLED,
    UNITY_HOST,
    UNITY_PORT,
    UNITY_INVERT_Y,
    CALIBRATION_ENABLED,
    TARGET_X1,
    TARGET_Y1,
    TARGET_X2,
    TARGET_Y2,
    TARGET_PAPER_MM,
    RING_DIAMETERS_MM,
    INNER_TEN_DIAMETER_MM,
    PELLET_DIAMETER_MM,
    CONTROL_SERVER_ENABLED,
    CONTROL_SERVER_HOST,
    CONTROL_SERVER_PORT,
    DEVICE_NAME,
    DEVICE_ID,
    AUTH_TOKEN,
    APP_VERSION,
    MJPEG_QUALITY,
    is_headless,
)

from core.camera import create_camera_stream
from core.ffmpeg_stream import FFmpegStream  # noqa: F401  (kept for type hints / external imports)
from core.utils import FPSTimer, resize_to_fit
from core.detector import LaserDetector
from core.unity_sender import UnitySender
from core.control_server import ControlState, start_control_server
from core.calibration_tweaks import (
    CalibrationTweaks,
    load_tweaks,
    save_tweaks,
)


def clamp01(v):
    return max(0.0, min(1.0, v))


def point_inside_target(x, y, x1, y1, x2, y2):
    return x1 <= x <= x2 and y1 <= y <= y2


def map_point_to_target(x, y, x1, y1, x2, y2):
    if x2 <= x1 or y2 <= y1:
        return None

    nx = (x - x1) / float(x2 - x1)
    ny = (y - y1) / float(y2 - y1)

    return clamp01(nx), clamp01(ny)


BULL_TO_PAPER_RATIO = TARGET_PAPER_MM / 59.5  # bull (rings 7-10) is 59.5 mm
BULL_DIAMETER_MM = 59.5


def detect_bull_ellipse(frame):
    """Fit ellipse to dark bullseye for perspective-correct alignment.
    Returns (cx, cy, semi_a_px, semi_b_px, angle_deg) or None.
    OpenCV fitEllipse: angle is rotation of the FIRST axis (width) from x-axis.
    Here we store: a = semi-axis along the width direction, b = along height.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    h, w = gray.shape
    frame_area = h * w
    best = None
    best_score = 0.0
    for c in contours:
        if len(c) < 30:
            continue
        area = cv2.contourArea(c)
        if area < frame_area * 0.0005 or area > frame_area * 0.5:
            continue
        try:
            (cx, cy), (W, H), angle = cv2.fitEllipse(c)
        except cv2.error:
            continue
        if W <= 0 or H <= 0:
            continue
        a = W / 2.0
        b = H / 2.0
        ratio = min(a, b) / max(a, b)
        if ratio < 0.4:
            continue
        ellipse_area = math.pi * a * b
        if ellipse_area <= 0:
            continue
        fill = area / ellipse_area
        if fill < 0.7 or fill > 1.25:
            continue
        score = area * (0.6 + 0.4 * ratio)
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy), float(a), float(b), float(angle))
    return best


def detect_concentric_ring_ellipses(frame, bull, max_dist_factor=0.25):
    """Find printed ring ellipses concentric with bull using edge contours.

    Returns list of (radius_mm, ellipse) where ellipse = (cx, cy, a, b, angle).
    Only ellipses that are clearly concentric with `bull` and whose size matches
    a known ISSF ring (within tight tolerance) are returned.
    """
    if bull is None:
        return []
    bcx, bcy, ba, bb, _ = bull
    bull_r_mm = BULL_DIAMETER_MM / 2.0
    expected_mm = [d / 2.0 for d in RING_DIAMETERS_MM]  # ring 1..10 radii

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 60, 60)
    edges = cv2.Canny(gray, 40, 120)

    # Crop to expected paper region to avoid background noise
    h, w = gray.shape
    pad = int(max(ba, bb) * (TARGET_PAPER_MM / BULL_DIAMETER_MM) * 1.15)
    x0 = max(0, int(bcx - pad)); y0 = max(0, int(bcy - pad))
    x1 = min(w, int(bcx + pad)); y1 = min(h, int(bcy + pad))
    edges_crop = edges[y0:y1, x0:x1].copy()

    contours, _ = cv2.findContours(edges_crop, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    bull_axis_max = max(ba, bb)
    matches = []  # (radius_mm, (cx,cy,a,b,ang), error_ratio)
    for c in contours:
        if len(c) < 40:
            continue
        try:
            (ex, ey), (W, H), ang = cv2.fitEllipse(c)
        except cv2.error:
            continue
        if W <= 0 or H <= 0:
            continue
        a = W / 2.0
        b = H / 2.0
        # Reject contours that don't actually look like an ellipse arc
        # (residual: avg distance from contour points to fitted ellipse should be small)
        ex_full = ex + x0
        ey_full = ey + y0
        # Concentric with bull?
        d_center = math.hypot(ex_full - bcx, ey_full - bcy)
        if d_center > bull_axis_max * max_dist_factor:
            continue
        bull_ratio = min(ba, bb) / max(ba, bb)
        e_ratio = min(a, b) / max(a, b)
        if abs(bull_ratio - e_ratio) > 0.12:
            continue
        # Residual check via sampled ellipse vs contour bounding
        est_mm = (max(a, b) / max(ba, bb)) * (BULL_DIAMETER_MM / 2.0)
        cand = [(abs(est_mm - rm), rm) for rm in (d / 2.0 for d in RING_DIAMETERS_MM)
                if rm > (BULL_DIAMETER_MM / 2.0) * 1.05]
        if not cand:
            continue
        cand.sort()
        err, snap_mm = cand[0]
        if err / snap_mm > 0.04:  # tighter: within 4%
            continue
        # Contour length should be ~ ellipse perimeter (filter partial arcs strongly)
        perim = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        if perim > 0 and len(c) < perim * 0.55:
            continue
        matches.append((snap_mm, (float(ex_full), float(ey_full),
                                  float(a), float(b), float(ang)),
                        err / snap_mm))

    # Deduplicate: keep best match per ring
    by_ring = {}
    for snap_mm, ell, err in matches:
        if snap_mm not in by_ring or by_ring[snap_mm][2] > err:
            by_ring[snap_mm] = (snap_mm, ell, err)
    return [(snap_mm, ell) for snap_mm, ell, _ in by_ring.values()]


def refine_scale_from_rings(frame, bull):
    """Use multi-ellipse fit on printed rings to compute precise paper geometry.
    Returns dict {
        'Pa': float, 'Pb': float, 'angle': float,
        'rings_by_mm': {radius_mm: (cx, cy, a, b, angle)} for directly detected rings
    } or None.

    Per-ring estimates are weighted by their radius squared so the OUTER rings
    dominate the fit. This matches what the operator actually cares about
    (scoring near ring 1) and prevents inner-ring jitter from drifting the
    paper outline outwards.
    """
    if bull is None:
        return None
    bcx, bcy, ba, bb, ang = bull
    rings = detect_concentric_ring_ellipses(frame, bull)
    if not rings:
        return None
    outer_mm = RING_DIAMETERS_MM[0] / 2.0
    weighted = []  # list of (weight, Pa_est, Pb_est, angle, snap_mm, ellipse)
    rings_by_mm = {}
    for snap_mm, (ex, ey, a, b, eang) in rings:
        # Normalize so 'a_major' is along true major axis
        if a >= b:
            a_major, b_minor, ang_major = a, b, eang
        else:
            a_major, b_minor, ang_major = b, a, (eang + 90.0) % 180.0
        scale_to_outer = outer_mm / snap_mm
        Pa_est = a_major * scale_to_outer
        Pb_est = b_minor * scale_to_outer
        # Weight by radius^2: outer rings (large snap_mm) are more reliable
        # for outer-ring scale than inner rings.
        weight = (snap_mm / outer_mm) ** 2
        weighted.append((weight, Pa_est, Pb_est, ang_major))
        # Keep best (closest center) per ring
        prev = rings_by_mm.get(snap_mm)
        if prev is None:
            rings_by_mm[snap_mm] = (ex, ey, a_major, b_minor, ang_major)
        else:
            d_new = math.hypot(ex - bcx, ey - bcy)
            d_old = math.hypot(prev[0] - bcx, prev[1] - bcy)
            if d_new < d_old:
                rings_by_mm[snap_mm] = (ex, ey, a_major, b_minor, ang_major)

    total_w = sum(w for w, _, _, _ in weighted)
    if total_w <= 0:
        return None

    # Robust fit: drop ring estimates whose Pa or Pb is more than 1.5× MAD
    # away from the median. A single bad ring (e.g. Canny edge that latched
    # onto the OUTSIDE of a printed line) can otherwise drag the mean out
    # by 3-5%, which is enough to make the outermost cyan ring extend past
    # the printed paper. Olympic-grade accuracy needs this filtering.
    def _median(xs):
        s = sorted(xs)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    pas = [pa for _, pa, _, _ in weighted]
    pbs = [pb for _, _, pb, _ in weighted]
    med_pa = _median(pas)
    med_pb = _median(pbs)
    mad_pa = max(_median([abs(p - med_pa) for p in pas]), med_pa * 0.005)
    mad_pb = max(_median([abs(p - med_pb) for p in pbs]), med_pb * 0.005)
    kept = [
        (w, pa, pb, a)
        for (w, pa, pb, a) in weighted
        if abs(pa - med_pa) <= mad_pa * 1.5
        and abs(pb - med_pb) <= mad_pb * 1.5
    ]
    if kept:
        weighted = kept
    total_w = sum(w for w, _, _, _ in weighted)
    if total_w <= 0:
        return None

    Pa = sum(w * pa for w, pa, _, _ in weighted) / total_w
    Pb = sum(w * pb for w, _, pb, _ in weighted) / total_w
    sin_sum = sum(w * math.sin(2 * math.radians(a)) for w, _, _, a in weighted)
    cos_sum = sum(w * math.cos(2 * math.radians(a)) for w, _, _, a in weighted)
    refined_ang = math.degrees(math.atan2(sin_sum, cos_sum) / 2.0) % 180.0

    # For a near-circular fit the angle is mathematically meaningless and
    # any small estimation noise gets amplified in the visualised rectangle.
    # Snap to 0 when the eccentricity is below 8% (stricter than before).
    if Pa > 0 and Pb > 0:
        ratio = min(Pa, Pb) / max(Pa, Pb)
        if ratio > 0.92:
            refined_ang = 0.0

    return {
        "Pa": Pa, "Pb": Pb, "angle": refined_ang,
        "rings_by_mm": rings_by_mm,
    }


def bull_to_roi(cx, cy, r, frame_w, frame_h):
    half = int(r * BULL_TO_PAPER_RATIO)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(frame_w, cx + half)
    y2 = min(frame_h, cy + half)
    side = min(x2 - x1, y2 - y1)
    cx2 = (x1 + x2) // 2
    cy2 = (y1 + y2) // 2
    h2 = side // 2
    return cx2 - h2, cy2 - h2, cx2 + h2, cy2 + h2


NUM_RINGS = 10


def mm_per_px(x1, x2):
    return TARGET_PAPER_MM / float(x2 - x1)


def score_for_hit(x, y, x1, y1, x2, y2):
    """ISSF scoring with pellet-edge rule.
    Returns (score, dist_mm, max_radius_mm, center_xy).
    A hit counts for ring R if pellet edge touches the ring (i.e.
    distance_from_center - pellet_radius <= ring_radius).
    Highest matching ring wins. 0 if pellet does not touch ring 1.
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    scale = mm_per_px(x1, x2)
    dist_mm = math.hypot(x - cx, y - cy) * scale
    pellet_r = PELLET_DIAMETER_MM / 2.0
    best = 0
    for idx, diam in enumerate(RING_DIAMETERS_MM):
        ring_r = diam / 2.0
        ring_score = idx + 1  # idx 0 -> ring 1, idx 9 -> ring 10
        if dist_mm - pellet_r <= ring_r:
            if ring_score > best:
                best = ring_score
    max_r_mm = RING_DIAMETERS_MM[0] / 2.0
    return best, dist_mm, max_r_mm, (int(cx), int(cy))


def draw_target_rings(frame, x1, y1, x2, y2, color=(255, 0, 0)):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    scale_px_per_mm = (x2 - x1) / TARGET_PAPER_MM
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    # Draw rings 1..10 (true ISSF diameters)
    for idx, diam in enumerate(RING_DIAMETERS_MM):
        r = int(diam / 2.0 * scale_px_per_mm)
        # Black fill background for rings 7..10 (bull) on first pass
        ring_score = idx + 1
        ring_color = color
        cv2.circle(frame, (cx, cy), r, ring_color, 1, cv2.LINE_AA)
    # Inner 10 (X)
    rx = int(INNER_TEN_DIAMETER_MM / 2.0 * scale_px_per_mm)
    cv2.circle(frame, (cx, cy), rx, color, 1, cv2.LINE_AA)
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 12, 1)


def draw_scoreboard(frame, total, shots, last_score, last_dist, frame_w):
    panel_w = 230
    panel_h = 130
    x0 = frame_w - panel_w - 10
    y0 = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 215, 255), 2)

    cv2.putText(frame, "SCORE", (x0 + 12, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(total), (x0 + 110, y0 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    avg = (total / shots) if shots > 0 else 0.0
    cv2.putText(frame, f"Shots: {shots}  Avg: {avg:.2f}", (x0 + 12, y0 + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    last_txt = "Last: -" if last_score is None else f"Last: {last_score}  ({last_dist:.0f}px)"
    cv2.putText(frame, last_txt, (x0 + 12, y0 + 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def draw_multiline_text(frame, lines, x, y, color=(255, 255, 255), scale=0.6, thickness=2, line_gap=24):
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )


def build_paper_corners_and_homography(bull_geom, ring_scale_axes, keystone_h, keystone_v,
                                       paper_rotation_deg=0.0, paper_scale=1.0,
                                       keystone_d1=0.0, keystone_d2=0.0):
    """Build the 4 image-space corners of the paper and a homography that
    maps canonical paper-mm coordinates (origin = TL, paper_mm × paper_mm)
    to those image corners.

    Without keystone tweaks the corners are the rectangle defined by the
    bull's rotated ellipse extended to ring 1 then to paper. Keystone
    tweaks shrink/grow opposing edges to simulate camera tilt.

      keystone_h / keystone_v  : axis-aligned tilts (bottom vs top, right vs left)
      keystone_d1              : pulls TL inward + pushes BR outward (TL↔BR diag)
      keystone_d2              : pulls TR inward + pushes BL outward (TR↔BL diag)

    ``paper_rotation_deg`` / ``paper_scale`` add an extra rotation/scale
    applied to the paper rectangle ONLY (does not affect ring projection
    when the same homography is reused for scoring).

    Returns (corners_image_4x2 np.ndarray, H_3x3 np.ndarray) or None.
    """
    if bull_geom is None or ring_scale_axes is None:
        return None
    bcx, bcy, _ba, _bb, bang = bull_geom
    Pa_px, Pb_px = ring_scale_axes
    paper_to_ring1 = TARGET_PAPER_MM / RING_DIAMETERS_MM[0]
    half_a = Pa_px * paper_to_ring1 * float(paper_scale)
    half_b = Pb_px * paper_to_ring1 * float(paper_scale)
    theta = math.radians(bang + float(paper_rotation_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # In the bull-axis frame the four paper corners are:
    #   TL=(-A,-B)  TR=(+A,-B)  BR=(+A,+B)  BL=(-A,+B)
    kh = float(keystone_h)
    kv = float(keystone_v)
    raw = [
        (-half_a * (1.0 - kh), -half_b * (1.0 - kv)),  # TL
        (+half_a * (1.0 - kh), -half_b * (1.0 + kv)),  # TR
        (+half_a * (1.0 + kh), +half_b * (1.0 + kv)),  # BR
        (-half_a * (1.0 + kh), +half_b * (1.0 - kv)),  # BL
    ]

    # Diagonal keystone: shift each corner along its outward diagonal unit
    # vector. For a square reference the unit vectors are (±1, ±1)/√2.
    # d1 acts on the TL/BR diagonal, d2 on the TR/BL diagonal.
    d1 = float(keystone_d1)
    d2 = float(keystone_d2)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    # Per-corner diagonal offsets (rx, ry) added to ``raw`` above.
    diag_shifts = [
        (+d1 * half_a * inv_sqrt2, +d1 * half_b * inv_sqrt2),  # TL pulled toward (-,-) when d1>0 (inward)
        (-d2 * half_a * inv_sqrt2, +d2 * half_b * inv_sqrt2),  # TR pulled toward (+,-) inward when d2>0
        (-d1 * half_a * inv_sqrt2, -d1 * half_b * inv_sqrt2),  # BR pushed outward when d1>0
        (+d2 * half_a * inv_sqrt2, -d2 * half_b * inv_sqrt2),  # BL pushed outward when d2>0
    ]
    raw = [
        (rx + dx, ry + dy)
        for (rx, ry), (dx, dy) in zip(raw, diag_shifts)
    ]

    corners_image = np.empty((4, 2), dtype=np.float32)
    for i, (rx, ry) in enumerate(raw):
        corners_image[i, 0] = bcx + rx * cos_t - ry * sin_t
        corners_image[i, 1] = bcy + rx * sin_t + ry * cos_t

    paper_pts = np.array([
        [0.0, 0.0],
        [TARGET_PAPER_MM, 0.0],
        [TARGET_PAPER_MM, TARGET_PAPER_MM],
        [0.0, TARGET_PAPER_MM],
    ], dtype=np.float32)
    h_paper_to_image = cv2.getPerspectiveTransform(paper_pts, corners_image)
    return corners_image, h_paper_to_image


def project_paper_circle(h_paper_to_image, cx_mm, cy_mm, r_mm, n=96):
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    pts = np.empty((n, 1, 2), dtype=np.float32)
    pts[:, 0, 0] = cx_mm + r_mm * np.cos(angles)
    pts[:, 0, 1] = cy_mm + r_mm * np.sin(angles)
    return cv2.perspectiveTransform(pts, h_paper_to_image).reshape(-1, 2).astype(np.int32)


def main():
    # Pick the camera backend (PiCam on a Pi, FFmpeg/RTSP on a Mac dev box,
    # or whatever was forced via SHOOTERRANGE_CAMERA_BACKEND). All three
    # share the same open()/read()/release() interface so the rest of the
    # pipeline stays identical.
    stream = create_camera_stream(config)

    if not stream.open():
        print("Camera stream failed to open.")
        return

    # Headless mode = no cv2 windows. On a Pi running as a service we still
    # want the detector loop, the control server, and the MJPEG preview to
    # work — only the local debug windows are skipped.
    headless = is_headless()
    print(f"[app] headless={headless} (gui={'off' if headless else 'on'})")

    detector = LaserDetector(
        lower_red_1=LOWER_RED_1,
        upper_red_1=UPPER_RED_1,
        lower_red_2=LOWER_RED_2,
        upper_red_2=UPPER_RED_2,
        lower_extra=LOWER_PURPLE,
        upper_extra=UPPER_PURPLE,
        min_area=MIN_AREA,
        max_area=MAX_AREA,
        shot_cooldown_ms=SHOT_COOLDOWN_MS,
    )

    button_detector = LaserDetector(
        lower_red_1=LOWER_RED_1,
        upper_red_1=UPPER_RED_1,
        lower_red_2=LOWER_RED_2,
        upper_red_2=UPPER_RED_2,
        lower_extra=LOWER_PURPLE,
        upper_extra=UPPER_PURPLE,
        min_area=MIN_AREA,
        max_area=MAX_AREA,
        shot_cooldown_ms=SHOT_COOLDOWN_MS,
    )

    if UNITY_FF:
        unity_sender = UnitySender(
            host=UNITY_HOST,
            port=UNITY_PORT,
            enabled=UNITY_UDP_ENABLED,
            invert_y=UNITY_INVERT_Y
        )
    else:
        unity_sender = None

    fps_timer = FPSTimer()

    total_score = 0
    shot_count = 0
    last_score = None
    last_dist = 0.0
    hit_history = []  # list of (x, y, score)

    # Reset button (bottom-left) - requires 2 consecutive hits
    btn_w, btn_h = 150, 70
    reset_hits = 0
    RESET_HIT_WINDOW_SEC = 3.0  # second hit must arrive within this window
    last_reset_hit_time = 0.0

    # Auto-align state. The bull (dark central disc) is detected every frame
    # and smoothed. Operator tweaks (scale/offset) are loaded once at startup
    # and persist across runs in calibration_tweaks.json.
    auto_align = True
    align_frozen = False
    bull_ema = None  # (cx, cy, a, b, angle)
    EMA_ALPHA = 0.25
    frozen_snapshot = None  # cached (bull_geom, Pa, Pb, rings_by_mm, tx1..ty2, tweaks)
    # Tweaks held in a 1-element list so the HTTP callbacks can mutate the
    # value visible to the cv2 main loop without going through threading
    # primitives. Reads/writes of a single Python list slot are atomic.
    tweaks_box: list[CalibrationTweaks] = [load_tweaks()]
    tweaks = tweaks_box[0]
    print(f"Calibration tweaks loaded: scale={tweaks.scale_factor:.3f} "
          f"offset=({tweaks.offset_x_mm:+.1f},{tweaks.offset_y_mm:+.1f}) mm")

    # Multi-frame Auto Adjust sampling buffer. Lives across frames; populated
    # once the mobile app triggers /api/calibration/auto.
    _auto_samples: list[tuple] = []
    _auto_samples_target: int = 0

    if not headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Laser Mask", cv2.WINDOW_AUTOSIZE)

    # Mobile control server (HTTP API + MJPEG preview).
    control_state = None
    if CONTROL_SERVER_ENABLED:
        control_state = ControlState(target_config={
            "paper_mm": float(TARGET_PAPER_MM),
            "ring_diameters_mm": list(RING_DIAMETERS_MM),
            "inner_ten_mm": float(INNER_TEN_DIAMETER_MM),
            "pellet_mm": float(PELLET_DIAMETER_MM),
            "discipline": "ISSF 10m Air Pistol",
        })
        try:
            start_control_server(
                control_state,
                host=CONTROL_SERVER_HOST,
                port=CONTROL_SERVER_PORT,
                version=APP_VERSION,
                device_name=DEVICE_NAME,
                device_id=DEVICE_ID,
                auth_token=AUTH_TOKEN,
            )

            # Live tweaks: mobile app can read/write via REST.
            def _get_tweaks():
                t = tweaks_box[0]
                return {
                    "scale_factor": t.scale_factor,
                    "offset_x_mm": t.offset_x_mm,
                    "offset_y_mm": t.offset_y_mm,
                    "rotation_deg": t.rotation_deg,
                    "aspect_ratio": t.aspect_ratio,
                    "keystone_h": t.keystone_h,
                    "keystone_v": t.keystone_v,
                    "keystone_d1": t.keystone_d1,
                    "keystone_d2": t.keystone_d2,
                    "paper_rotation_deg": t.paper_rotation_deg,
                    "paper_scale": t.paper_scale,
                }

            def _set_tweaks(body):
                t = tweaks_box[0]
                new_t = CalibrationTweaks(
                    scale_factor=float(body.get("scale_factor", t.scale_factor)),
                    offset_x_mm=float(body.get("offset_x_mm", t.offset_x_mm)),
                    offset_y_mm=float(body.get("offset_y_mm", t.offset_y_mm)),
                    rotation_deg=float(body.get("rotation_deg", t.rotation_deg)),
                    aspect_ratio=float(body.get("aspect_ratio", t.aspect_ratio)),
                    keystone_h=float(body.get("keystone_h", t.keystone_h)),
                    keystone_v=float(body.get("keystone_v", t.keystone_v)),
                    keystone_d1=float(body.get("keystone_d1", t.keystone_d1)),
                    keystone_d2=float(body.get("keystone_d2", t.keystone_d2)),
                    paper_rotation_deg=float(body.get("paper_rotation_deg", t.paper_rotation_deg)),
                    paper_scale=float(body.get("paper_scale", t.paper_scale)),
                ).clamp()
                tweaks_box[0] = new_t
                save_tweaks(new_t)

            control_state.get_tweaks = _get_tweaks
            control_state.set_tweaks = _set_tweaks
        except OSError as e:
            print(f"[control_server] failed to bind {CONTROL_SERVER_HOST}:"
                  f"{CONTROL_SERVER_PORT}: {e}")
            control_state = None

    if UNITY_FF:
        print(f"Unity UDP enabled: {UNITY_UDP_ENABLED}")
        print(f"Sending to Unity: {UNITY_HOST}:{UNITY_PORT}")
    else:
        print("Unity UDP disabled (UNITY_FF=False)")
    print(f"Calibration enabled: {CALIBRATION_ENABLED}")
    if CALIBRATION_ENABLED:
        print(f"Target ROI: ({TARGET_X1}, {TARGET_Y1}) -> ({TARGET_X2}, {TARGET_Y2})")
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("Nema frame-a ili je stream prekinut.")
                break

            frame_h, frame_w = frame.shape[:2]

            # Re-read tweaks each frame so live REST updates from the mobile
            # app take effect immediately.
            tweaks = tweaks_box[0]

            # The mobile "Auto adjust" button asks us to drop any smoothed
            # state and re-detect from scratch using the current frame.
            if control_state is not None and control_state.consume_rerun_request():
                bull_ema = None
                align_frozen = False
                frozen_snapshot = None
                if control_state is not None:
                    control_state.set_frozen(False)
                # Reset the multi-frame sampling buffer too.
                _auto_samples = []
                print("Auto adjust: re-running detection from scratch.")

            # Multi-frame Auto Adjust: collect N fits then commit their median.
            sample_n = (
                control_state.consume_sample_request()
                if control_state is not None
                else 0
            )
            if sample_n > 0:
                _auto_samples = []
                _auto_samples_target = int(sample_n)
                print(f"Auto adjust: collecting {_auto_samples_target} samples...")

            # ---- Bull detection (the spatial prior) ----------------------
            if not align_frozen:
                detected = detect_bull_ellipse(frame)
                if detected is not None:
                    if bull_ema is None or len(bull_ema) != 5:
                        bull_ema = detected
                    else:
                        bull_ema = tuple(
                            EMA_ALPHA * detected[i] + (1 - EMA_ALPHA) * bull_ema[i]
                            for i in range(5)
                        )
            current_bull = bull_ema if (bull_ema is not None and len(bull_ema) == 5) else None

            # Collect a sample for multi-frame Auto Adjust if we're in that
            # mode AND we just got a fresh detection. We accumulate raw bull
            # ellipses; the median is computed once we have enough samples.
            if _auto_samples_target > 0 and current_bull is not None:
                # Re-run a fresh per-frame detection (bypassing EMA) so we
                # get N independent measurements rather than N copies of the
                # same smoothed value.
                fresh = detect_bull_ellipse(frame)
                if fresh is not None:
                    _auto_samples.append(fresh)
                if len(_auto_samples) >= _auto_samples_target:
                    # Commit the median of the samples as the authoritative
                    # bull ellipse (replaces EMA so it sticks).
                    def _med(xs):
                        s = sorted(xs)
                        n = len(s)
                        if n == 0:
                            return 0.0
                        if n % 2 == 1:
                            return s[n // 2]
                        return (s[n // 2 - 1] + s[n // 2]) / 2.0

                    cxs = [s[0] for s in _auto_samples]
                    cys = [s[1] for s in _auto_samples]
                    aas = [s[2] for s in _auto_samples]
                    bbs = [s[3] for s in _auto_samples]
                    # Robust angle mean via doubled-angle vector sum.
                    sin_sum = sum(math.sin(2 * math.radians(s[4])) for s in _auto_samples)
                    cos_sum = sum(math.cos(2 * math.radians(s[4])) for s in _auto_samples)
                    ang_med = math.degrees(math.atan2(sin_sum, cos_sum) / 2.0) % 180.0
                    bull_ema = (
                        _med(cxs), _med(cys),
                        _med(aas), _med(bbs),
                        ang_med,
                    )
                    print(
                        f"Auto adjust done: median bull center=({bull_ema[0]:.1f},{bull_ema[1]:.1f}) "
                        f"axes=({bull_ema[2]:.1f},{bull_ema[3]:.1f}) ang={bull_ema[4]:.2f}° "
                        f"from {len(_auto_samples)} samples"
                    )
                    _auto_samples = []
                    _auto_samples_target = 0
                    current_bull = bull_ema
                    if control_state is not None:
                        control_state.signal_sample_done()

            # ---- Compute per-frame bull geometry + Pa/Pb (ring 1 axes) ----
            tx1, ty1, tx2, ty2 = TARGET_X1, TARGET_Y1, TARGET_X2, TARGET_Y2
            bull_geom = None
            ring_scale_axes = None
            rings_by_mm = {}
            calib_source = "none"

            if align_frozen and frozen_snapshot is not None:
                bull_geom, Pa, Pb, rings_by_mm, tx1, ty1, tx2, ty2 = frozen_snapshot
                ring_scale_axes = (Pa, Pb)
                calib_source = "frozen"
            elif current_bull is not None:
                bcx, bcy, ba, bb, bang = current_bull
                # Apply operator tweaks BEFORE the ring-refine step.
                #   scale_factor  → uniform size multiplier on both semi-axes
                #   aspect_ratio  → extra Y / X scale (1.0 = unchanged)
                #   rotation_deg  → extra rotation on top of the bull's angle
                ba_t = ba * tweaks.scale_factor
                bb_t = bb * tweaks.scale_factor * tweaks.aspect_ratio
                bang = bang + tweaks.rotation_deg
                # When the bull is nearly circular, cv2.fitEllipse returns
                # an essentially random angle. That noise then drives the
                # blue paper rectangle to weird tilts. Snap to 0 (only the
                # operator's rotation_deg tweak remains) when eccentricity
                # is below 5%.
                if ba_t > 0 and bb_t > 0:
                    ratio_bull = min(ba_t, bb_t) / max(ba_t, bb_t)
                    if ratio_bull > 0.95:
                        bang = tweaks.rotation_deg
                tweaked_bull = (bcx, bcy, ba_t, bb_t, bang)
                # Bull-only baseline scale (always available + safe).
                ring1_scale = RING_DIAMETERS_MM[0] / BULL_DIAMETER_MM
                Pa_baseline = ba_t * ring1_scale
                Pb_baseline = bb_t * ring1_scale
                refined = refine_scale_from_rings(frame, tweaked_bull)
                if refined is not None:
                    Pa_r, Pb_r = refined["Pa"], refined["Pb"]
                    # SAFETY: refine_scale_from_rings sometimes latches onto
                    # contours that aren't ISSF rings (table edge, monitor
                    # bezel, ...). Reject any refinement that disagrees with
                    # the bull-only scale by more than 8% — prevents the
                    # cyan rings from blowing up to fill the screen.
                    a_drift = max(Pa_r, Pa_baseline) / max(min(Pa_r, Pa_baseline), 1e-6)
                    b_drift = max(Pb_r, Pb_baseline) / max(min(Pb_r, Pb_baseline), 1e-6)
                    if a_drift <= 1.08 and b_drift <= 1.08:
                        Pa, Pb = Pa_r, Pb_r
                        bang = refined["angle"]
                        rings_by_mm = refined["rings_by_mm"]
                        calib_source = "refined"
                    else:
                        Pa, Pb = Pa_baseline, Pb_baseline
                        calib_source = f"bull-only (refine drift {max(a_drift, b_drift):.2f}x)"
                else:
                    Pa, Pb = Pa_baseline, Pb_baseline
                    calib_source = "bull-only"
                ring_scale_axes = (Pa, Pb)
                bull_geom = (bcx, bcy, ba_t, bb_t, bang)
                # Bounding box for ROI = full paper (so misses on paper still
                # get detected and registered as score 0).
                paper_to_ring1 = TARGET_PAPER_MM / RING_DIAMETERS_MM[0]
                Pa_paper = Pa * paper_to_ring1
                Pb_paper = Pb * paper_to_ring1
                theta = math.radians(bang)
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                hx = math.sqrt((Pa_paper * cos_t) ** 2 + (Pb_paper * sin_t) ** 2)
                hy = math.sqrt((Pa_paper * sin_t) ** 2 + (Pb_paper * cos_t) ** 2)
                tx1 = max(0, int(bcx - hx))
                ty1 = max(0, int(bcy - hy))
                tx2 = min(frame_w, int(bcx + hx))
                ty2 = min(frame_h, int(bcy + hy))

            if CALIBRATION_ENABLED:
                # Mask the rectangular ROI down to the actual paper area.
                # Without this, red blobs *outside* the paper get detected
                # and turn into false hits. We use ring 1 scaled up to the
                # paper extent so even shots that miss ring 1 but land on
                # paper still get registered (as score 0).
                roi = frame[ty1:ty2, tx1:tx2].copy()
                if bull_geom is not None and ring_scale_axes is not None:
                    bcx_full, bcy_full, _ba_g, _bb_g, bang_g = bull_geom
                    Pa_px, Pb_px = ring_scale_axes
                    paper_to_ring1 = TARGET_PAPER_MM / RING_DIAMETERS_MM[0]
                    paper_a = max(1, int(Pa_px * paper_to_ring1))
                    paper_b = max(1, int(Pb_px * paper_to_ring1))
                    paper_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                    cv2.ellipse(
                        paper_mask,
                        (int(bcx_full - tx1), int(bcy_full - ty1)),
                        (paper_a, paper_b),
                        bang_g, 0, 360, 255, -1,
                    )
                    roi[paper_mask == 0] = 0
                detection, roi_mask = detector.detect(roi)
                mask = cv2.copyMakeBorder(
                    roi_mask,
                    ty1,
                    frame_h - ty2,
                    tx1,
                    frame_w - tx2,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
                if detection is not None:
                    detection["x"] += tx1
                    detection["y"] += ty1
            else:
                detection, mask = detector.detect(frame)

            # Draw calibration overlay (paper-shaped quad + ISSF rings).
            # Uses a single homography from canonical paper-mm space so that
            # rotation, aspect, AND keystone tweaks all combine correctly.
            paper_homography = None  # set when we draw via homography
            if CALIBRATION_ENABLED:
                if bull_geom is not None and ring_scale_axes is not None:
                    bcx, bcy, ba, bb, bang = bull_geom
                    Pa_px, Pb_px = ring_scale_axes
                    color = (255, 180, 0)
                    outer_mm = RING_DIAMETERS_MM[0] / 2.0

                    # Apply operator centre offset (mm → px) along the ellipse's
                    # major/minor axes so the centre tweak is perspective-aware.
                    # When frozen, the snapshot already has the offset baked
                    # in — re-applying it would drift the centre every frame.
                    theta = math.radians(bang)
                    cos_t, sin_t = math.cos(theta), math.sin(theta)
                    if not align_frozen:
                        mm_per_px_a = outer_mm / Pa_px if Pa_px > 0 else 0.0
                        mm_per_px_b = outer_mm / Pb_px if Pb_px > 0 else 0.0
                        off_px_a = tweaks.offset_x_mm / mm_per_px_a if mm_per_px_a > 0 else 0.0
                        off_px_b = tweaks.offset_y_mm / mm_per_px_b if mm_per_px_b > 0 else 0.0
                        bcx = bcx + off_px_a * cos_t - off_px_b * sin_t
                        bcy = bcy + off_px_a * sin_t + off_px_b * cos_t
                        bull_geom = (bcx, bcy, ba, bb, bang)

                    built = build_paper_corners_and_homography(
                        bull_geom, ring_scale_axes,
                        tweaks.keystone_h, tweaks.keystone_v,
                        tweaks.paper_rotation_deg, tweaks.paper_scale,
                        tweaks.keystone_d1, tweaks.keystone_d2,
                    )
                    if built is not None:
                        corners_image, paper_homography = built
                        cv2.polylines(
                            frame, [corners_image.astype(np.int32)],
                            True, (255, 0, 0), 2, cv2.LINE_AA,
                        )
                        # The ring/scoring homography ignores the paper-only
                        # tweaks so adjusting the rectangle never affects
                        # actual scoring.
                        ring_built = build_paper_corners_and_homography(
                            bull_geom, ring_scale_axes,
                            tweaks.keystone_h, tweaks.keystone_v,
                            0.0, 1.0,
                            tweaks.keystone_d1, tweaks.keystone_d2,
                        )
                        ring_homography = ring_built[1] if ring_built is not None else paper_homography
                        cx_mm = TARGET_PAPER_MM / 2.0
                        for diam in RING_DIAMETERS_MM:
                            pts = project_paper_circle(
                                ring_homography, cx_mm, cx_mm, diam / 2.0
                            )
                            cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)
                        pts_inner = project_paper_circle(
                            ring_homography, cx_mm, cx_mm,
                            INNER_TEN_DIAMETER_MM / 2.0,
                        )
                        cv2.polylines(frame, [pts_inner], True, color, 1, cv2.LINE_AA)
                        cv2.drawMarker(
                            frame, (int(bcx), int(bcy)), color,
                            cv2.MARKER_CROSS, 12, 1,
                        )
                        # Use ring_homography for scoring downstream.
                        paper_homography = ring_homography
                else:
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2),
                                  (255, 0, 0), 2)
                    draw_target_rings(frame, tx1, ty1, tx2, ty2,
                                      color=(255, 180, 0))

            # Draw past hits (pellet-sized circles)
            scale_px_per_mm = (tx2 - tx1) / TARGET_PAPER_MM if tx2 > tx1 else 1.0
            pellet_r_px = max(2, int(PELLET_DIAMETER_MM / 2.0 * scale_px_per_mm))
            for hx, hy, hs in hit_history[-30:]:
                cv2.circle(frame, (hx, hy), pellet_r_px, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (hx, hy), 2, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(frame, str(hs), (hx + pellet_r_px + 2, hy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

            if detection is not None:
                raw_x = detection["x"]
                raw_y = detection["y"]
                area = detection["area"]

                raw_nx = raw_x / float(frame_w)
                raw_ny = raw_y / float(frame_h)

                calibrated = False
                inside_roi = False
                send_detection = None

                if CALIBRATION_ENABLED:
                    inside_roi = point_inside_target(
                        raw_x, raw_y,
                        tx1, ty1,
                        tx2, ty2
                    )

                    if inside_roi:
                        mapped = map_point_to_target(
                            raw_x, raw_y,
                            tx1, ty1,
                            tx2, ty2
                        )

                        if mapped is not None:
                            cal_nx, cal_ny = mapped
                            calibrated = True

                            send_detection = {
                                "x": raw_x,
                                "y": raw_y,
                                "area": area,
                                "nx": cal_nx,
                                "ny": cal_ny,
                            }
                else:
                    send_detection = {
                        "x": raw_x,
                        "y": raw_y,
                        "area": area,
                        "nx": raw_nx,
                        "ny": raw_ny,
                    }

                # Don't register hits until calibration is confirmed (frozen with 'n')
                if (
                    detector.should_emit_shot()
                    and send_detection is not None
                    and (align_frozen or not auto_align)
                ):
                    send_nx = send_detection["nx"]
                    send_ny = send_detection["ny"]
                    unity_ny = 1.0 - send_ny if UNITY_INVERT_Y else send_ny

                    # Default fallback (no calibration): rectangle scoring.
                    hit_score, hit_dist, hit_maxr, _ = score_for_hit(
                        raw_x, raw_y, tx1, ty1, tx2, ty2
                    )
                    x_mm_centered = 0.0
                    y_mm_centered = 0.0
                    is_inner_ten = False

                    if paper_homography is not None:
                        # Best path: invert the paper homography so the
                        # pellet pixel goes straight into paper-mm space.
                        # This is the only correct way to handle keystone.
                        h_image_to_paper = np.linalg.inv(paper_homography)
                        pt = np.array([[[float(raw_x), float(raw_y)]]],
                                      dtype=np.float32)
                        out = cv2.perspectiveTransform(pt, h_image_to_paper)
                        px_mm = float(out[0, 0, 0])
                        py_mm = float(out[0, 0, 1])
                        cx_mm = TARGET_PAPER_MM / 2.0
                        x_mm_centered = px_mm - cx_mm
                        y_mm_centered = py_mm - cx_mm
                        dist_mm = math.hypot(x_mm_centered, y_mm_centered)
                        pellet_r = PELLET_DIAMETER_MM / 2.0
                        best = 0
                        for idx, diam in enumerate(RING_DIAMETERS_MM):
                            if dist_mm - pellet_r <= diam / 2.0:
                                best = max(best, idx + 1)
                        hit_score = best
                        hit_dist = dist_mm
                        is_inner_ten = (
                            hit_score == 10
                            and (dist_mm + pellet_r) <= (INNER_TEN_DIAMETER_MM / 2.0)
                        )
                    elif bull_geom is not None and ring_scale_axes is not None:
                        # Fallback: score against the ring-1-scaled ellipse frame.
                        bcx, bcy, ba, bb, bang = bull_geom
                        Pa_px, Pb_px = ring_scale_axes
                        theta = math.radians(bang)
                        cos_t, sin_t = math.cos(theta), math.sin(theta)
                        dx = raw_x - bcx
                        dy = raw_y - bcy
                        rx = dx * cos_t + dy * sin_t
                        ry = -dx * sin_t + dy * cos_t
                        outer_mm = RING_DIAMETERS_MM[0] / 2.0
                        nd = math.hypot(rx / Pa_px, ry / Pb_px)  # 1.0 == ring 1 edge
                        dist_mm = nd * outer_mm
                        pellet_r = PELLET_DIAMETER_MM / 2.0
                        best = 0
                        for idx, diam in enumerate(RING_DIAMETERS_MM):
                            if dist_mm - pellet_r <= diam / 2.0:
                                best = max(best, idx + 1)
                        hit_score = best
                        hit_dist = dist_mm
                        x_mm_centered = (rx / max(Pa_px, 1e-6)) * outer_mm
                        y_mm_centered = (ry / max(Pb_px, 1e-6)) * outer_mm
                        is_inner_ten = (
                            hit_score == 10
                            and (dist_mm + pellet_r) <= (INNER_TEN_DIAMETER_MM / 2.0)
                        )

                    # Decide what to do with this detection.
                    # The "paper area" is roughly a square of side TARGET_PAPER_MM
                    # around the bull centre. A real shot lands inside that
                    # area. A noise blob far outside isn't a shot at all.
                    ring1_radius_mm = RING_DIAMETERS_MM[0] / 2.0
                    paper_half_mm = TARGET_PAPER_MM / 2.0
                    on_paper = (
                        abs(x_mm_centered) <= paper_half_mm
                        and abs(y_mm_centered) <= paper_half_mm
                    )
                    # Within ~3mm past ring 1 we still consider it a real but
                    # missed shot (score 0). Beyond paper bounds it's noise.
                    is_real_shot = on_paper and hit_dist <= (paper_half_mm + 3.0)

                    if not is_real_shot:
                        print(
                            f"DROP raw=({raw_x},{raw_y}) dist={hit_dist:.1f}mm "
                            f"area={area} score={hit_score} (off paper)"
                        )
                    else:
                        # Hits outside ring 1 (score 0) still count: a fired
                        # shot is a fired shot, just worth zero points.
                        total_score += hit_score
                        shot_count += 1
                        last_score = hit_score
                        last_dist = hit_dist
                        hit_history.append((raw_x, raw_y, hit_score))

                        kind = "MISS" if hit_score == 0 else "HIT"
                        print(
                            f"{kind} raw=({raw_x},{raw_y}) "
                            f"raw_norm=({raw_nx:.4f},{raw_ny:.4f}) "
                            f"send_norm=({send_nx:.4f},{send_ny:.4f}) "
                            f"unity_norm=({send_nx:.4f},{unity_ny:.4f}) "
                            f"area={area} score={hit_score} dist={hit_dist:.1f}/{hit_maxr:.1f} "
                            f"total={total_score} shots={shot_count} "
                            f"inside_roi={inside_roi} calibrated={calibrated}"
                        )

                        send_detection["score"] = hit_score
                        if unity_sender is not None:
                            unity_sender.send_hit(send_detection)

                        # Publish to mobile app (matches HitSchema in src/api/schemas.ts).
                        if control_state is not None:
                            x_norm = clamp01(send_detection.get("nx", raw_nx))
                            y_norm = clamp01(send_detection.get("ny", raw_ny))
                            control_state.publish_hit({
                                "x_norm": x_norm,
                                "y_norm": y_norm,
                                "score": int(hit_score),
                                "ring": int(hit_score),
                                "x_mm": float(x_mm_centered),
                                "y_mm": float(y_mm_centered),
                                "dist_mm": float(max(0.0, hit_dist)),
                                "is_inner_ten": bool(is_inner_ten),
                            })

                        play_shoot()

            if SHOW_FPS:
                fps = fps_timer.update()
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            draw_scoreboard(frame, total_score, shot_count, last_score, last_dist, frame_w)

            # Reset button bbox (bottom-left)
            btn_x1 = 15
            btn_y2 = frame_h - 15
            btn_x2 = btn_x1 + btn_w
            btn_y1 = btn_y2 - btn_h

            # Detect laser-shot inside button area (uses cooldown + arming)
            btn_roi = frame[btn_y1:btn_y2, btn_x1:btn_x2]
            btn_det, _ = button_detector.detect(btn_roi)
            laser_on_button = btn_det is not None
            button_hit_now = laser_on_button and button_detector.should_emit_shot()

            # Drop stale first-hit if window expired
            if reset_hits == 1 and (time.time() - last_reset_hit_time) > RESET_HIT_WINDOW_SEC:
                reset_hits = 0

            if button_hit_now:
                reset_hits += 1
                last_reset_hit_time = time.time()
                print(f"Reset button hit {reset_hits}/2")

            if reset_hits >= 2:
                total_score = 0
                shot_count = 0
                last_score = None
                last_dist = 0.0
                hit_history.clear()
                reset_hits = 0
                print("Score reset by 2 consecutive hits.")

            # Draw button with light-gray semi-transparent background
            btn_color = (0, 215, 255) if laser_on_button else (180, 180, 180)
            overlay = frame.copy()
            cv2.rectangle(overlay, (btn_x1, btn_y1), (btn_x2, btn_y2), (200, 200, 200), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, 2)
            cv2.putText(frame, "RESET", (btn_x1 + 18, btn_y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "RESET", (btn_x1 + 18, btn_y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            status_txt = f"Hits: {reset_hits}/2"
            cv2.putText(frame, status_txt, (btn_x1 + 14, btn_y1 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, status_txt, (btn_x1 + 14, btn_y1 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Calibration debug HUD: shows which calibration path is active so
            # you can immediately see whether paper detection is working.
            hud_txt = (
                f"CAL: {calib_source}  "
                f"s={tweaks.scale_factor:.3f}  "
                f"a={tweaks.aspect_ratio:.3f}  "
                f"r={tweaks.rotation_deg:+.1f}°  "
                f"o=({tweaks.offset_x_mm:+.1f},{tweaks.offset_y_mm:+.1f})mm"
            )
            hud_color = (0, 255, 0) if calib_source == "refined" else (
                (0, 200, 255) if "bull" in calib_source or calib_source == "frozen"
                else (0, 0, 255)
            )
            cv2.putText(frame, hud_txt, (20, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hud_txt, (20, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)

            display_frame = resize_to_fit(frame, DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT)
            display_mask = resize_to_fit(mask, 600, 400)

            if not headless:
                cv2.imshow(WINDOW_NAME, display_frame)
                cv2.imshow("Laser Mask", display_mask)

            # Push the annotated frame to the mobile preview stream — but
            # ONLY when somebody is actually watching. JPEG encoding eats
            # ~3-8 ms per frame on Pi 5, which translates directly to fewer
            # missed laser pulses during live shooting. The `preview_wanted()`
            # check returns True while the calibration screen is polling
            # snapshots (with grace window) or an MJPEG stream is open.
            if control_state is not None and control_state.preview_wanted():
                ok_jpg, jpg_buf = cv2.imencode(
                    ".jpg",
                    display_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, int(MJPEG_QUALITY)],
                )
                if ok_jpg:
                    control_state.push_frame(jpg_buf.tobytes())

            # In headless mode there is no GUI event pump and no key input.
            # The control server's HTTP requests still drive freeze/tweaks,
            # and a tiny sleep keeps CPU sane when the camera is faster than
            # the detector pipeline.
            if headless:
                time.sleep(0.001)
                key = 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                total_score = 0
                shot_count = 0
                last_score = None
                last_dist = 0.0
                hit_history.clear()
                print("Score reset.")
            if key == ord("a"):
                auto_align = not auto_align
                bull_ema = None
                align_frozen = False
                frozen_snapshot = None
                if control_state is not None:
                    control_state.set_frozen(False)
                print(f"Auto-align: {auto_align}")
            # ---- Operator calibration tweaks ---------------------------
            # Scale:    '+' / '-'   ring size  ±0.5%
            # Centre:    h / l       left / right ±0.5 mm
            #            k / j       up / down  ±0.5 mm
            # Rotation:  , / .       rotate  ±0.5°
            # Aspect:    [ / ]       Y/X ratio  ±0.5%
            # Reset:     0           restore defaults
            tweak_changed = False
            tweak_deltas = {
                ord("+"): {"scale_factor": +0.005},
                ord("="): {"scale_factor": +0.005},
                ord("-"): {"scale_factor": -0.005},
                ord("_"): {"scale_factor": -0.005},
                ord("h"): {"offset_x_mm": -0.5},
                ord("l"): {"offset_x_mm": +0.5},
                ord("k"): {"offset_y_mm": -0.5},
                ord("j"): {"offset_y_mm": +0.5},
                ord(","): {"rotation_deg": -0.5},
                ord("."): {"rotation_deg": +0.5},
                ord("["): {"aspect_ratio": -0.005},
                ord("]"): {"aspect_ratio": +0.005},
                ord("y"): {"keystone_h": -0.005},
                ord("u"): {"keystone_h": +0.005},
                ord("i"): {"keystone_v": -0.005},
                ord("o"): {"keystone_v": +0.005},
                # Diagonal keystones: t/g for d1 (TL-BR), b/m for d2 (TR-BL)
                ord("t"): {"keystone_d1": -0.005},
                ord("g"): {"keystone_d1": +0.005},
                ord("b"): {"keystone_d2": -0.005},
                ord("m"): {"keystone_d2": +0.005},
                ord(";"): {"paper_rotation_deg": -0.5},
                ord("'"): {"paper_rotation_deg": +0.5},
                ord("<"): {"paper_scale": -0.005},
                ord(">"): {"paper_scale": +0.005},
            }
            if key in tweak_deltas:
                delta = tweak_deltas[key]
                tweaks = CalibrationTweaks(
                    scale_factor=tweaks.scale_factor + delta.get("scale_factor", 0.0),
                    offset_x_mm=tweaks.offset_x_mm + delta.get("offset_x_mm", 0.0),
                    offset_y_mm=tweaks.offset_y_mm + delta.get("offset_y_mm", 0.0),
                    rotation_deg=tweaks.rotation_deg + delta.get("rotation_deg", 0.0),
                    aspect_ratio=tweaks.aspect_ratio + delta.get("aspect_ratio", 0.0),
                    keystone_h=tweaks.keystone_h + delta.get("keystone_h", 0.0),
                    keystone_v=tweaks.keystone_v + delta.get("keystone_v", 0.0),
                    keystone_d1=tweaks.keystone_d1 + delta.get("keystone_d1", 0.0),
                    keystone_d2=tweaks.keystone_d2 + delta.get("keystone_d2", 0.0),
                    paper_rotation_deg=tweaks.paper_rotation_deg + delta.get("paper_rotation_deg", 0.0),
                    paper_scale=tweaks.paper_scale + delta.get("paper_scale", 0.0),
                ).clamp()
                tweak_changed = True
            if key == ord("0"):
                tweaks = CalibrationTweaks()
                tweak_changed = True
            if tweak_changed:
                tweaks_box[0] = tweaks
                save_tweaks(tweaks)
                print(
                    f"Tweaks: scale={tweaks.scale_factor:.3f} "
                    f"aspect={tweaks.aspect_ratio:.3f} "
                    f"rot={tweaks.rotation_deg:+.1f}° "
                    f"offset=({tweaks.offset_x_mm:+.1f},{tweaks.offset_y_mm:+.1f}) mm"
                )

            # Determine the requested next freeze state from either the 'n' key
            # or a pending HTTP request from the mobile app. HTTP wins if both
            # arrive in the same tick.
            freeze_target = None
            if key == ord("n"):
                freeze_target = not align_frozen
            if control_state is not None:
                http_req = control_state.consume_freeze_request()
                if http_req is not None:
                    freeze_target = bool(http_req)

            if freeze_target is not None and freeze_target != align_frozen:
                align_frozen = freeze_target
                if align_frozen:
                    # Snapshot the current bull-based geometry so subsequent
                    # frames keep the same calibration.
                    if bull_geom is not None and ring_scale_axes is not None:
                        Pa_cache, Pb_cache = ring_scale_axes
                        frozen_snapshot = (bull_geom, Pa_cache, Pb_cache,
                                           dict(rings_by_mm), tx1, ty1, tx2, ty2)
                    else:
                        align_frozen = False
                        print("Cannot freeze: bull not detected yet.")
                else:
                    frozen_snapshot = None
                if control_state is not None:
                    control_state.set_frozen(align_frozen)
                print(f"Align frozen: {align_frozen}")

    finally:
        if unity_sender is not None:
            unity_sender.close()
        stream.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()