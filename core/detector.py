import cv2
import numpy as np
import time


class LaserDetector:
    def __init__(
        self,
        lower_red_1,
        upper_red_1,
        lower_red_2,
        upper_red_2,
        min_area=2,
        max_area=120,
        shot_cooldown_ms=120,
        # Optional third HSV range — used for purple/violet "halos" that
        # appear when a bright red/green laser overexposes the CSI sensor.
        # Pass None on both to disable the third mask.
        lower_extra=None,
        upper_extra=None,
    ):
        self.lower_red_1 = np.array(lower_red_1, dtype=np.uint8)
        self.upper_red_1 = np.array(upper_red_1, dtype=np.uint8)
        self.lower_red_2 = np.array(lower_red_2, dtype=np.uint8)
        self.upper_red_2 = np.array(upper_red_2, dtype=np.uint8)
        self.lower_extra = (
            np.array(lower_extra, dtype=np.uint8) if lower_extra is not None else None
        )
        self.upper_extra = (
            np.array(upper_extra, dtype=np.uint8) if upper_extra is not None else None
        )

        self.min_area = min_area
        self.max_area = max_area
        self.shot_cooldown_ms = shot_cooldown_ms

        self.last_shot_time = 0
        self.kernel = np.ones((3, 3), np.uint8)
        self.armed = True

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.lower_red_1, self.upper_red_1)
        mask2 = cv2.inRange(hsv, self.lower_red_2, self.upper_red_2)
        mask = cv2.bitwise_or(mask1, mask2)
        if self.lower_extra is not None and self.upper_extra is not None:
            mask3 = cv2.inRange(hsv, self.lower_extra, self.upper_extra)
            mask = cv2.bitwise_or(mask, mask3)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

        best_idx = -1
        best_area = 0
        all_areas = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            all_areas.append(area)
            if self.min_area <= area <= self.max_area and area > best_area:
                best_area = area
                best_idx = i

        if all_areas:
            print(f"[DEBUG] blobs={len(all_areas)} areas={all_areas} min_cfg={self.min_area} max_cfg={self.max_area} accepted={'YES' if best_idx != -1 else 'NO'}")

        if best_idx == -1:
            self.armed = True
            return None, mask

        cx, cy = centroids[best_idx]
        cx, cy = int(cx), int(cy)

        return {
            "x": cx,
            "y": cy,
            "area": int(best_area),
            "nx": cx / frame.shape[1],
            "ny": cy / frame.shape[0],
        }, mask

    def should_emit_shot(self):
        if not self.armed:
            return False
        now_ms = int(time.time() * 1000)
        if now_ms - self.last_shot_time >= self.shot_cooldown_ms:
            self.last_shot_time = now_ms
            self.armed = False
            return True
        return False