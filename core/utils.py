import time
import cv2


class FPSTimer:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0.0

    def update(self):
        current = time.time()
        dt = current - self.prev_time
        self.prev_time = current
        if dt > 0:
            self.fps = 1.0 / dt
        return self.fps


def resize_to_fit(frame, max_width=1000, max_height=700):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))