import socket
import json


class UnitySender:
    def __init__(self, host="127.0.0.1", port=5055, enabled=True, invert_y=True):
        self.host = host
        self.port = port
        self.enabled = enabled
        self.invert_y = invert_y
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_hit(self, detection):
        if not self.enabled:
            return

        ny = float(detection["ny"])
        if self.invert_y:
            ny = 1.0 - ny

        payload = {
            "type": "hit",
            "nx": float(detection["nx"]),
            "ny": ny,
            "x": int(detection["x"]),
            "y": int(detection["y"]),
            "area": int(detection["area"]),
        }

        message = json.dumps(payload).encode("utf-8")
        self.sock.sendto(message, (self.host, self.port))

    def close(self):
        if self.sock:
            self.sock.close()