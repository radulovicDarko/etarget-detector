import subprocess
import numpy as np
import cv2


class FFmpegStream:
    def __init__(self, ffmpeg_path, rtsp_url, transport="udp", scale="960:540"):
        self.ffmpeg_path = ffmpeg_path
        self.rtsp_url = rtsp_url
        self.transport = transport
        self.scale = scale
        self.process = None
        self.buffer = bytearray()

    def open(self):
        command = [
            self.ffmpeg_path,
            "-loglevel", "error",
            "-rtsp_transport", self.transport,
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "0",
            "-probesize", "32",
            "-i", self.rtsp_url,
            "-vf", f"scale={self.scale}",
            "-an",
            "-c:v", "mjpeg",
            "-q:v", "5",
            "-f", "image2pipe",
            "pipe:1"
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0
        )
        return self.process is not None

    def read(self):
        if self.process is None or self.process.stdout is None:
            return False, None

        while True:
            chunk = self.process.stdout.read(4096)
            if not chunk:
                return False, None

            self.buffer.extend(chunk)

            start = self.buffer.find(b'\xff\xd8')
            end = self.buffer.find(b'\xff\xd9')

            if start != -1 and end != -1 and end > start:
                jpg = self.buffer[start:end + 2]
                del self.buffer[:end + 2]

                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                return True, frame

    def release(self):
        if self.process is None:
            return

        try:
            if self.process.stdout:
                try:
                    self.process.stdout.close()
                except Exception:
                    pass
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
            except Exception:
                pass
            try:
                self.process.wait(timeout=2)
            except Exception:
                pass
        finally:
            self.process = None
            self.buffer = bytearray()