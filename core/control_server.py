"""HTTP + WebSocket control server consumed by the React Native mobile app.

The Pi is intentionally a thin sensor: it detects laser hits and broadcasts
them live, plus exposes calibration tools and a camera preview. It does NOT
store sessions or hits — that responsibility lives on the mobile app and a
remote backend. This keeps the Pi tiny, free of SQLite churn, and avoids
turning it into yet another database to back up.

Endpoints:

  GET  /api/health                       -> {status, version, uptime_s}
  POST /api/pair                         -> {token, device_name, device_id}
  GET  /api/target/config                -> target geometry (mm)
  GET  /api/stream/preview.mjpeg         -> multipart MJPEG (browser/VLC)
  GET  /api/stream/preview.jpg           -> single JPEG snapshot (mobile poll)
  POST /api/calibration/freeze           -> queue freeze (== keypress 'n')
  POST /api/calibration/unfreeze         -> queue unfreeze
  GET  /api/calibration/tweaks           -> current operator tweaks
  POST /api/calibration/tweaks           -> update operator tweaks
  POST /api/calibration/auto             -> reset tweaks + sample N frames
  GET  /ws/hits                          -> WebSocket; pushes hit + calibration
                                            messages matching the mobile schema

Implementation notes:
- Pure stdlib. WebSocket (RFC 6455) text frames are written directly on the
  hijacked HTTP socket after the upgrade handshake — no third-party dep.
- Threading model: ThreadingHTTPServer spawns one thread per connection;
  each WS client thread owns a queue and blocks on it.
- The cv2 main loop pushes JPEGs via ``state.push_frame``, publishes hits via
  ``state.publish_hit``, and consumes pending freeze requests via
  ``state.consume_freeze_request``.
"""
from __future__ import annotations

import base64
import hashlib
import json
import queue
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------

class ControlState:
    """Thread-safe shared state between the cv2 main loop and the HTTP server."""

    def __init__(self, target_config: Optional[Dict[str, Any]] = None) -> None:
        self._frame_lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._frame_event = threading.Event()

        self._req_lock = threading.Lock()
        self._freeze_request: Optional[bool] = None  # True=freeze, False=unfreeze

        self._subs_lock = threading.Lock()
        self._subscribers: List["queue.Queue[str]"] = []

        # Counter incremented while a preview consumer (MJPEG stream OR
        # snapshot poll) is actively waiting for a frame. The cv2 main
        # loop reads this via `preview_wanted()` and skips JPEG encoding
        # entirely when nobody's looking — saves ~3-8 ms/frame on the Pi
        # during live sessions, which translates directly to fewer missed
        # laser pulses.
        self._preview_lock = threading.Lock()
        self._preview_consumers = 0
        # Snapshot polls (preview.jpg) are short — we keep encoding warm
        # for a brief window after the last poll so back-to-back polls
        # don't toggle the encode on/off.
        self._preview_grace_until = 0.0
        self._preview_grace_seconds = 1.5

        self.start_time = time.time()
        self.is_frozen = False  # mirrored from main loop, read by HTTP

        # Optional callbacks for live calibration tweaks. The cv2 main loop
        # registers these so the mobile app can read/update operator tweaks
        # via REST without restarting the Python process.
        self.get_tweaks = None  # () -> dict | None
        self.set_tweaks = None  # (dict) -> None

        # Pending "rerun auto-detection" request from the mobile Auto Adjust
        # button. The cv2 main loop reads + clears this each frame.
        self._rerun_lock = threading.Lock()
        self._rerun_request = False
        # Multi-frame Auto Adjust: when the mobile app calls /auto we ask the
        # cv2 loop to collect N fits, then signal completion via this event
        # so the HTTP handler can return only once the sampling is finished.
        self._sample_request = 0
        self._sample_done = threading.Event()
        self._sample_done.set()

        self.target_config: Dict[str, Any] = target_config or {
            "paper_mm": 170.0,
            "ring_diameters_mm": [
                155.5, 139.5, 123.5, 107.5, 91.5, 75.5, 59.5, 43.5, 27.5, 11.5,
            ],
            "inner_ten_mm": 5.0,
            "pellet_mm": 4.5,
            "discipline": "ISSF 10m Air Pistol",
        }

    # ---- frame I/O ----
    def push_frame(self, jpeg_bytes: bytes) -> None:
        with self._frame_lock:
            self._jpeg = jpeg_bytes
        self._frame_event.set()
        self._frame_event.clear()

    def get_latest_jpeg(self, timeout: float = 1.0) -> Optional[bytes]:
        self._frame_event.wait(timeout=timeout)
        with self._frame_lock:
            return self._jpeg

    # ---- preview consumer accounting ----
    def preview_wanted(self) -> bool:
        """True when at least one HTTP client is reading the preview stream
        OR a snapshot was polled within the grace window. Used by the cv2
        main loop to skip JPEG encoding entirely when nobody is watching.
        """
        with self._preview_lock:
            if self._preview_consumers > 0:
                return True
            return time.time() < self._preview_grace_until

    def preview_consumer_enter(self) -> None:
        with self._preview_lock:
            self._preview_consumers += 1

    def preview_consumer_exit(self) -> None:
        with self._preview_lock:
            if self._preview_consumers > 0:
                self._preview_consumers -= 1

    def preview_snapshot_polled(self) -> None:
        """Mark a one-shot snapshot poll. Keeps encoding warm for a brief
        window so back-to-back polls (the calibration screen polls every
        100 ms) don't churn the on/off boundary on every frame."""
        with self._preview_lock:
            self._preview_grace_until = time.time() + self._preview_grace_seconds

    # ---- freeze request queue ----
    def request_freeze(self, freeze: bool) -> None:
        with self._req_lock:
            self._freeze_request = freeze

    def consume_freeze_request(self) -> Optional[bool]:
        with self._req_lock:
            r = self._freeze_request
            self._freeze_request = None
            return r

    def request_rerun_detection(self) -> None:
        with self._rerun_lock:
            self._rerun_request = True

    def consume_rerun_request(self) -> bool:
        with self._rerun_lock:
            r = self._rerun_request
            self._rerun_request = False
            return r

    def auto_adjust_blocking(self, samples: int = 25, timeout_s: float = 4.0) -> bool:
        """Block the calling thread while the cv2 main loop collects N fresh
        bull/ring fits, then commits their robust median.

        Returns True on success. The actual sampling is done on the cv2 thread
        via the ``samples_request`` flag below; this method just sets it and
        waits for completion.
        """
        with self._rerun_lock:
            self._sample_request = int(samples)
            self._sample_done.clear()
        ok = self._sample_done.wait(timeout=timeout_s)
        return ok

    def consume_sample_request(self) -> int:
        with self._rerun_lock:
            n = self._sample_request
            self._sample_request = 0
            return n

    def signal_sample_done(self) -> None:
        self._sample_done.set()

    def set_frozen(self, frozen: bool) -> None:
        if self.is_frozen == frozen:
            return
        self.is_frozen = frozen
        self._broadcast({"type": "calibration", "state": "frozen" if frozen else "live"})

    # ---- subscribers (WS hit stream) ----
    def add_subscriber(self) -> "queue.Queue[str]":
        q: "queue.Queue[str]" = queue.Queue(maxsize=200)
        with self._subs_lock:
            self._subscribers.append(q)
        return q

    def remove_subscriber(self, q: "queue.Queue[str]") -> None:
        with self._subs_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def subscriber_count(self) -> int:
        with self._subs_lock:
            return len(self._subscribers)

    def _broadcast(self, message: Dict[str, Any]) -> None:
        text = json.dumps(message)
        with self._subs_lock:
            subs = list(self._subscribers)
        n_dropped = 0
        for q in subs:
            try:
                q.put_nowait(text)
            except queue.Full:
                # Slow client — drop oldest by draining one. We log this
                # because a full queue means the WS client (mobile app) is
                # consuming hits slower than we produce them, which is the
                # most common cause of "hits feel laggy" reports.
                n_dropped += 1
                try:
                    q.get_nowait()
                    q.put_nowait(text)
                except Exception:
                    pass
        if n_dropped > 0 and message.get("type") == "hit":
            print(
                f"[ws] WARNING: {n_dropped}/{len(subs)} subscribers were "
                f"full — slow consumer dropping older hits"
            )

    # ---- sessions ----
    # Pi is stateless wrt sessions — it just streams hits. The mobile app
    # owns the session lifecycle and persistence (locally for guests, to a
    # remote backend for logged-in users). The WS message includes a fixed
    # session_id="live" placeholder so the existing mobile schemas keep
    # accepting it; the mobile WS handler rewrites it to the locally active
    # session id before storing.

    def publish_hit(self, hit: Dict[str, Any]) -> None:
        """Broadcast a fresh hit to all WebSocket subscribers.

        ``hit`` should already contain x_norm, y_norm, score, ring, x_mm,
        y_mm, dist_mm, is_inner_ten. ``ts`` and a placeholder session_id
        are added here.
        """
        ts = time.time()
        message = {
            "type": "hit",
            "session_id": "live",
            "ts": ts,
            **hit,
        }
        t0 = time.time()
        self._broadcast(message)
        t_broadcast_ms = (time.time() - t0) * 1000.0
        n_subs = self.subscriber_count()
        print(
            f"[hit] ts={ts:.3f} score={hit.get('score')} ring={hit.get('ring')} "
            f"dist={hit.get('dist_mm', 0.0):.1f}mm "
            f"subs={n_subs} broadcast={t_broadcast_ms:.1f}ms"
        )


# --------------------------------------------------------------------------
# WebSocket helpers (RFC 6455, server-side text frames only)
# --------------------------------------------------------------------------

def _ws_accept(sec_key: str) -> str:
    digest = hashlib.sha1((sec_key + _WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def _ws_encode_text(payload: str) -> bytes:
    data = payload.encode("utf-8")
    n = len(data)
    header = bytes([0x81])  # FIN | opcode=text
    if n < 126:
        header += bytes([n])
    elif n < 65536:
        header += bytes([126]) + struct.pack(">H", n)
    else:
        header += bytes([127]) + struct.pack(">Q", n)
    return header + data


def _ws_encode_close() -> bytes:
    return bytes([0x88, 0x00])


def _ws_encode_pong(payload: bytes = b"") -> bytes:
    n = len(payload)
    if n < 126:
        return bytes([0x8A, n]) + payload
    raise ValueError("pong too large")


# --------------------------------------------------------------------------
# HTTP/WS request handler
# --------------------------------------------------------------------------

class _ControlHandler(BaseHTTPRequestHandler):
    state: ControlState
    version: str
    device_name: str
    device_id: str
    auth_token: str

    def log_message(self, format, *args):  # noqa: A002
        pass

    # ---- helpers ----
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _json(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception:
            return {}

    def _split_path(self) -> str:
        return self.path.split("?", 1)[0]

    # ---- routes ----
    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = self._split_path()
        if path == "/api/health":
            self._json(200, {
                "status": "ok",
                "version": self.version,
                "uptime_s": time.time() - self.state.start_time,
            })
            return
        if path == "/api/calibration/tweaks":
            getter = self.state.get_tweaks
            if getter is None:
                self._json(503, {"error": "tweaks_unavailable"})
                return
            self._json(200, getter())
            return
        if path == "/api/target/config":
            self._json(200, self.state.target_config)
            return
        if path == "/api/stream/preview.mjpeg":
            self._stream_mjpeg()
            return
        if path == "/api/stream/preview.jpg":
            self._serve_snapshot()
            return
        if path == "/ws/hits":
            self._handle_websocket()
            return
        self._json(404, {"error": "not_found", "path": path})

    def do_POST(self):  # noqa: N802
        path = self._split_path()
        if path == "/api/pair":
            self._read_json()
            self._json(200, {
                "token": self.auth_token,
                "device_name": self.device_name,
                "device_id": self.device_id,
            })
            return
        if path == "/api/calibration/freeze":
            self.state.request_freeze(True)
            self._json(200, {"ok": True, "frozen": True})
            return
        if path == "/api/calibration/unfreeze":
            self.state.request_freeze(False)
            self._json(200, {"ok": True, "frozen": False})
            return
        if path == "/api/calibration/tweaks":
            setter = self.state.set_tweaks
            getter = self.state.get_tweaks
            if setter is None or getter is None:
                self._json(503, {"error": "tweaks_unavailable"})
                return
            body = self._read_json()
            try:
                setter(body)
            except Exception as e:  # noqa: BLE001
                self._json(400, {"error": "invalid_tweaks", "detail": str(e)})
                return
            self._json(200, getter())
            return
        if path == "/api/calibration/auto":
            # "Auto adjust" = reset every operator tweak to its default and
            # let the bull-detection + ring-refinement run unmodified. After
            # this the rings should be exactly where the camera sees them.
            setter = self.state.set_tweaks
            getter = self.state.get_tweaks
            if setter is None or getter is None:
                self._json(503, {"error": "tweaks_unavailable"})
                return
            try:
                setter({
                    "scale_factor": 1.0,
                    "offset_x_mm": 0.0,
                    "offset_y_mm": 0.0,
                    "rotation_deg": 0.0,
                    "aspect_ratio": 1.0,
                    "keystone_h": 0.0,
                    "keystone_v": 0.0,
                    "keystone_d1": 0.0,
                    "keystone_d2": 0.0,
                    "paper_rotation_deg": 0.0,
                    "paper_scale": 1.0,
                })
                # Drop EMA so detection runs from a fresh frame, then collect
                # multiple fits and commit their median for olympic-grade
                # stability.
                self.state.request_rerun_detection()
                self.state.auto_adjust_blocking(samples=25, timeout_s=4.0)
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": "auto_failed", "detail": str(e)})
                return
            self._json(200, getter())
            return
        self._json(404, {"error": "not_found", "path": path})

    # ---- snapshot + MJPEG ----
    def _serve_snapshot(self) -> None:
        # Keep encoding warm — the calibration screen polls this every
        # 100 ms, so a "consumer just left" between polls would otherwise
        # cause a one-frame gap.
        self.state.preview_snapshot_polled()
        jpeg = self.state.get_latest_jpeg(timeout=2.0)
        if jpeg is None:
            self._json(503, {"error": "no_frame_yet"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(jpeg)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream_mjpeg(self) -> None:
        boundary = "frame"
        self.send_response(200)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={boundary}",
        )
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self._cors()
        self.end_headers()
        # Long-lived consumer — register so the cv2 loop knows to keep
        # encoding frames. Always pair with consumer_exit on disconnect.
        self.state.preview_consumer_enter()
        try:
            while True:
                jpeg = self.state.get_latest_jpeg(timeout=2.0)
                if jpeg is None:
                    continue
                head = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode("ascii")
                try:
                    self.wfile.write(head)
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
        except Exception as e:  # noqa: BLE001
            print(f"[control_server] mjpeg client dropped: {e}")
        finally:
            self.state.preview_consumer_exit()

    # ---- WebSocket ----
    def _handle_websocket(self) -> None:
        sec_key = self.headers.get("Sec-WebSocket-Key")
        upgrade = (self.headers.get("Upgrade") or "").lower()
        connection = (self.headers.get("Connection") or "").lower()
        if not sec_key or "websocket" not in upgrade or "upgrade" not in connection:
            self._json(400, {"error": "expected_websocket_upgrade"})
            return

        accept = _ws_accept(sec_key)
        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode("ascii")
        try:
            self.wfile.write(handshake)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

        sock = self.connection
        sock.settimeout(0.5)

        sub = self.state.add_subscriber()
        # One-shot calibration state on connect.
        try:
            sub.put_nowait(json.dumps({
                "type": "calibration",
                "state": "frozen" if self.state.is_frozen else "live",
            }))
        except queue.Full:
            pass

        print(f"[control_server] ws client connected ({self.state.subscriber_count()} total)")
        try:
            while True:
                # Drain queued outgoing messages.
                drained = False
                try:
                    while True:
                        msg = sub.get_nowait()
                        try:
                            sock.sendall(_ws_encode_text(msg))
                            drained = True
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            return
                except queue.Empty:
                    pass

                # Try a non-blocking peek for control frames.
                try:
                    data = sock.recv(2)
                except OSError:
                    if drained:
                        continue
                    continue
                if not data:
                    return
                if len(data) < 2:
                    continue
                b1, b2 = data[0], data[1]
                opcode = b1 & 0x0F
                masked = (b2 & 0x80) != 0
                payload_len = b2 & 0x7F
                if payload_len == 126:
                    ext = sock.recv(2)
                    if len(ext) < 2:
                        return
                    payload_len = struct.unpack(">H", ext)[0]
                elif payload_len == 127:
                    ext = sock.recv(8)
                    if len(ext) < 8:
                        return
                    payload_len = struct.unpack(">Q", ext)[0]
                mask_key = b""
                if masked:
                    mask_key = sock.recv(4)
                    if len(mask_key) < 4:
                        return
                payload = b""
                remaining = payload_len
                while remaining > 0:
                    chunk = sock.recv(min(4096, remaining))
                    if not chunk:
                        return
                    payload += chunk
                    remaining -= len(chunk)
                if masked and mask_key:
                    payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
                if opcode == 0x8:  # close
                    try:
                        sock.sendall(_ws_encode_close())
                    except Exception:
                        pass
                    return
                if opcode == 0x9:  # ping
                    try:
                        sock.sendall(_ws_encode_pong(payload[:125]))
                    except Exception:
                        return
                # Text frames from client (e.g. {"type":"ping"}) are ignored.
        finally:
            self.state.remove_subscriber(sub)
            print(f"[control_server] ws client disconnected ({self.state.subscriber_count()} total)")


# --------------------------------------------------------------------------
# Server bootstrap
# --------------------------------------------------------------------------

def start_control_server(
    state: ControlState,
    host: str = "0.0.0.0",
    port: int = 8080,
    version: str = "0.1.0",
    device_name: str = "ShooterRange",
    device_id: str = "shooterrange-pi",
    auth_token: str = "dev-token",
) -> ThreadingHTTPServer:
    handler_cls = type(
        "BoundControlHandler",
        (_ControlHandler,),
        {
            "state": state,
            "version": version,
            "device_name": device_name,
            "device_id": device_id,
            "auth_token": auth_token,
        },
    )
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    thread = threading.Thread(
        target=httpd.serve_forever,
        name="ControlServer",
        daemon=True,
    )
    thread.start()
    print(f"[control_server] listening on http://{host}:{port}")
    return httpd
