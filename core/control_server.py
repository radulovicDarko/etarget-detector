"""HTTP + WebSocket control server consumed by the React Native mobile app.

Endpoints:

  GET  /api/health                       -> {status, version, uptime_s}
  POST /api/pair                         -> {token, device_name, device_id}
  GET  /api/target/config                -> target geometry (mm)
  POST /api/session/start                -> {session_id, started_at}
  POST /api/session/{id}/end             -> end summary
  POST /api/session/{id}/reset           -> 204
  GET  /api/session/{id}                 -> full session
  GET  /api/sessions                     -> list
  GET  /api/stream/preview.mjpeg         -> multipart MJPEG (browser/VLC)
  GET  /api/stream/preview.jpg           -> single JPEG snapshot (mobile poll)
  POST /api/calibration/freeze           -> queue freeze (== keypress 'n')
  POST /api/calibration/unfreeze         -> queue unfreeze
  GET  /ws/hits                          -> WebSocket; pushes hit/reset/session_*
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
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit

from .session_store import SessionStore


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

        self._sessions_lock = threading.RLock()
        # Persistent SQLite-backed session storage. Survives Pi restarts.
        self.sessions = SessionStore()

        self._subs_lock = threading.Lock()
        self._subscribers: List["queue.Queue[str]"] = []

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
        for q in subs:
            try:
                q.put_nowait(text)
            except queue.Full:
                # Slow client — drop oldest by draining one.
                try:
                    q.get_nowait()
                    q.put_nowait(text)
                except Exception:
                    pass

    # ---- sessions ----
    def start_session(
        self,
        shooter_id: str,
        discipline: str,
        shots_per_target: Optional[int],
        targets_per_session: Optional[int],
    ) -> Dict[str, Any]:
        res = self.sessions.start_session(
            shooter_id=shooter_id,
            discipline=discipline,
            shots_per_target=shots_per_target,
            targets_per_session=targets_per_session,
        )
        self._broadcast({
            "type": "session_started",
            "session_id": res["session_id"],
            "started_at": res["started_at"],
        })
        return res

    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        summary = self.sessions.end_session(session_id)
        if summary is None:
            return None
        self._broadcast({"type": "session_ended", "summary": summary})
        return summary

    def reset_session(self, session_id: str) -> bool:
        if not self.sessions.reset_session(session_id):
            return False
        self._broadcast({"type": "reset"})
        return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get_session(session_id)

    def list_sessions(
        self,
        shooter_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self.sessions.list_sessions(shooter_id=shooter_id, limit=limit, offset=offset)

    def publish_hit(self, hit: Dict[str, Any]) -> None:
        """Append a hit to the active session and broadcast it.

        ``hit`` should already contain x_norm, y_norm, score, ring, x_mm,
        y_mm, dist_mm, is_inner_ten. ``ts`` and ``session_id`` are added here.
        """
        ts = time.time()
        session_id = self.sessions.get_active_id()
        if session_id is not None:
            try:
                self.sessions.append_hit(session_id, ts, hit)
            except Exception as e:  # noqa: BLE001
                print(f"[control_server] failed to persist hit: {e}")
        else:
            session_id = "pending"
        message = {"type": "hit", "session_id": session_id, "ts": ts, **hit}
        self._broadcast(message)


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
        if path == "/api/sessions":
            qs = parse_qs(urlsplit(self.path).query)
            shooter_id = qs.get("shooter_id", [None])[0]
            try:
                limit = max(1, min(500, int(qs.get("limit", [50])[0])))
            except (ValueError, TypeError):
                limit = 50
            try:
                offset = max(0, int(qs.get("offset", [0])[0]))
            except (ValueError, TypeError):
                offset = 0
            self._json(200, self.state.list_sessions(
                shooter_id=shooter_id, limit=limit, offset=offset,
            ))
            return
        if path.startswith("/api/session/"):
            rest = path[len("/api/session/"):]
            if rest and "/" not in rest:
                s = self.state.get_session(rest)
                if s is None:
                    self._json(404, {"error": "session_not_found", "id": rest})
                    return
                self._json(200, s)
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
        if path == "/api/session/start":
            body = self._read_json()
            res = self.state.start_session(
                shooter_id=str(body.get("shooter_id", "anonymous")),
                discipline=str(body.get("discipline", "unknown")),
                shots_per_target=body.get("shots_per_target"),
                targets_per_session=body.get("targets_per_session"),
            )
            self._json(200, res)
            return
        if path.startswith("/api/session/"):
            rest = path[len("/api/session/"):]
            if rest.endswith("/hit"):
                # Inject a synthetic hit into the active session. Used by the
                # mobile "Demo" discipline so a fake shot persists to
                # sessions.db and shows up in History exactly like a real
                # laser-detected shot. The server is the source of truth, so
                # it both stores AND broadcasts via WS — the same UI code
                # path that draws real hits also draws these.
                sid = rest[:-len("/hit")]
                body = self._read_json()
                # Validate against the same shape SessionStore.append_hit
                # expects. We coerce/clamp so a bad client can't poison the
                # database with NaNs or bogus rings.
                try:
                    ring = max(0, min(10, int(body.get("ring", 0))))
                    score = max(0, min(10, int(body.get("score", ring))))
                    hit = {
                        "x_norm": float(body.get("x_norm", 0.5)),
                        "y_norm": float(body.get("y_norm", 0.5)),
                        "score": score,
                        "ring": ring,
                        "x_mm": float(body.get("x_mm", 0.0)),
                        "y_mm": float(body.get("y_mm", 0.0)),
                        "dist_mm": float(max(0.0, body.get("dist_mm", 0.0))),
                        "is_inner_ten": bool(body.get("is_inner_ten", False)),
                    }
                except (TypeError, ValueError) as e:
                    self._json(400, {"error": "invalid_hit", "detail": str(e)})
                    return
                # The active session is whatever was started last — that's
                # the one the mobile app is showing. If nothing is active
                # the hit is broadcast with session_id="pending" but not
                # persisted (mirrors publish_hit's contract).
                active_id = self.state.sessions.get_active_id()
                if active_id is not None and active_id != sid:
                    self._json(409, {
                        "error": "session_not_active",
                        "active_id": active_id,
                        "requested_id": sid,
                    })
                    return
                self.state.publish_hit(hit)
                self._json(200, {"ok": True, "session_id": sid})
                return
            if rest.endswith("/end"):
                sid = rest[:-len("/end")]
                summary = self.state.end_session(sid)
                if summary is None:
                    self._json(404, {"error": "session_not_found", "id": sid})
                    return
                self._json(200, summary)
                return
            if rest.endswith("/reset"):
                sid = rest[:-len("/reset")]
                if not self.state.reset_session(sid):
                    self._json(404, {"error": "session_not_found", "id": sid})
                    return
                self.send_response(204)
                self._cors()
                self.end_headers()
                return
        self._json(404, {"error": "not_found", "path": path})

    # ---- snapshot + MJPEG ----
    def _serve_snapshot(self) -> None:
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
