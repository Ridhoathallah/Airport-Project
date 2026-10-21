"""
=============================================================
  AIRPORT X-RAY QUEUE — CAMERA-ONLY PROTOTYPE
=============================================================

  Pipeline (camera-only, this phase):
    Camera → YOLO person detection → SORT tracking
          → Queue ROI → Occupancy → Counting line → ENTER/EXIT
          → Flow rate → Waiting time → Pixel spacing
          → Normalized occupancy → Rule-based congestion
          → Congestion stopwatch → Camera-only overlay

  Vision layer: YOLOv5s (COCO class 0 = person), full-person boxes
  tracked with SORT (head-crop mode optional for steep camera angles)
  Analytics layer: app/queue_prototype.py (occupancy, line crossings,
  flow, waiting, spacing, density, classifier, stopwatch, overlay).

  The queue-monitor engine (Sensor 1 / SENSOR_01 / XRAY01 / people-m²)
  is a SEPARATE legacy feature, disabled by default. Enable it only for
  the future sensor-fusion phase with --monitor.

  Usage:
    # Webcam (default: camera.source from config/camera.yaml, else index 0):
    python app/detect_people.py

    # Define the queue layout (ROI + counting line + X-ray side) with the webcam:
    python app/detect_people.py --define-roi

    # Preview a saved layout before analysis:
    python app/detect_people.py --roi-preview

    # Record events to JSONL for evaluation:
    python app/detect_people.py --record output/events.jsonl

    # Smartphone IP camera (e.g. IP Webcam / DroidCam):
    python app/detect_people.py --source http://<phone-ip>:8080/video

    # Video file / image:
    python app/detect_people.py --source input/your_video.mp4

  Output:
    - Window with ROI polygon, counting line (ENTER/EXIT arrows),
      tracked boxes + IDs, and the queue-metrics panel
    - Press 'q' to quit, 'r' to reload config/camera.yaml
"""

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

try:
    import cv2
    import numpy as np
    import torch
except ModuleNotFoundError as e:
    print(f"Missing dependency: {e.name}")
    print("The app must be run with the project virtualenv.")
    print("  source venv/bin/activate && python app/detect_people.py")
    print("  or simply:  ./run.sh")
    print("If the venv is missing, run:  python3 -m venv venv && \\")
    print("  venv/bin/python -m pip install -r requirements/requirements.txt")
    sys.exit(1)

from sort_tracker import SortTracker
from queue_prototype import (
    BACKLOG_UNAVAILABLE_MSG,
    QueueAnalytics,
    SystemMetrics,
    DashboardSeries,
    build_dashboard_model,
    define_layout_interactive,
    draw_dashboard,
    draw_overlay,
    _draw_video_layer,
    fmt_duration,
    iso_from_unix,
    load_camera_config,
    preview_layout,
)
from snapshot import SnapshotBuilder
from snapshot_holder import get_holder
from reload_signal import consume_config_reload
from shutdown import shutdown_requested

# ─────────────────────────────────────────────────────────────
#  LEGACY QUEUE-MONITOR ENGINE  (opt-in with --monitor)
# ─────────────────────────────────────────────────────────────
# The camera prototype is the default path. The core/ queue-monitoring
# engine (Sensor 1 emission, SENSOR_01, XRAY01, people/m²) is a separate
# future-phase feature and only activates when --monitor is passed.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    sys.path.insert(0, str(_PROJECT_ROOT))
    from core.models import SensorHealth, SensorReading, utc_now_iso
    from core.monitor import QueueMonitor
    _CORE_AVAILABLE = True
except Exception:
    QueueMonitor = SensorReading = None
    SensorHealth = utc_now_iso = None
    _CORE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit here — not scattered in the code)
# ─────────────────────────────────────────────────────────────

# Path to YOLOv5 repository (relative to this file's location)
YOLOV5_REPO = Path(__file__).resolve().parent.parent / "yolov5"

# Path to the model weights
MODEL_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "yolov5s.pt"

# Camera-only prototype: camera layout + classifier thresholds
DEFAULT_CAMERA_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "camera.yaml"

# Minimum confidence to show a detection (0.0 – 1.0)
CONFIDENCE_THRESHOLD = 0.50

# COCO class index for "person"
PERSON_CLASS_ID = 0

# Inference image size (pixels). 640 is standard for YOLOv5s; 480 keeps
# the pipeline above 24 FPS on Apple Silicon with negligible accuracy cost.
IMG_SIZE = 480

# Device: "cpu" | "mps" (Apple Silicon) | "cuda" (NVIDIA)
# "mps" gives a speed boost on Apple M-series chips.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Bounding box colour (BGR) and thickness
BOX_COLOR     = (0, 230, 118)   # green
BOX_THICKNESS = 2

# Label font settings
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.65
FONT_THICKNESS   = 2
LABEL_BG_COLOR   = (0, 230, 118)
LABEL_TEXT_COLOR = (0, 0, 0)

# Tracking (SORT) parameters
TRACK_MAX_AGE      = 30    # frames a track survives without a detection
TRACK_MIN_HITS     = 3     # matches needed before a new ID is reported
TRACK_IOU_THRESHOLD = 0.30 # min IoU to associate a detection with a track

# Head-only mode (opt-in via config/camera.yaml → camera.detect_mode:
# "head"): the top HEAD_RATIO of each person box is treated as the head
# (the body is ignored). The box is narrowed to the central
# HEAD_WIDTH_RATIO so it frames the head instead of the shoulders.
# 0.20–0.25 suits a frontal CCTV view. Default mode is "person" — full
# YOLO person boxes are fed straight to SORT.
HEAD_RATIO         = 0.22
HEAD_WIDTH_RATIO   = 0.60  # fallback head box width, as a fraction of the person box
HEAD_MIN_HEIGHT    = 10    # px — clamp to avoid degenerate head boxes

# Face refinement: after the head crop, run OpenCV's YuNet face
# detector (`cv2.FaceDetectorYN`). DISABLED by default in the camera-only
# prototype; enable with --face when wanted.
FACE_REFINEMENT     = False
FACE_MODEL_PATH     = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
FACE_SCORE_THRESHOLD = 0.40  # 0.0–1.0 — lower finds more (smaller) faces
FACE_INPUT_SIZE     = 320    # px — YuNet is run on this downscaled, aspect-correct canvas
FACE_CLOSEUP_SIZE   = 128    # px — min dimension of the upscaled head-crop close-up scan
FACE_MIN_SIZE       = 16     # px — ignore sub-face-sized boxes (full-resolution pixels)
MAX_FACE_CROPS      = 8      # per-frame cap on the close-up head-crop face scans

# Consistent frame pacing: the video loop is capped at TARGET_FPS so the
# output runs at a steady rate (24/30) instead of as fast as the hardware
# allows. 0 = unlimited.
TARGET_FPS          = 30.0

# Live capture (webcam / IP camera / FPV camera) is forced to this frame
# rate: the property is requested on the capture device and the reader
# thread paces to it, so the pipeline receives a steady stream.
CAPTURE_FPS         = 30.0

# Output window name
WINDOW_NAME = "Airport X-Ray Queue — Camera-Only Prototype"

# Live view is shown at this width (0 = native camera resolution). A
# slightly smaller window keeps the GUI fast enough to hold 24 FPS.
DISPLAY_WIDTH = 960

# Legacy queue-monitor engine (Sensor 1 / SENSOR_01 / XRAY01 / people-m²):
# DISABLED by default in the camera-only phase. Opt in with --monitor.
MONITOR_ENABLED = False
SENSOR_ID       = "SENSOR_01"   # must match config/default.yaml → sensors.s1.id
SENSOR_TYPE     = "vision"      # must match config/default.yaml → sensors.s1.type
ZONE_AREA_M2    = 50.0          # fallback monitored area if config is unavailable

# Show the raw people/px² value on the panel (clearly labelled uncalibrated).
SHOW_RAW_DENSITY = False

# ─────────────────────────────────────────────────────────────


def load_model():
    """Load the YOLOv5 model from the local repository."""
    print(f"\n[INFO] Loading YOLOv5s from: {MODEL_WEIGHTS}")
    print(f"[INFO] Device: {DEVICE.upper()}")

    # Add YOLOv5 to Python path so its modules can be imported
    if str(YOLOV5_REPO) not in sys.path:
        sys.path.insert(0, str(YOLOV5_REPO))

    model = torch.hub.load(
        str(YOLOV5_REPO),       # local repo path
        "custom",               # use custom weights
        path=str(MODEL_WEIGHTS),
        source="local",         # load from local, not from internet
        verbose=False,
    )

    model.conf    = CONFIDENCE_THRESHOLD   # confidence threshold
    model.classes = [PERSON_CLASS_ID]      # only detect persons
    model.to(DEVICE)

    print("[INFO] Model loaded ✅")
    print(f"[INFO] Confidence threshold : {CONFIDENCE_THRESHOLD}")
    print(f"[INFO] Detecting class      : person (COCO class {PERSON_CLASS_ID})")
    return model


def crop_head_boxes(det_array):
    """
    Reduce each person detection box to its head region.

    The head is taken as the top HEAD_RATIO of the person's box, narrowed
    to the central HEAD_WIDTH_RATIO so the fallback frames the head rather
    than the shoulders. The body below is dropped; confidence is unchanged.

    Parameters
    ----------
    det_array : np.ndarray
        (N, 5): (x1, y1, x2, y2, confidence) person detections.

    Returns
    -------
    np.ndarray
        (N, 5): (x1, y1, x2, y1 + height*HEAD_RATIO, confidence)
    """
    if len(det_array) == 0:
        return det_array

    head = det_array.copy()
    heights = head[:, 3] - head[:, 1]
    head_heights = np.maximum(heights * HEAD_RATIO, HEAD_MIN_HEIGHT)
    head[:, 3] = head[:, 1] + head_heights

    widths = head[:, 2] - head[:, 0]
    half_w = widths * HEAD_WIDTH_RATIO * 0.5
    center_x = (head[:, 0] + head[:, 2]) * 0.5
    head[:, 0] = center_x - half_w
    head[:, 2] = center_x + half_w
    return head


def refine_face_boxes(head_array, face_boxes):
    """
    Shrink each head-crop box to the nearest detected face.

    Faces are (full-resolution) boxes already gathered by
    `detect_faces` / the close-up scan. For every head crop, the nearest
    detected face is used; a face box starts at the forehead/brow line,
    so the hair above is excluded. Head crops with no matching face are
    kept unchanged (fallback).

    Parameters
    ----------
    head_array : np.ndarray  (N, 5) head boxes from crop_head_boxes
    face_boxes : np.ndarray | None
        (M, 6): (x1, y1, x2, y2, cx, cy) full-resolution face boxes.

    Returns
    -------
    tuple
        (refined, unmatched) — refined (N, 5) boxes and the indices of
        head crops that still have no face.
    """
    if face_boxes is None or len(face_boxes) == 0 or len(head_array) == 0:
        return head_array.copy(), list(range(len(head_array)))

    refined = head_array.copy()
    unmatched = []
    for i, head in enumerate(head_array):
        head_cx = (head[0] + head[2]) / 2.0
        head_cy = (head[1] + head[3]) / 2.0
        head_w = head[2] - head[0]
        head_h = head[3] - head[1]
        margin_x = head_w
        margin_y = head_h * 0.8

        candidates = [
            b for b in face_boxes
            if (head[0] - margin_x <= b[4] <= head[2] + margin_x)
            and (head[1] - margin_y <= b[5] <= head[3] + margin_y)
            and (b[3] - b[1]) >= FACE_MIN_SIZE
        ]
        if len(candidates) == 0:
            unmatched.append(i)
            continue

        best = min(
            candidates,
            key=lambda b: (b[4] - head_cx) ** 2 + (b[5] - head_cy) ** 2,
        )
        refined[i][0] = best[0]
        refined[i][1] = best[1]
        refined[i][2] = best[2]
        refined[i][3] = best[3]

    return refined, unmatched


def detect_faces(frame, face_detector):
    """
    Run YuNet over the whole frame on a small aspect-correct canvas.

    The full frame is letterboxed (not stretched) into a FACE_INPUT_SIZE
    square, detected once, and the face boxes are mapped back to
    full-resolution pixels. This is ~10x cheaper than scanning the
    native frame and catches the faces of the people in view.

    Parameters
    ----------
    frame         : np.ndarray  BGR frame
    face_detector : cv2.FaceDetectorYN

    Returns
    -------
    np.ndarray | None
        (M, 6): (x1, y1, x2, y2, cx, cy) in full-resolution pixels.
    """
    target = FACE_INPUT_SIZE
    fh, fw = frame.shape[:2]
    scale = max(fw, fh) / target
    nw = max(int(round(fw / scale)), 1)
    nh = max(int(round(fh / scale)), 1)
    pad_x = (target - nw) // 2
    pad_y = (target - nh) // 2

    canvas = np.full((target, target, 3), 114, np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(frame, (nw, nh))
    face_detector.setInputSize((target, target))
    _, faces = face_detector.detect(canvas)
    if faces is None or len(faces) == 0:
        return None

    boxes = []
    for f in faces:
        fx = (f[0] - pad_x) * scale
        fy = (f[1] - pad_y) * scale
        fw_ = f[2] * scale
        fh_ = f[3] * scale
        boxes.append((fx, fy, fx + fw_, fy + fh_, fx + fw_ / 2, fy + fh_ / 2))
    return np.array(boxes)


def refine_head_boxes(frame, face_detector, head_array):
    """
    Refine each head crop to the detected face using YuNet.

    A fast global scan (downscaled letterbox) covers the whole frame;
    any head crop still without a face gets a close-up scan of its own
    upscaled region so small / distant faces are recovered. Head crops
    with no face are kept as the narrowed head box (fallback).

    Parameters
    ----------
    frame         : np.ndarray  BGR frame
    face_detector : cv2.FaceDetectorYN
    head_array    : np.ndarray  (N, 5) head boxes from crop_head_boxes

    Returns
    -------
    np.ndarray
        (N, 5) boxes refined to faces where a face was found.
    """
    if len(head_array) == 0:
        return head_array

    refined, unmatched = refine_face_boxes(head_array, detect_faces(frame, face_detector))

    if len(unmatched) == 0:
        return refined

    fh, fw = frame.shape[:2]
    for i in unmatched[:MAX_FACE_CROPS]:
        x1 = max(0, int(head_array[i, 0]))
        y1 = max(0, int(head_array[i, 1]))
        x2 = min(fw, int(head_array[i, 2]))
        y2 = min(fh, int(head_array[i, 3]))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        margin = max(int((x2 - x1) * 0.5), 16)
        ox = max(0, x1 - margin)
        oy = max(0, y1 - margin)
        crop = frame[oy:min(fh, y2 + margin), ox:min(fw, x2 + margin)]
        if crop.size == 0:
            continue

        ch, cw = crop.shape[:2]
        scale = FACE_CLOSEUP_SIZE / max(cw, ch)
        nw = max(int(round(cw * scale)), 1)
        nh = max(int(round(ch * scale)), 1)
        small = cv2.resize(crop, (nw, nh))

        face_detector.setInputSize((nw, nh))
        _, faces = face_detector.detect(small)
        if faces is None or len(faces) == 0:
            continue

        best = faces[np.argmax(faces[:, -1])]
        bx = ox + best[0] / scale
        by = oy + best[1] / scale
        bw = best[2] / scale
        bh = best[3] / scale
        if bw >= FACE_MIN_SIZE and bh >= FACE_MIN_SIZE:
            refined[i][0] = bx
            refined[i][1] = by
            refined[i][2] = bx + bw
            refined[i][3] = by + bh

    return refined


class FrameStream:
    """
    Background video source that always hands out the latest frame.

    A dedicated reader thread keeps a webcam / IP-camera stream flowing
    while the main loop processes, so capture latency never blocks
    inference. Video files are read at their native rate. Each frame is
    tagged with a generation number so callers can skip stale frames.
    """

    def __init__(self, source, width=0, height=0):
        src = int(source) if str(source).isdigit() else source
        self._source = str(source)
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open source: {source}")

        self._is_webcam = str(source).isdigit()
        self._is_file = (
            not self._is_webcam
            and not self._source.startswith(("http://", "https://", "rtsp://"))
        )
        if self._is_webcam:
            if width > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._is_file:
            self._cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        self._file_fps = self._cap.get(cv2.CAP_PROP_FPS)

        # FPS the capture device reports after the request (live sources
        # only; video files keep their native rate). This is what we
        # *requested*, not necessarily what the camera delivers.
        self.requested_fps = CAPTURE_FPS if not self._is_file else 0.0
        # A video file's native frame rate is its own "reported" rate; a
        # live source reports whatever the device claims after the request.
        self.reported_fps = float(self._file_fps) if self._file_fps > 0 else 0.0
        # Kept for backward compatibility — use delivered_fps for actual rate.
        self.capture_fps = self.reported_fps
        # Measured frame rate the reader thread actually delivers.
        # Updated every ~2 s in the reader loop; 0 until measured. Measured
        # for *every* source type: a file that is read at its native rate
        # still delivers frames, and reporting 0 for it made the whole
        # dashboard claim the camera was offline while detection worked.
        self.delivered_fps = 0.0
        self._delivered = 0
        self._fps_t0 = time.monotonic()
        # Epoch seconds of the most recently delivered frame (0 = none yet).
        self._last_frame_unix = 0.0

        self._lock = threading.Lock()
        self._frame = None
        self._gen = 0
        self._ended = False
        self._running = False
        # Actual capture resolution, updated from the first delivered frame.
        self.frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        self.frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        # Fail fast when a live device opens but delivers no frames (macOS
        # camera privacy permission off, camera in use by another app, or a
        # dead IP stream). Video files deliver frames immediately, so they are
        # exempt from this check.
        if not self._is_file:
            first = self._grab_first_frame(self._cap, timeout=2.0)
            if first is None:
                self._cap.release()
                raise ValueError(
                    f"Camera source {source} opened but delivered no frames — "
                    "macOS camera privacy permission may be off (System Settings → "
                    "Privacy & Security → Camera → enable your terminal app, then "
                    "relaunch), another app may be using the camera, or the stream "
                    "is dead. Run --list-cameras to see available devices."
                )
            self.frame_w, self.frame_h = first.shape[1], first.shape[0]

        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    @staticmethod
    def _grab_first_frame(cap, timeout=2.0):
        """Try to read one usable frame so 'opened but no frames' fails fast."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ret, f = cap.read()
            if ret and f is not None:
                return f
            time.sleep(0.05)
        return None

    def _read_loop(self):
        if self._is_file:
            interval = 1.0 / self._file_fps if self._file_fps > 0 else (1.0 / CAPTURE_FPS if CAPTURE_FPS > 0 else 0.0)
        else:
            # Live sources (webcam / IP / FPV camera) are paced to a steady
            # CAPTURE_FPS so the pipeline gets a consistent 30 FPS stream.
            interval = 1.0 / CAPTURE_FPS if CAPTURE_FPS > 0 else 0.0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                # Flush the frame-rate measurement: a clip shorter than one
                # window would otherwise never report the rate it delivered.
                now = time.monotonic()
                elapsed = now - self._fps_t0
                if self._delivered > 0 and elapsed > 0:
                    self.delivered_fps = self._delivered / elapsed
                with self._lock:
                    self._ended = True
                break
            with self._lock:
                if not self.frame_w:
                    self.frame_w, self.frame_h = frame.shape[1], frame.shape[0]
                self._frame = frame
                self._gen += 1
                self._last_frame_unix = time.time()
            # Rolling delivered-frame-rate measurement, so the caller can
            # tell source-bound from compute-bound regardless of source type.
            # The window is short enough that the rate is known within the
            # first second -- a longer window left the camera reporting
            # "no signal" for the first seconds of every run.
            self._delivered += 1
            now = time.monotonic()
            if now - self._fps_t0 >= 1.0:
                self.delivered_fps = self._delivered / (now - self._fps_t0)
                self._delivered = 0
                self._fps_t0 = now
            if interval > 0:
                time.sleep(interval)

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread.start()
        return self

    def read(self):
        """Return (frame, generation); (None, -1) once the source has ended."""
        with self._lock:
            if self._ended:
                return None, -1
            return self._frame, self._gen

    @property
    def last_frame_unix(self) -> float:
        """Epoch seconds of the most recently delivered frame (0 = none yet).

        This is the real age of the newest frame, which is what the
        dashboard's LIVE / STALE / OFFLINE decision must be based on.
        """
        with self._lock:
            return self._last_frame_unix

    def release(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()


def open_source(source, width=0, height=0):
    """
    Open an input source: webcam index, video/image file, or a
    smartphone IP-camera stream URL (http/rtsp).

    ``width``/``height`` request a capture resolution (webcam only; the
    camera falls back to its nearest supported size, which is reported
    via ``FrameStream.frame_w/frame_h``).

    Returns (is_image, image, stream).
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    stream_prefixes = ("http://", "https://", "rtsp://")
    source_str = str(source)
    source_path = Path(source_str)

    if source_path.suffix.lower() in image_extensions and source_path.exists():
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        print(f"[INFO] Input source : IMAGE — {source_path.name}")
        return True, image, None

    try:
        stream = FrameStream(source, width=width, height=height)
    except ValueError as exc:
        raise ValueError(
            f"{exc}\n"
            "  Hint: run --list-cameras to see available camera devices, try a "
            "different --source index, and check macOS Camera privacy permission "
            "(System Settings → Privacy & Security → Camera)."
        ) from exc

    if source_str.isdigit():
        print(f"[INFO] Input source : WEBCAM (device {source})")
    elif source_str.startswith(stream_prefixes):
        print(f"[INFO] Input source : IP CAMERA STREAM — {source_str}")
    else:
        print(f"[INFO] Input source : VIDEO — {source_path.name}")

    return False, None, stream


def grab_frame(source, width=0, height=0):
    """Grab a single usable frame for layout definition / preview."""
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError(f"Could not open source: {source}")
    if str(source).isdigit():
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    for _ in range(20):
        ret, f = cap.read()
        if ret:
            frame = f
        time.sleep(0.03)
    cap.release()
    if frame is None:
        raise RuntimeError("Could not grab a frame from the source")
    return frame


class OccupancySampleRecorder:
    """Writes 1 Hz occupancy samples to JSONL for offline evaluation.

    Each row records the analytics snapshot's ``people_in_queue`` value
    tagged with a monotonic relative timestamp (``t_rel_s``, seconds since
    this recorder was created), so scripts/evaluate_counts.py can align
    manual ground truth against it. Recording only — this never touches
    detection, tracking or analytics behaviour.
    """

    SAMPLE_INTERVAL_S = 1.0

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        self._t0 = time.monotonic()
        self._next_due = self.SAMPLE_INTERVAL_S
        self.rows = 0

    def maybe_write(self, snapshot, source_frame=None, source_fps=None):
        """Write at most one sample per SAMPLE_INTERVAL_S; returns True if written."""
        t_rel = time.monotonic() - self._t0
        if t_rel < self._next_due:
            return False
        self._next_due = t_rel + self.SAMPLE_INTERVAL_S
        row = {
            "record_type": "occupancy_sample",
            "t_rel_s": round(t_rel, 3),
            "event_time": iso_from_unix(time.time()),
            "source_frame": int(source_frame) if source_frame is not None else None,
            "source_fps": round(float(source_fps), 3) if source_fps else None,
            "people_in_queue": int(snapshot["people_in_queue"]),
            "state": snapshot["state"],
            "roi_configured": bool(snapshot["config_status"]["roi_configured"]),
        }
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        self.rows += 1
        return True

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()


def run_define_layout(source, camera_config_path, width=0, height=0):
    """Interactive: draw ROI polygon, counting line, and X-ray side point."""
    print(f"[INFO] Interactive layout definition on source: {source}")
    print("[INFO] Draw the queue ROI (>= 3 clicks), then the counting line,")
    print("[INFO] then click the X-ray side point. Press 's' to save or Esc to cancel.")
    frame = grab_frame(source, width, height)
    saved = define_layout_interactive(frame, camera_config_path)
    if saved:
        cfg = load_camera_config(camera_config_path)
        errors = cfg.validate_roi(frame.shape) + cfg.validate_line(frame.shape)
        if errors:
            print("[WARN] Saved layout has validation issues:")
            for e in errors:
                print(f"       - {e}")
    else:
        print("[INFO] Layout unchanged.")


def run_preview_layout(source, camera_config, width=0, height=0):
    """Show the saved layout (ROI + line + direction) on a live frame."""
    print(f"[INFO] Layout preview on source: {source}")
    frame = grab_frame(source, width, height)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, preview_layout(frame, camera_config))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def match_track_confs(tracks, det_array):
    """Best detection confidence whose centre falls inside each track box."""
    confs = {}
    if len(det_array) == 0:
        return confs
    for t in tracks:
        x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
        best = 0.0
        for d in det_array:
            dx, dy = (d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0
            if x1 <= dx <= x2 and y1 <= dy <= y2 and d[4] > best:
                best = float(d[4])
        confs[int(t[4])] = best
    return confs


def run_detection(source=0, camera_config=None, monitor=None, detect_mode=None,
                  sensor_id=SENSOR_ID, sensor_type=SENSOR_TYPE, record_path=None,
                  debug=False, legacy_hud=False, samples_path=None, api_enabled=False,
                  headless=False):
    """Main camera-only prototype loop.

    ``monitor`` is the LEGACY ``core.monitor.QueueMonitor`` (opt-in via
    --monitor). The default path never emits SensorReading / people-m².

    ``detect_mode`` overrides ``camera.detect_mode`` from camera.yaml:
    "person" (full body boxes, default) or "head" (head-crop tracking).

    ``samples_path`` enables --record-samples: 1 Hz occupancy JSONL for
    offline evaluation (scripts/evaluate_counts.py).

    ``api_enabled`` when True, updates the thread-safe snapshot holder for
    the FastAPI server after each frame.

    ``headless`` when True, disables OpenCV desktop GUI windows for server / dashboard mode.
    """
    model = load_model()

    if camera_config is None:
        camera_config = load_camera_config(DEFAULT_CAMERA_CONFIG_PATH)

    if detect_mode:
        camera_config.detect_mode = detect_mode.lower()

    is_image, image, stream = open_source(source, camera_config.width, camera_config.height)

    if str(source).isdigit():
        source_label = "WEBCAM"
    elif str(source).startswith(("http://", "https://", "rtsp://")):
        source_label = "IP-CAM"
    else:
        source_label = str(Path(str(source)).name) or "VIDEO"

    if monitor is not None and _CORE_AVAILABLE:
        area = monitor.settings.zone.area_m2
        print(f"[INFO] (legacy) Queue-monitor engine active — sensor {sensor_id} → zone {monitor.settings.zone_id}")
        print(f"[INFO] (legacy) Monitored area {area:.1f} m² — density from tracked head count")

    # Actual capture resolution (native for the webcam, or the nearest
    # supported size when a specific resolution was requested).
    if is_image:
        cam_w, cam_h = image.shape[1], image.shape[0]
    else:
        assert stream is not None
        stream.start()
        frame, _ = stream.read()
        while frame is None:
            time.sleep(0.01)
            frame, _ = stream.read()
        cam_w, cam_h = (stream.frame_w or frame.shape[1]), (stream.frame_h or frame.shape[0])

    if camera_config.width <= 0:
        camera_config.width = cam_w
    if camera_config.height <= 0:
        camera_config.height = cam_h

    scale = camera_config.display_scale if camera_config.display_scale > 0 else 1.0
    disp_w = max(160, int(cam_w * scale))
    disp_h = max(120, int(cam_h * scale))
    # Native → display coordinate multiplier used by the HUD layer only.
    # Analytics (ROI, line, tracks) stay in native pixels.
    coord_scale = (disp_w / cam_w) if cam_w > 0 else 1.0

    print("\n[INFO] CAMERA-ONLY PROTOTYPE — airport X-ray queue analytics")
    print(f"[INFO] Camera       : {cam_w}×{cam_h}"
          + ("" if is_image else f"  (requested {camera_config.width}×{camera_config.height})"))
    if not is_image and stream is not None and stream.reported_fps > 0:
        print(f"[INFO] Capture FPS  : requested {stream.requested_fps:.0f}"
              f"  reported {stream.reported_fps:.1f}"
              f"  measured {stream.delivered_fps:.1f}" if stream.delivered_fps > 0 else "")
    elif not is_image and stream is not None:
        print(f"[INFO] Capture FPS  : measuring ...")
    print(f"[INFO] Display      : {disp_w}×{disp_h}  (display_scale {scale:.2f})")
    print(f"[INFO] Inference    : {IMG_SIZE}px  on {DEVICE}")
    print(f"[INFO] Detect mode  : {camera_config.detect_mode}")
    print(f"[INFO] ROI          : {'configured' if camera_config.roi_configured else 'NOT CONFIGURED — run --define-roi'}")
    print(f"[INFO] Counting line: {'configured' if camera_config.line_configured else 'NOT CONFIGURED'}")
    if camera_config.baseline_available:
        print(f"[INFO] Backlog baseline: {camera_config.baseline_people_per_minute:.2f} people/min")
    else:
        print(f"[INFO] Backlog check   : UNAVAILABLE ({BACKLOG_UNAVAILABLE_MSG})")
    print(f"[INFO] Calibration     : {'configured (real-world measurements enabled)' if camera_config.calibrated else 'NONE — pixel-relative only (no metres)'}")
    print(f"[INFO] Face refinement : {'enabled (YuNet)' if FACE_REFINEMENT else 'disabled'}")
    if record_path:
        print(f"[INFO] Event record    : {record_path}")
    print(f"[INFO] Runtime Mode : {'HEADLESS / LIVE (FastAPI stream only)' if headless else 'STANDALONE (OpenCV GUI window)'}")

    recorder = None
    if samples_path:
        if is_image:
            print("[WARN] --record-samples requires a video/stream source — ignoring.")
        else:
            try:
                recorder = OccupancySampleRecorder(samples_path)
            except OSError as exc:
                print(f"[ERROR] Cannot write occupancy samples to {samples_path}: {exc}")
                sys.exit(1)
            print(f"[INFO] Occupancy samples: {samples_path} (1 Hz)")

    analytics = QueueAnalytics(camera_config, record_path=record_path)
    series = DashboardSeries(camera_config.dashboard_chart_history)
    # Window creation only in standalone GUI mode.
    if not headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("\n[INFO] Starting detection. Press 'q' to quit, 'r' to reload config.\n")
    else:
        print("\n[INFO] Starting detection in HEADLESS / LIVE mode (FastAPI stream active).\n")

    if is_image:
        print("[INFO] Render      : legacy overlay (single-shot image)")
    else:
        print(f"[INFO] Render      : {'legacy HUD (--legacy-hud)' if legacy_hud else 'monitoring dashboard'}")
    print("-" * 45)

    fps          = 0.0
    frame_count  = 0
    fps_interval = 10
    t_start      = time.time()

    # Windowed per-component averages (ms/frame): inference, display/render,
    # and capture interval (from FrameStream.delivered_fps). Measured values.
    inf_ms = disp_ms = cap_ms = 0.0
    trk_ms = anl_ms = 0.0
    inf_sum = disp_sum = cap_sum = trk_sum = anl_sum = 0.0
    n_window = 0

    # ── IMAGE MODE ────────────────────────────────────────────
    if is_image:
        assert image is not None
        frame = image.copy()
        results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size=IMG_SIZE)
        dets = results.xyxy[0].cpu().numpy()
        dets = dets[dets[:, 5] == PERSON_CLASS_ID]
        det_array = dets[:, :5] if len(dets) > 0 else np.empty((0, 5))
        track_input = crop_head_boxes(det_array) if camera_config.detect_mode == "head" else det_array
        tracks = SortTracker(
            max_age=TRACK_MAX_AGE, min_hits=TRACK_MIN_HITS, iou_threshold=TRACK_IOU_THRESHOLD,
        ).update(track_input)
        snap = analytics.update(tracks, frame.shape[:2], track_confs=match_track_confs(tracks, det_array))
        display = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA) \
            if (disp_w, disp_h) != (frame.shape[1], frame.shape[0]) else frame
        display = draw_overlay(display, snap, tracks, camera_config, fps=0.0,
                               show_raw_density=SHOW_RAW_DENSITY,
                               track_confs=match_track_confs(tracks, det_array),
                               debug=debug, frame_index=None,
                               display_scale=coord_scale)
        print(f"PERSONS: {len(tracks)}   STATE: {snap['state']}   (image mode — no FPS)")
        print("-" * 45)
        cv2.imshow(WINDOW_NAME, display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── VIDEO / WEBCAM MODE ───────────────────────────────────
    assert stream is not None
    stream.start()

    tracker = SortTracker(
        max_age=TRACK_MAX_AGE,
        min_hits=TRACK_MIN_HITS,
        iou_threshold=TRACK_IOU_THRESHOLD,
    )

    face_detector = None
    if FACE_REFINEMENT:
        if not Path(FACE_MODEL_PATH).exists():
            print(f"[WARN] Face model not found ({FACE_MODEL_PATH.name}) — using plain head crop.")
            print("[WARN] Download from https://github.com/opencv/opencv_zoo into models/")
        else:
            print("[INFO] Face refinement enabled (YuNet)")
            face_detector = cv2.FaceDetectorYN_create(
                str(FACE_MODEL_PATH),
                "",
                (FACE_INPUT_SIZE, FACE_INPUT_SIZE),
                score_threshold=FACE_SCORE_THRESHOLD,
            )

    # Wait for the first frame so the loop starts with a real image.
    frame, last_gen = stream.read()
    while frame is None:
        time.sleep(0.01)
        frame, last_gen = stream.read()

    frame_budget = (1.0 / TARGET_FPS) if TARGET_FPS > 0 else 0.0
    next_frame_time = time.monotonic()
    recent_events = []

    while True:
        if shutdown_requested():
            print("\n[INFO] Shutdown requested -- stopping the detection loop.")
            break
        frame, gen = stream.read()
        if frame is None:
            print("\n[INFO] End of video stream.")
            break
        if gen == last_gen:
            # No new frame yet — keep the GUI event loop alive anyway so
            # the window keeps repainting while we wait for the camera.
            if shutdown_requested():
                print("\n[INFO] Shutdown requested -- stopping the detection loop.")
                break
            if not headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[INFO] Quit by user.")
                    break
            else:
                time.sleep(0.001)
            continue
        last_gen = gen

        frame_count += 1

        t_inf_start = time.time()
        results     = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size=IMG_SIZE)
        t_inf_end   = time.time()

        dets = results.xyxy[0].cpu().numpy()
        dets = dets[dets[:, 5] == PERSON_CLASS_ID]
        if len(dets) > 0:
            det_array = dets[:, :5]
        else:
            det_array = np.empty((0, 5))

        # Default: track full YOLO person boxes (crisp body boxes). In "head"
        # mode, track only the top-of-body head crop instead.
        if camera_config.detect_mode == "head":
            track_input = crop_head_boxes(det_array)
            if face_detector is not None and len(track_input) > 0:
                track_input = refine_head_boxes(frame, face_detector, track_input)
        else:
            track_input = det_array

        t_trk_start = time.time()
        tracks = tracker.update(track_input)
        track_confs = match_track_confs(tracks, det_array)
        t_trk_end = time.time()

        # ── Camera analytics (occupancy / line / flow / wait / classify) ──
        t_anl_start = time.time()
        snap = analytics.update(
            tracks, frame.shape[:2],
            track_confs=track_confs,
        )
        t_anl_end = time.time()
        recent_events.extend(snap["events_since_last"])
        if recorder is not None:
            recorder.maybe_write(snap, source_frame=gen,
                                 source_fps=stream.reported_fps or None)

        # ── LEGACY queue-monitor engine (opt-in --monitor) ──
        if monitor is not None and _CORE_AVAILABLE:
            ts = utc_now_iso()
            area = monitor.settings.zone.area_m2
            density_now = (len(tracks) / area) if area > 0 else 0.0
            monitor.ingest_sensor(SensorReading(
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                event_time=ts,
                server_received_time=ts,
                people_count=len(tracks),
                density=density_now,
                average_distance=0.0,
                flow_rate=0.0,
                confidence=1.0,
                health=SensorHealth.ONLINE,
            ))
            zone = monitor.snapshot()
            zone_state = zone.state.value
        else:
            zone_state = "-"

        if frame_count % fps_interval == 0:
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            if n_window > 0:
                inf_ms = inf_sum / n_window
                disp_ms = disp_sum / n_window
                cap_ms = cap_sum / n_window
                trk_ms = trk_sum / n_window
                anl_ms = anl_sum / n_window
                inf_sum = disp_sum = cap_sum = trk_sum = anl_sum = 0.0
                n_window = 0

        # ── Shared metrics model (console + dashboard use the SAME data) ──
        system = SystemMetrics(
            fps=fps, inference_ms=inf_ms, tracking_ms=trk_ms, analytics_ms=anl_ms,
            render_ms=disp_ms, capture_fps=stream.delivered_fps,
            frame_count=frame_count, detections=len(det_array), tracks=len(tracks),
            source=source_label, device=DEVICE, img_size=IMG_SIZE,
            last_frame_unix=stream.last_frame_unix,
        )
        dash = build_dashboard_model(snap, system, camera_config, recent_events)
        series.append(frame_count, dash.queue)

        # Visualization layer: resize to the display frame FIRST, then draw
        # the dashboard (or legacy HUD) directly on it (native→display
        # coordinates via coord_scale). Analytics above still run in native
        # pixels, so detection/tracking and queue metrics are unchanged.
        t_disp_start = time.time()
        display = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA) \
            if (disp_w, disp_h) != (frame.shape[1], frame.shape[0]) else frame
        if legacy_hud:
            display = draw_overlay(display, snap, tracks, camera_config,
                                   fps=fps, show_raw_density=SHOW_RAW_DENSITY,
                                   track_confs=track_confs, inf_ms=inf_ms,
                                   debug=debug, frame_index=frame_count,
                                   display_scale=coord_scale, disp_ms=disp_ms)
        else:
            display = draw_dashboard(display, dash, camera_config, tracks,
                                     track_confs=track_confs,
                                     frame_index=frame_count, series=series,
                                     display_scale=coord_scale)

        # Update snapshot holder for API server
        if api_enabled:
            # For the web dashboard camera stream, provide the clean camera video
            # with YOLO bounding boxes, track IDs, ROI polygon, and counting line,
            # NOT the standalone desktop OpenCV dashboard canvas (which contains
            # duplicated sidebar HUD and charts).
            video_frame_disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA) \
                if (disp_w, disp_h) != (frame.shape[1], frame.shape[0]) else frame
            stream_frame = _draw_video_layer(
                video_frame_disp, tracks, camera_config,
                track_confs=track_confs, display_scale=coord_scale,
                show_console_tip=False,
            )
            snap, _ = SnapshotBuilder.build(dash, system, camera_config, tracks, recent_events, stream_frame)
            # The raw display frame rides along so the mapping editor can
            # stream the camera WITHOUT burned-in annotations (?annotated=0).
            get_holder().update(snap, stream_frame, raw_frame=video_frame_disp)

            # Append to history buffer for dashboard trend charts
            from snapshot_buffer import get_history_buffer
            from dashboard_models import Sample
            now_ms = int(time.time() * 1000)
            get_history_buffer().append(Sample(
                t=now_ms,
                queue=dash.queue.people_in_queue,
                occupancy=dash.queue.occupancy_percent or 0.0,
                entry_rate=dash.queue.entry_flow,
                exit_rate=dash.queue.exit_flow,
                avg_wait=round(dash.queue.avg_wait_s / 60.0, 1),
            ))

        if not headless:
            cv2.imshow(WINDOW_NAME, display)
        t_disp_end = time.time()

        # Windowed per-component timings (measured, never faked).
        inf_sum += (t_inf_end - t_inf_start) * 1000
        disp_sum += (t_disp_end - t_disp_start) * 1000
        trk_sum += (t_trk_end - t_trk_start) * 1000
        anl_sum += (t_anl_end - t_anl_start) * 1000
        cap_sum += (1000.0 / stream.delivered_fps) if stream.delivered_fps > 0 else 0.0
        n_window += 1

        q = dash.queue
        cap_txt = f" cap: {cap_ms:5.1f}ms" if cap_ms > 0 else ""
        cong_txt = f" cong: {fmt_duration(q.congestion_duration_s)}" if q.congestion_active else ""
        print(
            f"\rDET: {q.people_detected:<3} IN-QUEUE: {q.people_in_queue:<3} "
            f"ENTR: {q.total_entries:<3} EXIT: {q.total_exits:<3} "
            f"FPS: {fps:5.1f} inf: {inf_ms:5.1f}ms trk: {trk_ms:4.1f}ms "
            f"disp: {disp_ms:5.1f}ms{cap_txt} "
            f"STATE: {q.state:<11} MOV: {q.movement:<6}{cong_txt} "
            f"REASONS: {','.join(q.reasons)}",
            end="", flush=True,
        )

        # Hot-reload requested by the API (e.g. PUT /api/dashboard/mapping).
        # Same path as the 'r' key below; the per-track crossing state is
        # reset so no stale side memory survives the geometry swap, and a
        # failed update never reaches here because the endpoint only
        # signals after a validated atomic persist.
        reload_source = consume_config_reload()
        if reload_source:
            camera_config = load_camera_config(camera_config.path or DEFAULT_CAMERA_CONFIG_PATH)
            analytics.config = camera_config
            analytics.reset_mapping()
            print(
                f"\n[INFO] Config reloaded via {reload_source} — "
                f"mapping active: {camera_config.mapping_active}, "
                f"ref: {getattr(getattr(camera_config, 'mapping', None), 'reference_point', 'n/a')}"
            )

        if not headless:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] Quit by user.")
                break
            if key == ord("r"):
                camera_config = load_camera_config(camera_config.path or DEFAULT_CAMERA_CONFIG_PATH)
                analytics.config = camera_config
                cfg_name = (camera_config.path or DEFAULT_CAMERA_CONFIG_PATH).name
                print(f"\n[INFO] Config reloaded ({cfg_name}) — ROI: "
                      f"{camera_config.roi_configured}, line: {camera_config.line_configured}")
        else:
            time.sleep(0.001)

        # Frame pacing: advance a wall-clock deadline by one frame budget
        # per iteration and sleep until it, so the stream runs at a
        # consistent rate with no cumulative drift. If we fall behind,
        # resync so future frames are paced correctly.
        if frame_budget > 0:
            next_frame_time += frame_budget
            delay = next_frame_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame_time = time.monotonic()

    stream.release()
    if recorder is not None:
        recorder.close()
        print(f"[INFO] Occupancy samples written: {recorder.rows} -> {recorder.path}")
    if not headless:
        cv2.destroyAllWindows()
    print("\n[INFO] Performance summary (measured, whole run):")
    print(f"[INFO]   FPS        : {fps:6.1f}")
    print(f"[INFO]   Inference  : {inf_ms:6.1f} ms/frame")
    print(f"[INFO]   Tracking   : {trk_ms:6.1f} ms/frame")
    print(f"[INFO]   Analytics  : {anl_ms:6.1f} ms/frame")
    print(f"[INFO]   Display    : {disp_ms:6.1f} ms/frame (draw + resize + imshow)")
    if stream.delivered_fps > 0:
        print(f"[INFO]   Source FPS : {stream.delivered_fps:6.1f} (measured)"
              f"  requested {stream.requested_fps:.0f}"
              f"  reported {stream.reported_fps:.1f}")
    print("[INFO] Detection stopped.")


# ─────────────────────────────────────────────────────────────
#  COMMAND-LINE INTERFACE
# ─────────────────────────────────────────────────────────────

def _probe_camera(idx, warmup_frames=20, warmup_delay=0.05):
    """Open camera at *idx*, read frames, return dict with results.

    Returns None if the camera cannot be opened or delivers no frames.
    The camera is always released before returning.
    """
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None

    # Try to read at least one frame.
    got = False
    shape = None
    for _ in range(warmup_frames):
        ok, f = cap.read()
        if ok and f is not None:
            got = True
            shape = (f.shape[1], f.shape[0])
            break
        time.sleep(warmup_delay)

    if not got:
        cap.release()
        return None

    # Measure actual FPS by reading frames for ~1 second.
    frame_count = 0
    t_start = time.monotonic()
    while time.monotonic() - t_start < 1.0:
        ok, _ = cap.read()
        if ok:
            frame_count += 1
        else:
            break
    elapsed = time.monotonic() - t_start
    measured_fps = frame_count / elapsed if elapsed > 0 else 0.0

    # Try to get the backend name.
    backend_id = cap.getBackendName() if hasattr(cap, "getBackendName") else ""

    # Read the FPS the device reports (after our CAP_PROP_FPS set).
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return {
        "index": idx,
        "opened": True,
        "frames": got,
        "width": shape[0],
        "height": shape[1],
        "reported_fps": reported_fps,
        "measured_fps": measured_fps,
        "backend": backend_id,
    }


def list_cameras():
    """Probe OpenCV video-capture indices and report what delivers frames.

    On macOS this also prints the devices the OS actually sees, so the user
    can match a device name (e.g. "1080P Web Camera") to an index.
    """
    # macOS device names from System Profiler.
    macos_names = {}
    if sys.platform == "darwin":
        try:
            import subprocess
            out = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                capture_output=True, text=True, timeout=30,
            ).stdout
            if out.strip():
                # Parse "Model:" lines to extract device names.
                for line in out.strip().splitlines():
                    line_s = line.strip()
                    if line_s.startswith("Model:"):
                        name = line_s.split(":", 1)[1].strip()
                        macos_names[name] = True
        except Exception:
            pass

    print("=" * 50)
    print("  AVAILABLE CAMERAS")
    print("=" * 50)
    print()

    available = []
    for idx in range(10):
        info = _probe_camera(idx)
        if info is None:
            continue
        available.append(info)

    if not available:
        print("  No cameras found.")
        print()
        print("  Troubleshooting:")
        print("  - Check macOS camera permission: System Settings →")
        print("    Privacy & Security → Camera → enable your terminal app")
        print("  - Close other apps that may be using the camera")
        print("  - Try: ./run.sh --source 1")
    else:
        # Determine recommended camera (first with width >= 640).
        recommended = None
        for info in available:
            if info["width"] >= 640:
                recommended = info["index"]
                break
        if recommended is None and available:
            recommended = available[0]["index"]

        for info in available:
            marker = " (recommended)" if info["index"] == recommended else ""
            backend_str = f"  backend: {info['backend']}" if info["backend"] else ""
            print(f"  [{info['index']}] {info['width']}x{info['height']}"
                  f"  fps: {info['measured_fps']:.1f}"
                  f"{backend_str}{marker}")

        print()
        print(f"  Recommended camera: [{recommended}]")
        print()
        print("  Commands:")
        print(f"    ./run.sh --source {recommended}          # use recommended camera")
        print(f"    ./run.sh --test-camera {recommended}     # test camera only")
        print(f"    ./run.sh --list-cameras     # show this list")

    print()


def test_camera(index=0):
    """Test a camera: open, read frames, measure FPS, report status.

    Does not load YOLO — isolates camera problems from CV problems.
    """
    print("=" * 50)
    print("  CAMERA TEST")
    print("=" * 50)
    print()
    print(f"  Index : {index}")
    print(f"  Source: OpenCV camera index {index}")
    print()

    info = _probe_camera(index, warmup_frames=30, warmup_delay=0.05)
    if info is None:
        print("  Status: FAIL")
        print()
        print("  Could not open camera or no frames received.")
        print()
        print("  Possible causes:")
        print("  - Camera index is incorrect (try --list-cameras)")
        print("  - Camera in use by another application")
        print("  - macOS camera permission not granted")
        print("    (System Settings → Privacy & Security → Camera)")
        print("  - USB webcam not properly connected")
        return False

    print(f"  Backend     : {info['backend'] or 'N/A'}")
    print(f"  Resolution  : {info['width']}x{info['height']}")
    print(f"  Reported FPS: {info['reported_fps']:.1f}")
    print(f"  Measured FPS: {info['measured_fps']:.1f}")
    print(f"  Frames OK   : YES")
    print()

    # Verify frames are actually changing (not frozen).
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        frames = []
        for _ in range(10):
            ok, f = cap.read()
            if ok and f is not None:
                frames.append(f)
            time.sleep(0.05)
        cap.release()

        if len(frames) >= 2:
            # Compare first and last frame to verify they differ.
            diff = np.mean(np.abs(frames[0].astype(float) - frames[-1].astype(float)))
            frozen = diff < 1.0
            print(f"  Frame change: {'NO (frozen!)' if frozen else 'YES'}")
            if frozen:
                print()
                print("  Status: FAIL — camera delivers frames but image is frozen.")
                print("  The camera may be in use by another application.")
                return False
        else:
            print("  Frame change: UNABLE TO TEST (too few frames)")

    print()
    print("  Status: PASS")
    print()
    print(f"  To use this camera: ./run.sh --source {index}")
    return True


def resolve_source(cli_source, camera_config):
    """Resolve the effective camera source with precedence: CLI → config → "0".

    cli_source is None when --source is not passed; the config value wins.
    Digit strings are converted to int (webcam index); URLs/paths pass through.
    """
    source = cli_source if cli_source is not None else (camera_config.source if camera_config else None)
    if source is None:
        source = "0"
    return int(source) if str(source).isdigit() else str(source)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Airport X-Ray Queue — Camera-Only Prototype (YOLO + SORT + queue analytics)"
    )
    parser.add_argument(
        "--source", default=None,
        help="Camera source: webcam index (e.g. 0), video file path, image "
             "path, or RTSP/HTTP URL.  Default: use camera.source from "
             "config/camera.yaml, else webcam index 0",
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="Probe OpenCV camera indices and list which ones deliver frames (also "
             "lists macOS camera devices), then exit",
    )
    parser.add_argument(
        "--test-camera", nargs="?", const="0", default=None, metavar="INDEX",
        help="Test a camera without loading YOLO: open, read frames, measure FPS. "
             "Optionally pass a camera index (default: 0). Then exit",
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--weights", default=str(MODEL_WEIGHTS),
        help="Path to model weights",
    )
    parser.add_argument(
        "--device", default=DEVICE,
        help="cpu | mps | cuda",
    )
    parser.add_argument(
        "--imgsz", type=int, default=IMG_SIZE,
        help=f"YOLO inference size (default: {IMG_SIZE})",
    )
    parser.add_argument(
        "--head-ratio", type=float, default=HEAD_RATIO,
        help=f"Top fraction of the person box treated as head (default: {HEAD_RATIO})",
    )
    parser.add_argument(
        "--detect-mode", default=None, choices=["person", "head"],
        help="Override camera.detect_mode: 'person' = full-body boxes (default), "
             "'head' = head-crop tracking for steep camera angles",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Overlay the classifier reasons on the video window",
    )
    parser.add_argument(
        "--legacy-hud", action="store_true",
        help="Use the legacy full-frame HUD instead of the monitoring dashboard",
    )
    parser.add_argument(
        "--no-face", action="store_true",
        help="Force face refinement OFF (face refinement is already off by default)",
    )
    parser.add_argument(
        "--face", action="store_true",
        help="Enable face refinement (YuNet) — opt-in, off by default",
    )
    parser.add_argument(
        "--fps", type=float, default=TARGET_FPS,
        help=f"Consistent target frame rate, 0 = unlimited (default: {TARGET_FPS})",
    )
    parser.add_argument(
        "--camera-config", default=None,
        help="Path to the camera-only prototype YAML config (default: config/camera.yaml)",
    )
    parser.add_argument(
        "--define-roi", action="store_true",
        help="Interactive: draw the queue ROI, counting line, and X-ray side point",
    )
    parser.add_argument(
        "--roi-preview", action="store_true",
        help="Preview the saved ROI + counting line before analysis",
    )
    parser.add_argument(
        "--record", default=None,
        help="Write every analytics event to a JSONL file (e.g. output/events.jsonl)",
    )
    parser.add_argument(
        "--record-samples", default=None,
        help="Record 1 Hz occupancy samples to a JSONL file for offline evaluation "
             "(e.g. output/samples.jsonl); see scripts/evaluate_counts.py",
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="LEGACY: enable the core/ queue-monitor engine (Sensor 1 / people-m²) — future phase only",
    )
    parser.add_argument(
        "--no-monitor", action="store_true",
        help="Force the legacy queue-monitor engine OFF (default is already off)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to the legacy queue-monitor engine YAML config (default: config/default.yaml)",
    )
    parser.add_argument(
        "--mode", default="standalone", choices=["standalone", "headless"],
        help="Runtime profile: 'standalone' (OpenCV GUI window) or 'headless' (FastAPI / Figma dashboard only)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without desktop OpenCV GUI window (equivalent to --mode headless)",
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Enable FastAPI server for dashboard integration (port 8080)",
    )
    parser.add_argument(
        "--api-host", default="0.0.0.0",
        help="API server bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--api-port", type=int, default=8080,
        help="API server port (default: 8080)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_cameras:
        list_cameras()
        sys.exit(0)

    if args.test_camera is not None:
        test_camera(int(args.test_camera))
        sys.exit(0)

    CONFIDENCE_THRESHOLD = args.conf
    MODEL_WEIGHTS        = Path(args.weights)
    DEVICE               = args.device
    IMG_SIZE             = args.imgsz
    HEAD_RATIO           = args.head_ratio
    FACE_REFINEMENT      = args.face and not args.no_face
    TARGET_FPS           = args.fps

    camera_config_path = Path(args.camera_config or DEFAULT_CAMERA_CONFIG_PATH)
    camera_config = load_camera_config(camera_config_path)

    if args.define_roi:
        print("=" * 50)
        print("  AIRPORT X-RAY QUEUE — CAMERA-ONLY PROTOTYPE")
        print("  Interactive layout definition")
        print("=" * 50)
        source = resolve_source(args.source, camera_config)
        run_define_layout(source, str(camera_config_path),
                          camera_config.width, camera_config.height)
        sys.exit(0)

    if args.roi_preview:
        source = resolve_source(args.source, camera_config)
        run_preview_layout(source, camera_config,
                           camera_config.width, camera_config.height)
        sys.exit(0)

    monitor = None
    if args.monitor and not args.no_monitor:
        if _CORE_AVAILABLE:
            monitor = QueueMonitor(config_path=args.config)
        else:
            print("[WARN] core/ engine not importable — running without legacy queue monitoring.")

    is_headless = args.headless or (args.mode == "headless")
    api_enabled = args.api or is_headless

    print("=" * 50)
    print("  AIRPORT X-RAY QUEUE — CAMERA-ONLY PROTOTYPE")
    print(f"  Runtime profile: {'HEADLESS / LIVE (Figma Dashboard)' if is_headless else 'STANDALONE (OpenCV GUI)'}")
    print("  YOLOv5s detection + SORT tracking + queue analytics")
    print("=" * 50)

    source = resolve_source(args.source, camera_config)

    # Start API server if requested or in headless mode
    api_thread = None
    if api_enabled:
        print(f"[INFO] Starting FastAPI server on {args.api_host}:{args.api_port}")
        try:
            from api_server import run_api_server
            api_thread = threading.Thread(
                target=run_api_server,
                args=(args.api_host, args.api_port),
                daemon=True,
            )
            api_thread.start()
            # Give server a moment to start
            time.sleep(1.0)
        except ImportError as e:
            print(f"[ERROR] FastAPI dependencies not installed: {e}")
            print("[ERROR] Install with: pip install fastapi uvicorn")
            sys.exit(1)

    run_detection(source=source, camera_config=camera_config, monitor=monitor,
                  detect_mode=args.detect_mode,
                  sensor_id=SENSOR_ID, sensor_type=SENSOR_TYPE, record_path=args.record,
                  debug=args.debug, legacy_hud=args.legacy_hud,
                  samples_path=args.record_samples,
                  api_enabled=api_enabled,
                  headless=is_headless)
