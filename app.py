"""
Safe Drive AI - Personalised drowsiness / fatigue monitor (Streamlit version)

Run locally:
    streamlit run app.py

Deploy: Streamlit Community Cloud (Python 3.11), see requirements.txt and packages.txt.

What changed compared to the Windows script:
    - cv2.VideoCapture / cv2.imshow  -> browser webcam via streamlit-webrtc
    - winsound / pyttsx3             -> beep + voice played in the user's browser
    - keys q / r / f                 -> Start/Stop button, Recalibrate, False alarm buttons
    - session report graph           -> shown on the page after you press Stop
"""
import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_FILE = os.path.join(BASE_DIR, "user_profiles.json")
LOG_FILE = os.path.join(BASE_DIR, "session_log.csv")
MODEL_FILE = os.path.join(BASE_DIR, "drowsiness_model.pkl")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# ---------- Settings ----------
CALIB_SECONDS = 5
EAR_FACTOR = 0.75
MAR_MARGIN = 0.35
EYE_FRAMES = 20
YAWN_FRAMES = 15
YAWN_LIMIT = 2
YAWN_WINDOW_SEC = 600
YAWN_ALERT_HOLD = 8
WINDOW = 30
MODEL_THRESHOLD = 0.70

PERCLOS_WINDOW = 60
MICROSLEEP_SEC = 1.0
MILD_TH = 30
SEVERE_TH = 60
LEVEL_NAMES = ["NORMAL", "MILD", "SEVERE"]
LEVEL_COLORS = [(0, 200, 0), (0, 165, 255), (0, 0, 255)]  # BGR (drawn on video)

HEAD_DELTA = 0.10
HEAD_NOD_SEC = 0.3
HEAD_DOWN_LONG_SEC = 2.0
NOD_WINDOW = 120

HAND_YAWN_SEC = 1.0

DIP_START_HOUR = 14
DIP_END_HOUR = 15
DIP_SCORE_BONUS = 10

DEFAULT_TZ = "Asia/Kolkata"   # the server may run in UTC, so the dip window uses this zone

VOICE_MSG = {
    1: ("आप थके हुए लग रहे हैं, कृपया सतर्क रहें।", "You look tired. Please stay alert."),
    2: ("आप बहुत थक गए हैं। कृपया रुककर आराम करें।", "You are very tired. Please stop and take a rest."),
    3: ("दोपहर दो से तीन बजे नींद सबसे ज़्यादा आती है, कृपया सतर्क रहें।",
        "Between 2 and 3 PM sleepiness is highest. Please stay alert."),
}
VOICE_COOLDOWN = {1: 30, 2: 10, 3: 1800}


def get_tz(name):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return None


def now_local(tz):
    return datetime.now(tz) if tz else datetime.now()


def in_dip_window(tz=None):
    return DIP_START_HOUR <= now_local(tz).hour < DIP_END_HOUR


# =====================================================================
# Feature functions
# =====================================================================
def euclid(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def eye_aspect_ratio(eye):
    A = euclid(eye[1], eye[5])
    B = euclid(eye[2], eye[4])
    C = euclid(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(pts):
    A = euclid(pts[61], pts[67])
    B = euclid(pts[62], pts[66])
    C = euclid(pts[63], pts[65])
    D = euclid(pts[60], pts[64])
    return (A + B + C) / (3.0 * D)


def head_down_index(pts):
    """How far the nose tip sits below the eye line, relative to eye width."""
    p_l = np.array(pts[36], dtype=float)
    p_r = np.array(pts[45], dtype=float)
    nose = np.array(pts[30], dtype=float)
    e = p_r - p_l
    width = np.linalg.norm(e)
    if width < 1e-6:
        return 0.0
    down = np.array([-e[1], e[0]]) / width
    if e[0] < 0:
        down = -down
    eye_mid = (p_l + p_r) / 2.0
    return float(np.dot(nose - eye_mid, down) / width)


L_IRIS, L_C1, L_C2 = 468, 33, 133
R_IRIS, R_C1, R_C2 = 473, 362, 263


def gaze_vertical(lm, w, h):
    vals = []
    for iris, c1, c2 in ((L_IRIS, L_C1, L_C2), (R_IRIS, R_C1, R_C2)):
        ix, iy = lm[iris].x * w, lm[iris].y * h
        ax, ay = lm[c1].x * w, lm[c1].y * h
        bx, by = lm[c2].x * w, lm[c2].y * h
        width = max(np.hypot(ax - bx, ay - by), 1e-6)
        vals.append((iy - (ay + by) / 2) / width)
    return float(np.mean(vals))


class EyeGate:
    """Separates looking down from a real blink / eye closure."""
    DEEP_RATIO = 0.73
    HYST = 0.08
    GAZE_DOWN = 0.06
    HEAD_DOWN = 0.05

    def __init__(self):
        self.ear_buf = deque(maxlen=3)
        self.gaze_buf = deque(maxlen=5)
        self.closed = False
        self.looking_down = False

    def update(self, ear, ear_th, gaze_dev, head_dev):
        self.ear_buf.append(ear)
        med = float(np.median(self.ear_buf))
        deep_th = ear_th * self.DEEP_RATIO
        if ear > deep_th and gaze_dev is not None:
            self.gaze_buf.append(gaze_dev)
        g = float(np.median(self.gaze_buf)) if self.gaze_buf else 0.0
        self.looking_down = (g > self.GAZE_DOWN or
                             (head_dev is not None and head_dev > self.HEAD_DOWN))
        thr = deep_th if self.looking_down else ear_th
        if self.closed:
            self.closed = med < thr * (1 + self.HYST)
        else:
            self.closed = med < thr
        return self.closed


MESH_TO_68 = {
    36: 33, 37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
    42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
    30: 1,
    48: 61, 49: 40, 50: 37, 51: 0, 52: 267, 53: 270,
    54: 291, 55: 321, 56: 314, 57: 17, 58: 84, 59: 91,
    60: 78, 61: 82, 62: 13, 63: 312, 64: 308, 65: 317,
    66: 14, 67: 87,
}


class FaceLandmarker:
    """Face landmarks from MediaPipe Face Mesh (returns dlib-style 68 points)."""

    def __init__(self):
        import mediapipe as mp
        self.gaze_v = 0.0
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        pts = [(0, 0)] * 68
        for k, idx in MESH_TO_68.items():
            pts[k] = (int(lm[idx].x * w), int(lm[idx].y * h))
        self.gaze_v = gaze_vertical(lm, w, h)
        return pts

    def close(self):
        try:
            self.mesh.close()
        except Exception:
            pass


def ear_mar(pts):
    ear = (eye_aspect_ratio(pts[42:48]) + eye_aspect_ratio(pts[36:42])) / 2.0
    return ear, mouth_aspect_ratio(pts)


def window_features(ears, mars, base_ear, base_mar):
    e = np.asarray(ears) / base_ear
    m = np.asarray(mars) - base_mar
    return [
        e.mean(), e.min(), float((e < EAR_FACTOR).mean()),
        m.mean(), m.max(), float((m > MAR_MARGIN).mean()),
    ]


class HandMouthDetector:
    CHECK = (0, 4, 8, 9, 12, 16, 20)

    def __init__(self):
        self.hands = None
        self.frame_i = 0
        self.last_hit = False
        try:
            import mediapipe as mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2, model_complexity=0,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        except Exception:
            self.hands = None

    def near_mouth(self, frame, mouth):
        if self.hands is None:
            return False
        self.frame_i += 1
        if self.frame_i % 2:
            return self.last_hit
        h, w = frame.shape[:2]
        xs = [p[0] for p in mouth]
        ys = [p[1] for p in mouth]
        margin = 0.6 * (max(xs) - min(xs))
        x1, x2 = min(xs) - margin, max(xs) + margin
        y1, y2 = min(ys) - margin, max(ys) + margin
        res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hit = False
        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                for i in self.CHECK:
                    lm = hand.landmark[i]
                    if x1 <= lm.x * w <= x2 and y1 <= lm.y * h <= y2:
                        hit = True
                        break
                if hit:
                    break
        self.last_hit = hit
        return hit

    def close(self):
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass


class FatigueTracker:
    """Tracks PERCLOS, blink rate/duration, microsleeps and a 0-100 fatigue score."""

    def __init__(self):
        self.hist = deque()
        self.blinks = deque()
        self.closed_since = None
        self.micro_flagged = False
        self.microsleeps = 0
        self.score = 0.0

    def update(self, now, closed):
        self.hist.append((now, closed))
        while self.hist and now - self.hist[0][0] > PERCLOS_WINDOW:
            self.hist.popleft()
        if closed:
            if self.closed_since is None:
                self.closed_since = now
            if not self.micro_flagged and now - self.closed_since >= MICROSLEEP_SEC:
                self.micro_flagged = True
                self.microsleeps += 1
        elif self.closed_since is not None:
            duration = now - self.closed_since
            if duration >= 0.05:
                self.blinks.append((now, duration))
            self.closed_since = None
            self.micro_flagged = False
        while self.blinks and now - self.blinks[0][0] > PERCLOS_WINDOW:
            self.blinks.popleft()

    @property
    def perclos(self):
        if not self.hist:
            return 0.0
        return sum(1 for _, c in self.hist if c) / len(self.hist)

    @property
    def history_span(self):
        return self.hist[-1][0] - self.hist[0][0] if len(self.hist) > 1 else 0.0

    @property
    def blink_rate(self):
        span = max(self.history_span, 1.0)
        return len(self.blinks) * 60.0 / min(span, PERCLOS_WINDOW)

    @property
    def avg_blink(self):
        if not self.blinks:
            return 0.0
        return sum(d for _, d in self.blinks) / len(self.blinks)

    def compute_score(self, eye_frac, yawns, yawn_limit, nods, model_prob):
        warm = min(self.history_span / 10.0, 1.0)
        parts = [
            (40, min(self.perclos / 0.30, 1.0) * warm),
            (15, min(max(self.avg_blink - 0.15, 0.0) / 0.35, 1.0)),
            (20, min(yawns / float(yawn_limit), 1.0)),
            (10, eye_frac),
            (15, min(nods / 2.0, 1.0)),
        ]
        if model_prob is not None:
            parts.append((15, model_prob))
        total_w = sum(w for w, _ in parts)
        raw = sum(w * v for w, v in parts) / total_w * 100.0
        self.score = 0.9 * self.score + 0.1 * raw
        return self.score


# =====================================================================
# Storage: user profiles + event log
# =====================================================================
_file_lock = threading.Lock()


def load_profiles():
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_profiles(profiles):
    try:
        with _file_lock, open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
    except OSError:
        pass


def make_profile(old, base_ear, base_mar, base_head, ear_th, mar_th, settings):
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "base_ear": base_ear, "base_mar": base_mar, "base_head": base_head,
        "base_gaze": old.get("base_gaze"),
        "ear_thresh": ear_th, "mar_thresh": mar_th,
        "ear_factor": settings["ear_factor"],
        "mar_margin": settings["mar_margin"],
        "head_delta": settings["head_delta"],
        "mild_th": settings["mild_th"],
        "severe_th": settings["severe_th"],
        "created": old.get("created", now),
        "last_used": now,
        "sessions": old.get("sessions", 0),
        "total_yawns": old.get("total_yawns", 0),
        "total_alerts": old.get("total_alerts", 0),
        "total_nods": old.get("total_nods", 0),
        "false_alarms": old.get("false_alarms", 0),
        "last_session_avg_fatigue": old.get("last_session_avg_fatigue", 0),
        "last_session_max_fatigue": old.get("last_session_max_fatigue", 0),
        "hourly": old.get("hourly", {}),
    }


def log_event(user, event, ear, mar):
    try:
        with _file_lock:
            new_file = not os.path.exists(LOG_FILE)
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["timestamp", "user", "event", "ear", "mar"])
                w.writerow([datetime.now().isoformat(timespec="seconds"),
                            user, event, f"{ear:.3f}", f"{mar:.3f}"])
    except OSError:
        pass


def peak_hour(hourly, min_samples=60):
    best = None
    for h, (s, n) in hourly.items():
        if n >= min_samples:
            avg = s / n
            if best is None or avg > best[1]:
                best = (int(h), avg)
    return best


def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        import joblib
        return joblib.load(MODEL_FILE)
    except Exception:
        return None


def default_settings():
    return {"ear_factor": EAR_FACTOR, "mar_margin": MAR_MARGIN,
            "head_delta": HEAD_DELTA, "mild_th": MILD_TH, "severe_th": SEVERE_TH}


# =====================================================================
# Session report (graph)
# =====================================================================
def make_report(user, timeline, events, settings, hourly):
    if len(timeline) < 2:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORT_DIR, exist_ok=True)
    t = [x[0] / 60.0 for x in timeline]
    score = [x[1] for x in timeline]
    perclos = [x[2] * 100 for x in timeline]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    ax1.plot(t, score, color="tab:blue", linewidth=1.8, label="Fatigue score")
    ax1.axhline(settings["mild_th"], color="orange", linestyle="--", linewidth=1, label="Mild")
    ax1.axhline(settings["severe_th"], color="red", linestyle="--", linewidth=1, label="Severe")
    colors = {"YAWN": "orange", "MICROSLEEP": "black", "HEAD NOD": "purple",
              "SEVERE ALERT": "red", "FALSE ALARM": "green"}
    shown = set()
    for sec, name in events:
        ax1.axvline(sec / 60.0, color=colors.get(name, "gray"), alpha=0.5,
                    label=None if name in shown else name)
        shown.add(name)
    ax1.set_xlim(0, max(t))
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Fatigue score (0-100)")
    ax1.legend(loc="upper left", fontsize=8, ncol=4)
    ax1.grid(alpha=0.3)

    ax2.plot(t, perclos, color="tab:purple")
    ax2.set_xlim(0, max(t))
    ax2.set_ylabel("PERCLOS % (last 60 s)")
    ax2.set_xlabel("Time (minutes)")
    ax2.grid(alpha=0.3)

    hrs = sorted(int(h) for h, (s, n) in hourly.items() if n >= 10)
    if hrs:
        avgs = [hourly[str(h)][0] / hourly[str(h)][1] for h in hrs]
        cols = ["orange" if DIP_START_HOUR <= h < DIP_END_HOUR else "steelblue" for h in hrs]
        ax3.bar([f"{h}:00" for h in hrs], avgs, color=cols)
        ax3.set_ylim(0, 100)
        ax3.set_title("Your average fatigue by hour of day (orange = 2-3 PM dip window)",
                      fontsize=9)
    else:
        ax3.text(0.5, 0.5, "Not enough hourly data yet", ha="center", va="center")
    ax3.set_ylabel("Avg fatigue")
    ax3.grid(alpha=0.3, axis="y")

    n = lambda k: sum(1 for _, e in events if e == k)
    fig.suptitle(
        f"Session report - {user} | {timeline[-1][0] / 60:.1f} min | "
        f"avg {np.mean(score):.0f} / max {np.max(score):.0f} | "
        f"yawns {n('YAWN')} | nods {n('HEAD NOD')} | "
        f"severe alerts {n('SEVERE ALERT')} | microsleeps {n('MICROSLEEP')}",
        fontsize=10)
    fig.tight_layout()

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in user)
    path = os.path.join(REPORT_DIR, f"report_{safe}_{datetime.now():%Y%m%d_%H%M%S}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# =====================================================================
# Live processor: runs once per webcam frame (in a background thread)
# =====================================================================
class DrowsyProcessor(VideoProcessorBase):
    def __init__(self, user="guest", use_hands=True, tz_name=DEFAULT_TZ):
        self.user = user
        self.use_hands = use_hands
        self.tz = get_tz(tz_name)
        self.lock = threading.Lock()

        self.profiles = load_profiles()
        self.profile = self.profiles.get(user)
        self.model = load_model()
        self.landmarker = None
        self.hand_det = None
        self.init_failed = False
        self.error = None
        self.finalized = False

        # requests coming from the buttons
        self.req_recalib = False
        self.req_false = False

        self.settings = default_settings()
        self.base_ear = self.base_mar = self.base_head = self.base_gaze = None
        self.ear_th, self.mar_th = 0.25, 0.60
        self.calibrating = True
        self.welcome = None
        if self.profile:
            p = self.profile
            self.base_ear, self.base_mar = p["base_ear"], p["base_mar"]
            self.base_head = p.get("base_head")
            self.base_gaze = p.get("base_gaze")
            for k in self.settings:
                self.settings[k] = p.get(k, self.settings[k])
            self.ear_th = self.base_ear * self.settings["ear_factor"]
            self.mar_th = self.base_mar + self.settings["mar_margin"]
            self.calibrating = False
            pk = peak_hour(p.get("hourly", {}))
            self.welcome = f"Welcome back {user}! Saved calibration loaded."
            if pk:
                self.welcome += (f" Your fatigue usually peaks around {pk[0]}:00-{pk[0] + 1}:00 "
                                 f"(average score {pk[1]:.0f}).")

        self.ear_samples, self.mar_samples = [], []
        self.head_samples, self.gaze_samples = [], []
        self.calib_start = None
        self.gate = EyeGate()
        self.tracker = FatigueTracker()
        self.eye_counter = self.yawn_counter = 0
        self.yawn_hist, self.yawn_pending, self.nod_times = [], [], []
        self.yawn_alert_until = 0.0
        self.last_yawn_time = -999.0
        self.head_down_since = None
        self.nod_counted = False
        self.last_nod_time = -999.0
        self.hand_since = None
        self.hand_counted = False
        self.hand_msg_until = 0.0
        self.session_yawns = self.session_alerts = self.session_false = self.session_nods = 0
        self.eye_alert_on = self.model_alert_on = False
        self.prev_level = 0
        self.ear_buf = deque(maxlen=WINDOW)
        self.mar_buf = deque(maxlen=WINDOW)
        self.frame_no = 0
        self.model_prob = 0.0
        self.timeline, self.events = [], []
        self.session_hourly = {}
        self.t0 = None
        self.last_sample = 0.0
        self.msg_text, self.msg_until = "", 0.0

        # values shown on the page
        self.v = dict(level=0, score=0.0, perclos=0.0, blinks=0.0, yawns=0, nods=0,
                      ear=0.0, mar=0.0, face=False, looking_down=False, dip=False,
                      calibrating=self.calibrating, calib_left=CALIB_SECONDS,
                      alerts=[], model_prob=0.0)

    # ---------- controls (called from the page) ----------
    def request_recalibrate(self):
        self.req_recalib = True

    def request_false_alarm(self):
        self.req_false = True

    def snapshot(self):
        d = dict(self.v)
        d["error"] = self.error
        d["welcome"] = self.welcome
        d["message"] = self.msg_text if time.time() < self.msg_until else ""
        return d

    # ---------- internal helpers ----------
    def _start_calibration(self):
        self.calibrating = True
        self.calib_start = None
        self.ear_samples, self.mar_samples = [], []
        self.head_samples, self.gaze_samples = [], []
        self.base_head = None
        self.base_gaze = None
        self.gate = EyeGate()
        self.settings = default_settings()
        self.tracker = FatigueTracker()
        self.eye_counter = self.yawn_counter = 0
        self.yawn_hist, self.yawn_pending, self.nod_times = [], [], []
        self.yawn_alert_until = 0.0
        self.head_down_since = None
        self.nod_counted = False
        self.hand_since = None
        self.hand_counted = False
        self.model_prob = 0.0
        self.prev_level = 0
        self.ear_buf.clear()
        self.mar_buf.clear()

    def _false_alarm(self, now):
        s = self.settings
        s["ear_factor"] = max(0.55, s["ear_factor"] - 0.02)
        s["mar_margin"] = min(0.60, s["mar_margin"] + 0.03)
        s["head_delta"] = min(0.20, s["head_delta"] + 0.01)
        s["mild_th"] = min(45, s["mild_th"] + 3)
        s["severe_th"] = min(80, s["severe_th"] + 3)
        self.ear_th = self.base_ear * s["ear_factor"]
        self.mar_th = self.base_mar + s["mar_margin"]
        self.yawn_alert_until = 0.0
        self.session_false += 1
        self.events.append((now - self.t0, "FALSE ALARM"))
        log_event(self.user, "FALSE_ALARM_FEEDBACK", 0, 0)
        p = self.profiles.get(self.user, {})
        self.profiles[self.user] = make_profile(p, self.base_ear, self.base_mar, self.base_head,
                                                self.ear_th, self.mar_th, s)
        self.profiles[self.user]["base_gaze"] = self.base_gaze
        self.profiles[self.user]["false_alarms"] = p.get("false_alarms", 0) + 1
        save_profiles(self.profiles)
        self.msg_text = "Feedback saved: less sensitive now"
        self.msg_until = time.time() + 2.5

    @staticmethod
    def _put(frame, text, pos, scale, color, thick=2):
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    # ---------- per frame ----------
    def recv(self, av_frame):
        img = av_frame.to_ndarray(format="bgr24")
        try:
            img = self._process(img)
        except Exception as e:           # never crash the video stream
            self.error = f"{type(e).__name__}: {e}"
            self._put(img, "Processing error (see page)", (10, 30), 0.7, (0, 0, 255))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _process(self, frame):
        if self.init_failed:
            self._put(frame, "MediaPipe failed to start", (10, 30), 0.7, (0, 0, 255))
            return frame

        if self.landmarker is None:
            try:
                self.landmarker = FaceLandmarker()
                self.hand_det = HandMouthDetector() if self.use_hands else None
            except Exception as e:
                self.init_failed = True
                self.error = f"MediaPipe Face Mesh could not start: {e}"
                return frame

        now = time.time()
        if self.t0 is None:
            self.t0 = now
        if self.calibrating and self.calib_start is None:
            self.calib_start = now

        if self.req_recalib:
            self.req_recalib = False
            self._start_calibration()
            self.calib_start = now
        if self.req_false:
            self.req_false = False
            if not self.calibrating:
                self._false_alarm(now)

        self.frame_no += 1
        h = frame.shape[0]
        s = self.settings
        pts = self.landmarker.get(frame)
        alerts = []
        level = self.prev_level
        score = self.tracker.score

        if pts is None:
            self.v.update(face=False, calibrating=self.calibrating)
            self._put(frame, "No face detected", (10, 30), 0.7, (0, 165, 255))
        else:
            left_eye, right_eye, mouth = pts[42:48], pts[36:42], pts[48:68]
            ear, mar = ear_mar(pts)
            head_idx = head_down_index(pts)

            for eye in (left_eye, right_eye):
                for i in range(6):
                    cv2.line(frame, eye[i], eye[(i + 1) % 6], (0, 255, 0), 1)
            for (x, y) in mouth:
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

            self.v.update(face=True, ear=ear, mar=mar)

            if self.calibrating:
                self.ear_samples.append(ear)
                self.mar_samples.append(mar)
                self.head_samples.append(head_idx)
                self.gaze_samples.append(self.landmarker.gaze_v)
                elapsed = now - self.calib_start
                left = max(0.0, CALIB_SECONDS - elapsed)
                self._put(frame, "CALIBRATING: look straight, eyes open, mouth closed",
                          (10, 30), 0.6, (0, 165, 255))
                self._put(frame, f"{left:.1f}s left", (10, 60), 0.7, (0, 165, 255))
                self.v.update(calibrating=True, calib_left=left)

                if elapsed >= CALIB_SECONDS and len(self.ear_samples) > 10:
                    self.base_ear = float(np.median(self.ear_samples))
                    self.base_mar = float(np.median(self.mar_samples))
                    self.base_head = float(np.median(self.head_samples))
                    self.base_gaze = float(np.median(self.gaze_samples))
                    self.settings = s = default_settings()
                    self.ear_th = self.base_ear * s["ear_factor"]
                    self.mar_th = self.base_mar + s["mar_margin"]
                    self.calibrating = False
                    self.tracker = FatigueTracker()
                    self.gate = EyeGate()
                    self.profiles[self.user] = make_profile(
                        self.profiles.get(self.user, {}), self.base_ear, self.base_mar,
                        self.base_head, self.ear_th, self.mar_th, s)
                    self.profiles[self.user]["base_gaze"] = self.base_gaze
                    save_profiles(self.profiles)
                    self.msg_text = "Calibration saved"
                    self.msg_until = time.time() + 3
                    self.v.update(calibrating=False)
            else:
                self.v.update(calibrating=False)
                rule_alert = False

                # learn head / gaze baselines for old profiles
                if self.base_head is None:
                    self.head_samples.append(head_idx)
                    if len(self.head_samples) >= 60:
                        self.base_head = float(np.median(self.head_samples))
                        if self.user in self.profiles:
                            self.profiles[self.user]["base_head"] = self.base_head
                            save_profiles(self.profiles)
                if self.base_gaze is None:
                    self.gaze_samples.append(self.landmarker.gaze_v)
                    if len(self.gaze_samples) >= 60:
                        self.base_gaze = float(np.median(self.gaze_samples))
                        if self.user in self.profiles:
                            self.profiles[self.user]["base_gaze"] = self.base_gaze
                            save_profiles(self.profiles)

                # eye closed? (looking down is told apart from a blink)
                gaze_dev = (self.landmarker.gaze_v - self.base_gaze) if self.base_gaze is not None else None
                head_dev = (head_idx - self.base_head) if self.base_head is not None else None
                closed = self.gate.update(ear, self.ear_th, gaze_dev, head_dev)
                self.v["looking_down"] = self.gate.looking_down
                if self.gate.looking_down:
                    self._put(frame, "LOOKING DOWN", (10, 210), 0.7, (255, 200, 0))

                # PERCLOS / blink tracking
                prev_micro = self.tracker.microsleeps
                self.tracker.update(now, closed)
                if self.tracker.microsleeps > prev_micro:
                    self.events.append((now - self.t0, "MICROSLEEP"))
                    log_event(self.user, "MICROSLEEP", ear, mar)

                # eye closure
                if closed:
                    self.eye_counter += 1
                    if self.eye_counter >= EYE_FRAMES:
                        rule_alert = True
                        alerts.append("Eyes closed")
                        self._put(frame, "DROWSINESS ALERT!", (10, 30), 0.9, (0, 0, 255))
                        if not self.eye_alert_on:
                            self.eye_alert_on = True
                            log_event(self.user, "EYES_CLOSED_ALERT", ear, mar)
                else:
                    self.eye_counter = 0
                    self.eye_alert_on = False

                # yawn (open mouth)
                if mar > self.mar_th:
                    self.yawn_counter += 1
                    if self.yawn_counter == YAWN_FRAMES:
                        self.session_yawns += 1
                        self.last_yawn_time = now
                        self.yawn_hist.append(now)
                        self.yawn_pending.append(now)
                        self.events.append((now - self.t0, "YAWN"))
                        log_event(self.user, "YAWN", ear, mar)
                    if self.yawn_counter >= YAWN_FRAMES:
                        self._put(frame, "YAWNING!", (10, 60), 0.9, (0, 0, 255))
                else:
                    self.yawn_counter = 0

                # yawn hidden behind a hand
                if self.hand_det is not None and self.hand_det.near_mouth(frame, mouth):
                    if self.hand_since is None:
                        self.hand_since = now
                    if now - self.hand_since >= HAND_YAWN_SEC and not self.hand_counted:
                        self.hand_counted = True
                        if now - self.last_yawn_time > 5:
                            self.session_yawns += 1
                            self.last_yawn_time = now
                            self.yawn_hist.append(now)
                            self.yawn_pending.append(now)
                            self.hand_msg_until = now + 2.0
                            self.events.append((now - self.t0, "YAWN"))
                            log_event(self.user, "YAWN_HAND_COVERED", ear, mar)
                else:
                    self.hand_since = None
                    self.hand_counted = False

                self.yawn_hist = [t for t in self.yawn_hist if now - t <= YAWN_WINDOW_SEC]
                self.yawn_pending = [t for t in self.yawn_pending if now - t <= YAWN_WINDOW_SEC]
                if len(self.yawn_pending) >= YAWN_LIMIT:
                    self.yawn_alert_until = now + YAWN_ALERT_HOLD
                    self.yawn_pending.clear()
                    log_event(self.user, "FREQUENT_YAWNING", ear, mar)
                if now < self.yawn_alert_until:
                    rule_alert = True
                    alerts.append("Frequent yawning")
                    self._put(frame, "FREQUENT YAWNING", (10, 90), 0.8, (0, 0, 255))

                # head nod / head down
                head_long = False
                if self.base_head is not None:
                    if head_idx - self.base_head > s["head_delta"]:
                        if self.head_down_since is None:
                            self.head_down_since = now
                        dur = now - self.head_down_since
                        if dur >= HEAD_NOD_SEC and not self.nod_counted:
                            self.nod_counted = True
                            self.session_nods += 1
                            self.nod_times.append(now)
                            self.last_nod_time = now
                            self.events.append((now - self.t0, "HEAD NOD"))
                            log_event(self.user, "HEAD_NOD", ear, mar)
                        if dur >= HEAD_DOWN_LONG_SEC and ear < self.ear_th * 1.2:
                            head_long = True
                            rule_alert = True
                            alerts.append("Head down")
                    else:
                        self.head_down_since = None
                        self.nod_counted = False
                self.nod_times = [t for t in self.nod_times if now - t <= NOD_WINDOW]
                if head_long:
                    self._put(frame, "HEAD DOWN!", (10, 150), 0.8, (0, 0, 255))
                elif now - self.last_nod_time < 1.5:
                    self._put(frame, "HEAD NOD", (10, 150), 0.8, (0, 165, 255))
                if now < self.hand_msg_until:
                    self._put(frame, "HAND-COVERED YAWN", (10, 180), 0.8, (0, 165, 255))

                # ML model on the last WINDOW frames (optional)
                self.ear_buf.append(ear)
                self.mar_buf.append(mar)
                if (self.model is not None and len(self.ear_buf) == WINDOW
                        and self.frame_no % 5 == 0):
                    feats = window_features(self.ear_buf, self.mar_buf, self.base_ear, self.base_mar)
                    self.model_prob = float(self.model.predict_proba([feats])[0][1])
                if self.model is not None:
                    if self.model_prob > MODEL_THRESHOLD:
                        rule_alert = True
                        alerts.append("Model: drowsy")
                        self._put(frame, "MODEL: DROWSY", (10, 120), 0.8, (0, 0, 255))
                        if not self.model_alert_on:
                            self.model_alert_on = True
                            log_event(self.user, "MODEL_ALERT", ear, mar)
                    else:
                        self.model_alert_on = False

                # fatigue score (+ time-of-day bonus) and level
                raw_score = self.tracker.compute_score(
                    min(self.eye_counter / float(EYE_FRAMES), 1.0),
                    len(self.yawn_hist), YAWN_LIMIT, len(self.nod_times),
                    self.model_prob if self.model is not None else None)
                dip = in_dip_window(self.tz)
                score = min(100.0, raw_score + (DIP_SCORE_BONUS if dip else 0))

                if rule_alert or score >= s["severe_th"]:
                    level = 2
                elif score >= s["mild_th"]:
                    level = 1
                else:
                    level = 0

                if level == 2 and self.prev_level != 2:
                    self.session_alerts += 1
                    self.events.append((now - self.t0, "SEVERE ALERT"))
                    log_event(self.user, "SEVERE_ALERT", ear, mar)
                self.prev_level = level

                if now - self.last_sample >= 1.0:
                    self.last_sample = now
                    self.timeline.append((now - self.t0, score, self.tracker.perclos, level))
                    hh = now_local(self.tz).hour
                    acc = self.session_hourly.setdefault(hh, [0.0, 0])
                    acc[0] += raw_score
                    acc[1] += 1

                self.v.update(level=level, score=score, perclos=self.tracker.perclos,
                              blinks=self.tracker.blink_rate, yawns=self.session_yawns,
                              nods=len(self.nod_times), dip=dip, alerts=alerts,
                              model_prob=self.model_prob)

                # on-screen info
                info = [
                    f"EAR: {ear:.2f} (th {self.ear_th:.2f})",
                    f"MAR: {mar:.2f} (th {self.mar_th:.2f})",
                    f"Yawns: {self.session_yawns}",
                    f"PERCLOS: {self.tracker.perclos * 100:.0f}%",
                    f"Blinks/min: {self.tracker.blink_rate:.0f}",
                    f"Nods (2 min): {len(self.nod_times)}",
                ]
                if self.model is not None:
                    info.append(f"Model: {self.model_prob:.2f}")
                for i, line in enumerate(info):
                    self._put(frame, line, (330, 30 + 25 * i), 0.6, (255, 0, 0))
                if dip:
                    self._put(frame, "2-3 PM dip: extra sensitive",
                              (330, 30 + 25 * len(info)), 0.55, (0, 140, 255))

                color = LEVEL_COLORS[level]
                bar_y = h - 45
                self._put(frame, f"Fatigue: {score:.0f}/100  {LEVEL_NAMES[level]}",
                          (10, bar_y - 8), 0.6, color)
                cv2.rectangle(frame, (10, bar_y), (210, bar_y + 12), (200, 200, 200), 1)
                cv2.rectangle(frame, (10, bar_y),
                              (10 + int(2 * min(score, 100)), bar_y + 12), color, -1)

        if time.time() < self.msg_until:
            self._put(frame, self.msg_text, (10, h - 75), 0.6, (0, 255, 0))
        self._put(frame, f"User: {self.user}", (10, h - 10), 0.6, (255, 255, 255))
        return frame

    # ---------- called once when the stream stops ----------
    def finalize(self):
        if self.finalized:
            return getattr(self, "result", None)
        self.finalized = True
        for obj in (self.hand_det, self.landmarker):
            if obj is not None:
                obj.close()

        scores = [x[1] for x in self.timeline]
        avg_f = float(np.mean(scores)) if scores else 0.0
        max_f = float(np.max(scores)) if scores else 0.0
        hourly_all, peak = {}, None
        if self.user in self.profiles:
            p = self.profiles[self.user]
            p["last_used"] = datetime.now().isoformat(timespec="seconds")
            p["sessions"] = p.get("sessions", 0) + 1
            p["total_yawns"] = p.get("total_yawns", 0) + self.session_yawns
            p["total_alerts"] = p.get("total_alerts", 0) + self.session_alerts
            p["total_nods"] = p.get("total_nods", 0) + self.session_nods
            p["last_session_avg_fatigue"] = round(avg_f, 1)
            p["last_session_max_fatigue"] = round(max_f, 1)
            hourly = p.setdefault("hourly", {})
            for hh, (sm, n) in self.session_hourly.items():
                old = hourly.get(str(hh), [0.0, 0])
                hourly[str(hh)] = [old[0] + sm, old[1] + n]
            hourly_all = hourly
            save_profiles(self.profiles)
            peak = peak_hour(hourly)

        report = None
        try:
            report = make_report(self.user, self.timeline, self.events,
                                 self.settings, hourly_all)
        except Exception as e:
            self.error = f"Report error: {e}"

        self.result = dict(
            user=self.user, avg=avg_f, max=max_f, yawns=self.session_yawns,
            nods=self.session_nods, alerts=self.session_alerts,
            false_alarms=self.session_false, peak=peak, report=report,
            sessions=self.profiles.get(self.user, {}).get("sessions", 0))
        return self.result


# =====================================================================
# Browser alerts (beep + speech), played in the user's browser
# =====================================================================
def browser_alert(ph, n, beep, speak_level):
    hi, en = ("", "")
    if speak_level:
        hi, en = VOICE_MSG[speak_level]
    html = f"""
    <script>
    // alert #{n}
    (function() {{
      const beep = {json.dumps(bool(beep))};
      const hi = {json.dumps(hi)};
      const en = {json.dumps(en)};
      try {{
        if (beep) {{
          const ac = new (window.AudioContext || window.webkitAudioContext)();
          const o = ac.createOscillator(), g = ac.createGain();
          o.type = "square"; o.frequency.value = 2500; g.gain.value = 0.15;
          o.connect(g); g.connect(ac.destination);
          o.start(); o.stop(ac.currentTime + 0.3);
        }}
      }} catch (e) {{}}
      try {{
        if ((hi || en) && "speechSynthesis" in window) {{
          const voices = window.speechSynthesis.getVoices();
          const hasHi = voices.length === 0 ||
                        voices.some(v => v.lang.toLowerCase().startsWith("hi"));
          const u = new SpeechSynthesisUtterance(hasHi ? hi : en);
          u.lang = hasHi ? "hi-IN" : "en-US";
          u.rate = 0.95;
          window.speechSynthesis.speak(u);
        }}
      }} catch (e) {{}}
    }})();
    </script>
    """
    with ph:
        components.html(html, height=0)


# =====================================================================
# Streamlit page
# =====================================================================
st.set_page_config(page_title="Safe Drive AI", page_icon="🚗", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #060b17, #0c1530, #101b3d, #0a2a33, #060b17);
    background-size: 400% 400%;
    animation: bgmove 22s ease infinite;
    background-attachment: fixed;
}
@keyframes bgmove { 0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;} }
.stApp::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
        radial-gradient(600px 400px at 8% 12%, rgba(0,212,170,0.16), transparent 70%),
        radial-gradient(700px 450px at 92% 8%, rgba(139,92,246,0.18), transparent 70%),
        radial-gradient(600px 400px at 70% 100%, rgba(59,130,246,0.14), transparent 70%);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1250px; position: relative; z-index: 1; }
footer { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(12,19,40,0.95), rgba(8,14,30,0.95));
    border-right: 1px solid rgba(255,255,255,0.07);
}
.brand { display:flex; align-items:center; gap:10px; margin: 4px 0 10px 0; }
.brand .logo { width:38px; height:38px; border-radius:12px; display:grid; place-items:center; font-size:20px;
    background: linear-gradient(135deg,#00D4AA,#6366F1); box-shadow: 0 6px 18px rgba(0,212,170,0.35); }
.brand b { font-size:1.05rem; color:#F1F5F9; } .brand small { display:block; color:#8EA0BE; font-size:0.72rem; }

/* ---------- hero ---------- */
.hero {
    position: relative; overflow: hidden; padding: 28px 32px; border-radius: 24px; margin-bottom: 18px;
    background: linear-gradient(120deg, rgba(0,212,170,0.18), rgba(99,102,241,0.22) 60%, rgba(139,92,246,0.18));
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 18px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.12);
}
.hero::after { content:"🚗"; position:absolute; right:28px; top:6px; font-size:96px; opacity:0.13; transform: rotate(-8deg); }
.hero h1 {
    margin:0; font-size:2.5rem; font-weight:800; letter-spacing:0.2px;
    background: linear-gradient(90deg,#FFFFFF,#7DF9E0 55%,#A5B4FC); -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color: transparent;
}
.hero p { margin:8px 0 0 0; color:#B4C2DA; font-size:1rem; max-width:720px; }
.chip {
    display:inline-block; margin:14px 8px 0 0; padding:5px 14px; border-radius:999px; font-size:0.78rem; font-weight:500;
    color:#D8FFF6; background: rgba(0,212,170,0.13); border:1px solid rgba(0,212,170,0.38);
}

/* ---------- how it works ---------- */
.steps { display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; margin-bottom:18px; }
.step {
    padding:16px 18px; border-radius:18px; background: rgba(255,255,255,0.045);
    border:1px solid rgba(255,255,255,0.09); backdrop-filter: blur(8px);
    transition: transform .2s ease, border-color .2s ease;
}
.step:hover { transform: translateY(-3px); border-color: rgba(0,212,170,0.55); }
.step .n { width:28px; height:28px; border-radius:50%; display:inline-grid; place-items:center; font-weight:700; font-size:0.85rem;
    color:#04121c; background: linear-gradient(135deg,#00D4AA,#7DF9E0); margin-right:8px; }
.step b { color:#F1F5F9; } .step p { margin:8px 0 0 0; color:#9FB0C9; font-size:0.86rem; }

/* ---------- status ---------- */
.status {
    display:flex; align-items:center; gap:12px; padding:15px 20px; border-radius:18px; margin:12px 0 10px 0;
    font-size:1.05rem; font-weight:600; color:#fff; border:1px solid rgba(255,255,255,0.14);
    box-shadow: 0 8px 28px rgba(0,0,0,0.3);
}
.status.ok   { background: linear-gradient(90deg, rgba(16,185,129,0.42), rgba(16,185,129,0.10)); }
.status.warn { background: linear-gradient(90deg, rgba(245,158,11,0.45), rgba(245,158,11,0.10)); }
.status.info { background: linear-gradient(90deg, rgba(59,130,246,0.45), rgba(59,130,246,0.10)); }
.status.bad  { background: linear-gradient(90deg, rgba(239,68,68,0.55), rgba(239,68,68,0.12)); animation: pulse 1s infinite; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.55);} 50% { box-shadow: 0 0 0 14px rgba(239,68,68,0);} }

/* ---------- gauge ---------- */
.gauge-card {
    padding:20px; border-radius:22px; text-align:center; margin-bottom:14px;
    background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.10);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35); backdrop-filter: blur(10px);
}
.gauge-title { color:#8EA0BE; font-size:0.78rem; letter-spacing:2px; font-weight:600; margin-bottom:12px; }
.ring { width:190px; height:190px; margin:0 auto; border-radius:50%; display:grid; place-items:center;
    box-shadow: 0 0 40px var(--glow); transition: background .4s ease; }
.ring-inner { width:150px; height:150px; border-radius:50%; display:flex; flex-direction:column; align-items:center; justify-content:center;
    background: radial-gradient(circle at 30% 25%, #14203f, #0a1226); border:1px solid rgba(255,255,255,0.08); }
.ring-num { font-size:3rem; font-weight:800; color:#F8FAFC; line-height:1; }
.ring-sub { font-size:0.85rem; font-weight:700; letter-spacing:2px; margin-top:6px; }

/* ---------- stat cards ---------- */
.stats { display:grid; grid-template-columns: repeat(2, 1fr); gap:12px; }
.stat {
    padding:13px 15px; border-radius:16px; background: rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.09); backdrop-filter: blur(8px);
    transition: transform .2s ease, border-color .2s ease;
}
.stat:hover { transform: translateY(-2px); border-color: rgba(125,249,224,0.5); }
.stat .ic { font-size:1.25rem; } .stat .lb { color:#8EA0BE; font-size:0.74rem; margin-top:2px; }
.stat .vl { color:#F8FAFC; font-size:1.45rem; font-weight:700; line-height:1.2; }

/* ---------- streamlit widgets ---------- */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.09);
    border-radius:16px; padding:14px 16px; box-shadow: 0 6px 22px rgba(0,0,0,0.25);
}
[data-testid="stMetricLabel"] p { color:#9FB0C9; }
[data-testid="stMetricValue"] { color:#F8FAFC; font-weight:700; }
video, [data-testid="stVideo"] { border-radius:20px; box-shadow: 0 14px 40px rgba(0,0,0,0.45); }
div[data-testid="stImage"] img { border-radius:16px; }
.stButton > button, .stDownloadButton > button {
    border-radius:12px; border:1px solid rgba(255,255,255,0.15); font-weight:600; color:#E6EDF7;
    background: linear-gradient(135deg, rgba(0,212,170,0.18), rgba(99,102,241,0.18));
    transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color:#00D4AA; color:#7DF9E0; transform: translateY(-1px); box-shadow: 0 8px 20px rgba(0,212,170,0.25);
}
/* ---------- landing page ---------- */
.land-hero { text-align:center; padding: 46px 20px 26px 20px; position:relative; }
.orb {
    width:130px; height:130px; margin:0 auto 18px auto; border-radius:50%; display:grid; place-items:center;
    background: radial-gradient(circle at 30% 25%, rgba(125,249,224,0.55), rgba(99,102,241,0.35) 60%, rgba(10,18,38,0.2));
    border:1px solid rgba(255,255,255,0.18); font-size:58px;
    box-shadow: 0 0 60px rgba(0,212,170,0.45), inset 0 0 30px rgba(255,255,255,0.12);
    animation: float 4s ease-in-out infinite, glow 3s ease-in-out infinite;
}
@keyframes float { 0%,100% { transform: translateY(0);} 50% { transform: translateY(-10px);} }
@keyframes glow  { 0%,100% { box-shadow: 0 0 40px rgba(0,212,170,0.35), inset 0 0 30px rgba(255,255,255,0.10);}
                   50%     { box-shadow: 0 0 80px rgba(99,102,241,0.55), inset 0 0 30px rgba(255,255,255,0.18);} }
.land-badge {
    display:inline-block; padding:5px 16px; border-radius:999px; font-size:0.74rem; font-weight:600; letter-spacing:3px;
    color:#CFFAF0; background:rgba(0,212,170,0.12); border:1px solid rgba(0,212,170,0.4);
}
.land-hero h1 {
    margin:18px 0 8px 0; font-size:4rem; line-height:1.08; font-weight:800;
    background: linear-gradient(90deg,#FFFFFF,#7DF9E0 50%,#A5B4FC); -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color: transparent;
}
.land-hero p { color:#B4C2DA; font-size:1.1rem; max-width:680px; margin:0 auto; }
.land-stats { display:flex; justify-content:center; gap:16px; flex-wrap:wrap; margin:22px 0 8px 0; }
.land-stat {
    min-width:150px; padding:14px 22px; border-radius:18px; text-align:center;
    background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.10); backdrop-filter: blur(8px);
}
.land-stat b { display:block; font-size:1.7rem; color:#F8FAFC; } .land-stat span { color:#8EA0BE; font-size:0.8rem; }
.land-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin:26px 0 30px 0; }
.land-card {
    padding:22px; border-radius:20px; background:rgba(255,255,255,0.045); border:1px solid rgba(255,255,255,0.09);
    backdrop-filter: blur(8px); transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.land-card:hover { transform: translateY(-5px); border-color: rgba(0,212,170,0.6); box-shadow: 0 16px 40px rgba(0,212,170,0.15); }
.land-card .ic { font-size:1.9rem; } .land-card b { display:block; margin:10px 0 6px 0; color:#F1F5F9; font-size:1.02rem; }
.land-card p { margin:0; color:#9FB0C9; font-size:0.86rem; line-height:1.5; }
.land-foot { text-align:center; color:#7F90AD; font-size:0.82rem; margin:16px 0 30px 0; }
[data-testid="stForm"] {
    border:1px solid rgba(255,255,255,0.12); border-radius:22px; padding:22px;
    background:rgba(255,255,255,0.05); backdrop-filter: blur(10px); box-shadow: 0 14px 44px rgba(0,0,0,0.4);
}
button[kind="primary"], button[kind="primaryFormSubmit"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
    background: linear-gradient(90deg,#00D4AA,#6366F1) !important; color:#04121c !important; font-weight:700 !important;
    font-size:1.05rem !important; padding:0.7rem 1rem !important; border:none !important; border-radius:14px !important;
    box-shadow: 0 10px 30px rgba(0,212,170,0.35);
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover,
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
    transform: translateY(-2px) scale(1.01); box-shadow: 0 14px 36px rgba(99,102,241,0.5); color:#04121c !important;
}
@media (max-width: 800px) { .steps, .land-grid { grid-template-columns: 1fr; } .hero h1 { font-size:1.8rem; } .land-hero h1 { font-size:2.6rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =====================================================================
# Landing (enter) page
# =====================================================================
st.session_state.setdefault("page", "landing")
st.session_state.setdefault("user_name", "guest")


def render_landing():
    st.markdown(
        '<style>[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], '
        '[data-testid="collapsedControl"] {display:none !important;}</style>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="land-hero">'
        '<div class="orb">👁️</div>'
        '<span class="land-badge">AI DRIVER SAFETY</span>'
        '<h1>Stay awake.<br>Stay alive.</h1>'
        '<p>Safe Drive AI watches for tired eyes, yawns and head nods through your camera, '
        'learns what is normal for you, and warns you before drowsiness becomes dangerous.</p>'
        '</div>'
        '<div class="land-stats">'
        '<div class="land-stat"><b>5 sec</b><span>one-time calibration</span></div>'
        '<div class="land-stat"><b>6</b><span>fatigue signals combined</span></div>'
        '<div class="land-stat"><b>0</b><span>videos stored</span></div>'
        '</div>'
        '<div class="land-grid">'
        '<div class="land-card"><div class="ic">👁️</div><b>Eye closure tracking</b>'
        '<p>PERCLOS, blink length and micro-sleeps, with looking down told apart from a real blink.</p></div>'
        '<div class="land-card"><div class="ic">🥱</div><b>Yawn detection</b>'
        '<p>Catches open-mouth yawns and yawns hidden behind your hand, and warns on frequent yawning.</p></div>'
        '<div class="land-card"><div class="ic">🙇</div><b>Head nod alerts</b>'
        '<p>Notices when your head drops and stays down while your eyes are narrowing.</p></div>'
        '<div class="land-card"><div class="ic">🧠</div><b>Personalised to you</b>'
        '<p>Learns your normal eyes and mouth, and becomes less sensitive when you mark a false alarm.</p></div>'
        '<div class="land-card"><div class="ic">🌤️</div><b>Knows your energy dips</b>'
        '<p>Extra sensitivity in the 2 to 3 PM post-lunch dip, plus a report of your tiring hours.</p></div>'
        '<div class="land-card"><div class="ic">🔊</div><b>Voice and beep alerts</b>'
        '<p>Spoken warnings in Hindi or English and a loud beep when fatigue turns severe.</p></div>'
        '</div>',
        unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        with st.form("enter_form", clear_on_submit=False):
            name = st.text_input("Your name", placeholder="e.g. Shivansh", key="landing_name")
            go = st.form_submit_button("Enter Safe Drive AI  →", type="primary",
                                       use_container_width=True)
        if go:
            st.session_state["user_name"] = (name or "").strip() or "guest"
            st.session_state["page"] = "app"
            st.rerun()
    st.markdown('<div class="land-foot">Runs in your browser · Allow camera access when asked · '
                'Works best in good lighting</div>', unsafe_allow_html=True)


if st.session_state["page"] == "landing":
    render_landing()
    st.stop()
st.markdown(
    '<div class="hero">'
    '<h1>Safe Drive AI</h1>'
    '<p>A personal drowsiness and fatigue monitor that learns your face, your blink pattern '
    'and your daily energy dips, then warns you before you doze off.</p>'
    '<span class="chip">⚡ Live processing</span>'
    '<span class="chip">🔒 Video is never stored</span>'
    '<span class="chip">🔊 Hindi / English voice alerts</span>'
    '<span class="chip">🧠 Personalised to you</span>'
    '</div>'
    '<div class="steps">'
    '<div class="step"><span class="n">1</span><b>Start the camera</b>'
    '<p>Enter your name in the sidebar, press START and allow camera access.</p></div>'
    '<div class="step"><span class="n">2</span><b>Calibrate</b>'
    '<p>Look straight for 5 seconds so the app learns your normal eyes and mouth.</p></div>'
    '<div class="step"><span class="n">3</span><b>Stay alert</b>'
    '<p>Get beep and voice warnings on tired eyes, yawns and head nods.</p></div>'
    '</div>',
    unsafe_allow_html=True,
)


def status_card(ph, kind, text):
    ph.markdown(f'<div class="status {kind}">{text}</div>', unsafe_allow_html=True)


GAUGE_COLORS = ["#10b981", "#f59e0b", "#ef4444"]


def render_dashboard(gauge_ph, stats_ph, snap):
    """Circular fatigue gauge + icon stat cards (pure HTML/CSS)."""
    lvl = int(snap["level"])
    score = max(0.0, min(100.0, float(snap["score"])))
    c = GAUGE_COLORS[lvl]
    gauge_ph.markdown(
        '<div class="gauge-card">'
        '<div class="gauge-title">FATIGUE LEVEL</div>'
        f'<div class="ring" style="--glow:{c}55; background: conic-gradient(from 0deg, {c} {score * 3.6:.0f}deg, '
        'rgba(255,255,255,0.08) 0deg);">'
        '<div class="ring-inner">'
        f'<div class="ring-num">{score:.0f}</div>'
        f'<div class="ring-sub" style="color:{c}">{LEVEL_NAMES[lvl]}</div>'
        '</div></div></div>',
        unsafe_allow_html=True)
    cards = [
        ("👁️", "PERCLOS (60 s)", f"{snap['perclos'] * 100:.0f}%"),
        ("💫", "Blinks / min", f"{snap['blinks']:.0f}"),
        ("🥱", "Yawns", f"{snap['yawns']}"),
        ("🙇", "Head nods (2 min)", f"{snap['nods']}"),
        ("🔍", "Eye ratio (EAR)", f"{snap['ear']:.2f}"),
        ("👄", "Mouth ratio (MAR)", f"{snap['mar']:.2f}"),
        ("⬇️", "Looking down", "Yes" if snap["looking_down"] else "No"),
        ("🌤️", "2-3 PM dip mode", "On" if snap["dip"] else "Off"),
    ]
    html = '<div class="stats">' + "".join(
        f'<div class="stat"><div class="ic">{ic}</div><div class="vl">{vl}</div><div class="lb">{lb}</div></div>'
        for ic, lb, vl in cards) + '</div>'
    stats_ph.markdown(html, unsafe_allow_html=True)


IDLE_SNAP = dict(level=0, score=0.0, perclos=0.0, blinks=0.0, yawns=0, nods=0, ear=0.0,
                 mar=0.0, looking_down=False, dip=False)

with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">🚗</div><div><b>Safe Drive AI</b><small>Control panel</small></div></div>', unsafe_allow_html=True)
    if st.button("🏠 Home page", use_container_width=True):
        st.session_state["page"] = "landing"
        st.rerun()
    user = st.text_input("Your name", key="user_name").strip() or "guest"
    use_hands = st.toggle("Hand-over-mouth yawn detection", value=True,
                          help="Turn off if the app feels slow.")
    sound_on = st.toggle("Beep alert", value=True)
    voice_on = st.toggle("Voice alert (Hindi / English)", value=True)
    tz_name = st.text_input("Timezone (for the 2-3 PM dip)", value=DEFAULT_TZ)
    st.divider()
    recal_btn = st.button("🔄 Recalibrate", use_container_width=True)
    false_btn = st.button("❌ False alarm", use_container_width=True,
                          help="The alarm rang but you were fine: become less sensitive.")
    st.caption("Press START below the video, allow the camera, then look straight "
               "for 5 seconds while it calibrates.")

left, right = st.columns([3, 2], gap="large")
with left:
    ctx = webrtc_streamer(
        key=f"drive-{user}-{use_hands}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                      "frameRate": {"ideal": 15}},
            "audio": False,
        },
        video_processor_factory=lambda: DrowsyProcessor(user, use_hands, tz_name),
        async_processing=True,
    )
    banner = st.empty()
    note = st.empty()
with right:
    gauge_ph = st.empty()
    stats_ph = st.empty()
alert_ph = st.empty()
render_dashboard(gauge_ph, stats_ph, IDLE_SNAP)

proc = ctx.video_processor
if ctx.state.playing and proc is not None:
    st.session_state["proc"] = proc
    st.session_state.pop("result", None)
    if recal_btn:
        proc.request_recalibrate()
    if false_btn:
        proc.request_false_alarm()

    last = st.session_state.setdefault("last_alert", {"beep": 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
    n_alert = st.session_state.get("n_alert", 0)

    while True:
        snap = proc.snapshot()
        if snap["error"]:
            note.error(snap["error"])
        elif snap["welcome"]:
            note.info(snap["welcome"])
        if snap["message"]:
            note.success(snap["message"])

        lvl = snap["level"]
        if not snap["face"]:
            status_card(banner, "warn", "👀 No face detected. Sit in front of the camera with good light.")
        elif snap["calibrating"]:
            status_card(banner, "info", f"🎯 Calibrating... look straight, eyes open, mouth closed "
                                        f"({snap['calib_left']:.0f}s left)")
        elif lvl == 2:
            status_card(banner, "bad", "🚨 SEVERE: " + (", ".join(snap["alerts"]) or "high fatigue score")
                                       + ". Please stop and rest.")
        elif lvl == 1:
            status_card(banner, "warn", "⚠️ MILD fatigue. Stay alert.")
        else:
            status_card(banner, "ok", "✅ Normal. You are alert.")

        render_dashboard(gauge_ph, stats_ph, snap)

        # sound + voice in the browser (with the same cooldowns as the desktop script)
        t = time.time()
        if snap["face"] and not snap["calibrating"]:
            beep = False
            speak = 0
            if lvl == 2:
                if sound_on and t - last["beep"] >= 1.5:
                    beep, last["beep"] = True, t
                if voice_on and t - last[2] >= VOICE_COOLDOWN[2]:
                    speak, last[2] = 2, t
            elif lvl == 1:
                if voice_on and t - last[1] >= VOICE_COOLDOWN[1]:
                    speak, last[1] = 1, t
            elif snap["dip"]:
                if voice_on and t - last[3] >= VOICE_COOLDOWN[3]:
                    speak, last[3] = 3, t
            if beep or speak:
                n_alert += 1
                st.session_state["n_alert"] = n_alert
                browser_alert(alert_ph, n_alert, beep, speak)
        time.sleep(0.5)

elif not ctx.state.playing and st.session_state.get("proc") is not None:
    # the stream was stopped: save everything and show the session report
    old = st.session_state.pop("proc")
    st.session_state["result"] = old.finalize()

res = st.session_state.get("result")
if res and not ctx.state.playing:
    st.subheader(f"Session summary for {res['user']}")
    a, b, c, d = st.columns(4)
    a.metric("Avg fatigue", f"{res['avg']:.0f}")
    b.metric("Max fatigue", f"{res['max']:.0f}")
    c.metric("Yawns / nods", f"{res['yawns']} / {res['nods']}")
    d.metric("Severe alerts", res["alerts"])
    if res["peak"]:
        st.info(f"Your most fatigue-prone hour so far: {res['peak'][0]}:00-"
                f"{res['peak'][0] + 1}:00 (average {res['peak'][1]:.0f}).")
    if res["report"] and os.path.exists(res["report"]):
        st.image(res["report"])
        with open(res["report"], "rb") as f:
            st.download_button("Download report (PNG)", f, file_name=os.path.basename(res["report"]),
                               mime="image/png")
    else:
        st.caption("Run for at least a few seconds after calibration to get a report graph.")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("Download event log (CSV)", f, file_name="session_log.csv",
                               mime="text/csv")
