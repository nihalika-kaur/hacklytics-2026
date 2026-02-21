"""
SafeDrive — AI-Enhanced Driver Fatigue Prevention System
=========================================================
Geometry-driven, temporal-aware fatigue detection.
Uses MediaPipe Tasks API (0.10.32+) — no mp.solutions needed.

Dependencies: opencv-python, mediapipe, numpy, scipy
Controls:    Q=quit  D=toggle mesh  S=snapshot
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import time
import os
import urllib.request

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════��═══════════════════════════════════════════════════════

WEBCAM_INDEX = 0
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# --- Eyelid thresholds ---
LID_CLOSE_RATIO = 0.018
LID_OPEN_RATIO = 0.030

# --- Blink temporal model ---
BLINK_MIN_FRAMES = 2
BLINK_MAX_FRAMES = 12
MICROSLEEP_FRAMES = 15

# --- Yawn detection ---
YAWN_RATIO_THRESHOLD = 0.08
YAWN_MIN_FRAMES = 25

# --- Head nod detection ---
NOD_PITCH_THRESHOLD = 15.0
NOD_RECOVERY_THRESHOLD = 8.0
NOD_WINDOW_FRAMES = 30

# --- PERCLOS ---
PERCLOS_WINDOW_SEC = 60

# --- Fatigue score weights ---
W_PERCLOS = 0.45
W_MICROSLEEP = 0.20
W_YAWN = 0.15
W_NOD = 0.10
W_BLINK_VAR = 0.10

# --- Colors (BGR) ---
COL_GREEN = (0, 220, 100)
COL_YELLOW = (0, 220, 255)
COL_RED = (0, 0, 255)
COL_CYAN = (255, 220, 0)
COL_WHITE = (255, 255, 255)
COL_GREY = (180, 180, 180)
COL_PANEL = (30, 30, 30)
COL_ORANGE = (0, 140, 255)

# ═══════════════════════════════════════════════════════════════
# MEDIAPIPE LANDMARK INDICES (468 FaceMesh)
# ═══════════════════════════════════════════════════════════════

R_EYE_UPPER = 159
R_EYE_LOWER = 145
L_EYE_UPPER = 386
L_EYE_LOWER = 374

R_EYE_POINTS = [33, 160, 158, 133, 153, 144]
L_EYE_POINTS = [263, 387, 385, 362, 380, 373]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 263
RIGHT_EYE_CORNER = 33
LEFT_MOUTH_CORNER = 287
RIGHT_MOUTH_CORNER = 57

FOREHEAD = 10

# solvePnP 3D model
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

POSE_LANDMARK_INDICES = [
    NOSE_TIP, CHIN, LEFT_EYE_CORNER, RIGHT_EYE_CORNER,
    LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER
]

# Mesh drawing connections (subset for visualization)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

RIGHT_EYE_CONN = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
LEFT_EYE_CONN = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263]
LIPS_CONN = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]


# ═══════════════════════════════════════════════════════════════
# GEOMETRY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def lm_px(landmarks, idx, w, h):
    """Convert NormalizedLandmark to pixel coords."""
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def get_lid_distance(landmarks, upper_idx, lower_idx, w, h):
    ux, uy = lm_px(landmarks, upper_idx, w, h)
    lx, ly = lm_px(landmarks, lower_idx, w, h)
    return abs(ly - uy), (ux, uy), (lx, ly)


def get_mouth_distance(landmarks, w, h):
    tx, ty = lm_px(landmarks, MOUTH_TOP, w, h)
    bx, by = lm_px(landmarks, MOUTH_BOTTOM, w, h)
    return abs(by - ty), (tx, ty), (bx, by)


def get_face_height(landmarks, w, h):
    _, fy = lm_px(landmarks, FOREHEAD, w, h)
    _, cy = lm_px(landmarks, CHIN, w, h)
    return max(abs(cy - fy), 1)


def compute_ear_6point(landmarks, indices, w, h):
    pts = [lm_px(landmarks, i, w, h) for i in indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def estimate_head_pose(landmarks, w, h):
    image_points = np.array(
        [lm_px(landmarks, idx, w, h) for idx in POSE_LANDMARK_INDICES],
        dtype=np.float64
    )
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    pitch = np.degrees(np.arctan2(-rmat[2][0], np.sqrt(rmat[2][1]**2 + rmat[2][2]**2)))
    yaw = np.degrees(np.arctan2(rmat[1][0], rmat[0][0]))
    roll = np.degrees(np.arctan2(rmat[2][1], rmat[2][2]))
    return pitch, yaw, roll


# ═══════════════════════════════════════════════════════════════
# FATIGUE TRACKER
# ═══════════════════════════════════════════════════════════════

class FatigueTracker:
    def __init__(self):
        self.closed_frames = 0
        self.blink_count = 0
        self.microsleep_count = 0
        self.perclos_history = deque()
        self.blink_timestamps = deque(maxlen=50)
        self.yawn_frames = 0
        self.yawn_count = 0
        self.pitch_history = deque(maxlen=NOD_WINDOW_FRAMES)
        self.nod_count = 0
        self.nod_cooldown = 0
        self.start_time = time.time()

    def update_eyes(self, lid_ratio, timestamp):
        is_closed = lid_ratio < LID_CLOSE_RATIO
        self.perclos_history.append((timestamp, is_closed))
        cutoff = timestamp - PERCLOS_WINDOW_SEC
        while self.perclos_history and self.perclos_history[0][0] < cutoff:
            self.perclos_history.popleft()

        if is_closed:
            self.closed_frames += 1
        else:
            if BLINK_MIN_FRAMES <= self.closed_frames <= BLINK_MAX_FRAMES:
                self.blink_count += 1
                self.blink_timestamps.append(timestamp)
            elif self.closed_frames > MICROSLEEP_FRAMES:
                self.microsleep_count += 1
            self.closed_frames = 0
        return is_closed

    def update_yawn(self, mouth_ratio):
        if mouth_ratio > YAWN_RATIO_THRESHOLD:
            self.yawn_frames += 1
        else:
            if self.yawn_frames > YAWN_MIN_FRAMES:
                self.yawn_count += 1
            self.yawn_frames = 0
        return mouth_ratio > YAWN_RATIO_THRESHOLD

    def update_head(self, pitch):
        self.pitch_history.append(pitch)
        if self.nod_cooldown > 0:
            self.nod_cooldown -= 1
            return False
        if len(self.pitch_history) < 10:
            return False
        recent = list(self.pitch_history)
        baseline = np.mean(recent[:5])
        min_pitch = min(recent[5:]) if len(recent) > 5 else baseline
        max_drop = baseline - min_pitch
        recovery = recent[-1] - min_pitch if len(recent) > 8 else 0
        if max_drop > NOD_PITCH_THRESHOLD and recovery > NOD_RECOVERY_THRESHOLD:
            self.nod_count += 1
            self.nod_cooldown = 20
            self.pitch_history.clear()
            return True
        return False

    @property
    def perclos(self):
        if not self.perclos_history:
            return 0.0
        return sum(1 for _, c in self.perclos_history if c) / len(self.perclos_history)

    @property
    def blink_rate(self):
        elapsed = max(time.time() - self.start_time, 1)
        return (self.blink_count / elapsed) * 60

    @property
    def blink_variability(self):
        if len(self.blink_timestamps) < 3:
            return 0.0
        ts = list(self.blink_timestamps)
        intervals = [ts[i] - ts[i-1] for i in range(1, len(ts))]
        if not intervals:
            return 0.0
        m = np.mean(intervals)
        return np.std(intervals) / m if m > 0 else 0.0

    @property
    def fatigue_score(self):
        p = min(self.perclos / 0.4, 1.0) * 100
        m = min(self.microsleep_count / 3, 1.0) * 100
        y = min(self.yawn_count / 3, 1.0) * 100
        n = min(self.nod_count / 3, 1.0) * 100
        bv = min(self.blink_variability / 0.8, 1.0) * 100
        return min(W_PERCLOS*p + W_MICROSLEEP*m + W_YAWN*y + W_NOD*n + W_BLINK_VAR*bv, 100.0)

    @property
    def risk_level(self):
        s = self.fatigue_score
        if s < 25: return "low"
        elif s < 55: return "medium"
        else: return "high"


# ═══════════════════════════════════════════════════════════════
# DRAWING
# ═══════════════════════════════════════════════════════════════

def draw_panel(frame, x, y, w, h, alpha=0.7):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), COL_PANEL, -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)


def put(frame, text, pos, color=COL_WHITE, scale=0.50, thick=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_lid_dots(frame, upper_pt, lower_pt, is_closed):
    color = COL_RED if is_closed else COL_GREEN
    cv2.circle(frame, upper_pt, 4, color, -1)
    cv2.circle(frame, lower_pt, 4, color, -1)
    cv2.line(frame, upper_pt, lower_pt, color, 2)


def draw_mouth_dots(frame, top_pt, bot_pt, is_yawning):
    color = COL_YELLOW if is_yawning else COL_GREEN
    cv2.circle(frame, top_pt, 4, color, -1)
    cv2.circle(frame, bot_pt, 4, color, -1)
    cv2.line(frame, top_pt, bot_pt, color, 2)


def draw_mesh_lines(frame, landmarks, indices, color, w, h):
    """Draw connected lines through a list of landmark indices."""
    pts = [lm_px(landmarks, i, w, h) for i in indices]
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i+1], color, 1, cv2.LINE_AA)


def risk_color(level):
    return {"low": COL_GREEN, "medium": COL_YELLOW, "high": COL_RED}.get(level, COL_WHITE)


# ═══════════════════════════════════════════════════════════════
# MODEL DOWNLOAD
# ═══════════════════════════════════════════════════════════════

def ensure_model():
    if os.path.isfile(MODEL_PATH):
        return True
    print(f"Downloading FaceLandmarker model (~2MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        print(f"Manually download from:\n  {MODEL_URL}")
        print(f"Save as: {MODEL_PATH}")
        return False


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    if not ensure_model():
        return

    # --- Camera ---
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {WEBCAM_INDEX}")
        print("Run this to find your camera:")
        print('  python -c "import cv2; [print(f\'Camera {i}: OK\' if cv2.VideoCapture(i).read()[0] else f\'Camera {i}: FAIL\') for i in range(5)]"')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, test_frame = cap.read()
    if not ret:
        print("ERROR: Camera opened but cannot read frames.")
        return
    actual_h, actual_w = test_frame.shape[:2]
    print(f"Camera: {actual_w}x{actual_h}")

    # --- FaceLandmarker (Tasks API) ---
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # --- State ---
    tracker = FatigueTracker()
    show_mesh = False
    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    print("=" * 50)
    print("SafeDrive Fatigue Detection — Running")
    print("=" * 50)
    print("Controls: Q=quit  D=toggle mesh  S=snapshot")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed. Retrying...")
            time.sleep(0.5)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)
        frame_count += 1

        # --- FaceLandmarker ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(now * 1000)

        try:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            result = None

        face_found = False
        avg_lid_ratio = 0.0
        mouth_ratio = 0.0
        pitch, yaw, roll = 0.0, 0.0, 0.0
        eyes_closed = False
        is_yawning = False

        if result and result.face_landmarks:
            face_found = True
            landmarks = result.face_landmarks[0]  # list of NormalizedLandmark

            face_h = get_face_height(landmarks, w, h)

            # === EYELID DOT COLLISION ===
            r_dist, r_up, r_lo = get_lid_distance(landmarks, R_EYE_UPPER, R_EYE_LOWER, w, h)
            l_dist, l_up, l_lo = get_lid_distance(landmarks, L_EYE_UPPER, L_EYE_LOWER, w, h)
            r_ratio = r_dist / face_h
            l_ratio = l_dist / face_h
            avg_lid_ratio = (r_ratio + l_ratio) / 2.0

            eyes_closed = tracker.update_eyes(avg_lid_ratio, now)

            draw_lid_dots(frame, r_up, r_lo, eyes_closed)
            draw_lid_dots(frame, l_up, l_lo, eyes_closed)

            # === MOUTH / YAWN ===
            m_dist, m_top, m_bot = get_mouth_distance(landmarks, w, h)
            mouth_ratio = m_dist / face_h
            is_yawning = tracker.update_yawn(mouth_ratio)
            draw_mouth_dots(frame, m_top, m_bot, is_yawning)

            # === HEAD POSE ===
            pitch, yaw, roll = estimate_head_pose(landmarks, w, h)
            tracker.update_head(pitch)

            # === FACE BOX ===
            xs = [int(landmarks[i].x * w) for i in range(len(landmarks))]
            ys = [int(landmarks[i].y * h) for i in range(len(landmarks))]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            box_col = COL_RED if tracker.risk_level == "high" else COL_CYAN
            cv2.rectangle(frame, (x_min-10, y_min-10), (x_max+10, y_max+10), box_col, 2)

            # === MESH OVERLAY ===
            if show_mesh:
                # Draw face oval, eyes, lips
                draw_mesh_lines(frame, landmarks, FACE_OVAL, (80, 80, 80), w, h)
                draw_mesh_lines(frame, landmarks, RIGHT_EYE_CONN, COL_GREEN, w, h)
                draw_mesh_lines(frame, landmarks, LEFT_EYE_CONN, COL_GREEN, w, h)
                draw_mesh_lines(frame, landmarks, LIPS_CONN, (180, 100, 255), w, h)
                # Draw all landmark dots
                for i in range(len(landmarks)):
                    px, py = lm_px(landmarks, i, w, h)
                    cv2.circle(frame, (px, py), 1, (60, 60, 60), -1)

        # ═══════════════════════════════════════════════════════
        # HUD PANEL
        # ═══════════════════════════════════════════════════════
        panel_w, panel_h = 310, 340
        draw_panel(frame, 8, 8, panel_w, panel_h)

        y0 = 30
        gap = 22

        put(frame, "SAFEDRIVE FATIGUE MONITOR", (16, y0), COL_CYAN, 0.50, 2)
        y0 += gap + 5

        put(frame, f"FPS: {fps:.0f}", (16, y0), COL_GREEN, 0.45)
        put(frame, f"Face: {'DETECTED' if face_found else 'NOT FOUND'}",
            (120, y0), COL_GREEN if face_found else COL_RED, 0.45)
        y0 += gap
        cv2.line(frame, (16, y0-8), (panel_w, y0-8), (60, 60, 60), 1)

        eye_col = COL_RED if eyes_closed else COL_GREEN
        put(frame, f"Eyes: {'CLOSED' if eyes_closed else 'OPEN'}", (16, y0), eye_col, 0.50, 2)
        y0 += gap
        put(frame, f"Lid Ratio: {avg_lid_ratio:.4f}", (16, y0),
            COL_RED if avg_lid_ratio < LID_CLOSE_RATIO else COL_WHITE, 0.42)
        y0 += gap
        put(frame, f"Blinks: {tracker.blink_count}  ({tracker.blink_rate:.0f}/min)",
            (16, y0), COL_WHITE, 0.42)
        y0 += gap
        put(frame, f"Microsleeps: {tracker.microsleep_count}",
            (16, y0), COL_RED if tracker.microsleep_count > 0 else COL_WHITE, 0.42)
        y0 += gap

        mouth_col = COL_YELLOW if is_yawning else COL_WHITE
        put(frame, f"Mouth: {mouth_ratio:.4f}  Yawns: {tracker.yawn_count}",
            (16, y0), mouth_col, 0.42)
        y0 += gap

        put(frame, f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}",
            (16, y0), COL_WHITE, 0.42)
        y0 += gap
        put(frame, f"Head Nods: {tracker.nod_count}",
            (16, y0), COL_ORANGE if tracker.nod_count > 0 else COL_WHITE, 0.42)
        y0 += gap
        cv2.line(frame, (16, y0-8), (panel_w, y0-8), (60, 60, 60), 1)

        perclos = tracker.perclos
        pc = COL_RED if perclos > 0.3 else (COL_YELLOW if perclos > 0.15 else COL_GREEN)
        put(frame, f"PERCLOS: {perclos:.1%}", (16, y0), pc, 0.50, 2)
        y0 += gap

        bv = tracker.blink_variability
        put(frame, f"Blink Variability: {bv:.2f}", (16, y0),
            COL_YELLOW if bv > 0.5 else COL_WHITE, 0.42)
        y0 += gap + 5

        score = tracker.fatigue_score
        risk = tracker.risk_level
        rc = risk_color(risk)
        put(frame, f"FATIGUE: {score:.0f}/100  [{risk.upper()}]", (16, y0), rc, 0.60, 2)
        y0 += 12

        bar_total = panel_w - 30
        cv2.rectangle(frame, (16, y0), (16+bar_total, y0+16), (50, 50, 50), -1)
        bar_fill = int(bar_total * min(score/100, 1.0))
        if bar_fill > 0:
            cv2.rectangle(frame, (16, y0), (16+bar_fill, y0+16), rc, -1)
        cv2.rectangle(frame, (16, y0), (16+bar_total, y0+16), (80, 80, 80), 1)

        # ═══════════════════════════════════════════════════════
        # EVENT LOG (Right Side)
        # ═══════════════════════════════════════════════════════
        if tracker.microsleep_count > 0 or tracker.yawn_count > 0 or tracker.nod_count > 0:
            log_w, log_h = 240, 100
            lx = w - log_w - 10
            ly = 10
            draw_panel(frame, lx, ly, log_w, log_h)
            put(frame, "EVENT LOG", (lx+10, ly+22), COL_CYAN, 0.45, 1)
            ely = ly + 42
            if tracker.microsleep_count > 0:
                put(frame, f"! Microsleeps: {tracker.microsleep_count}", (lx+10, ely), COL_RED, 0.42)
                ely += 20
            if tracker.yawn_count > 0:
                put(frame, f"  Yawns: {tracker.yawn_count}", (lx+10, ely), COL_YELLOW, 0.42)
                ely += 20
            if tracker.nod_count > 0:
                put(frame, f"  Head nods: {tracker.nod_count}", (lx+10, ely), COL_ORANGE, 0.42)

        # ═══════════════════════════════════════════════════════
        # ALERTS
        # ═══════════════════════════════════════════════════════
        if risk == "high":
            txt = "!! FATIGUE ALERT — PULL OVER AND REST !!"
            ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
            ax = (w - ts[0]) // 2
            ay = h - 35
            draw_panel(frame, ax-20, ay-32, ts[0]+40, 50, 0.75)
            if frame_count % 20 < 10:
                put(frame, txt, (ax, ay), COL_RED, 0.85, 2)
            else:
                put(frame, txt, (ax, ay), COL_WHITE, 0.85, 2)
        elif risk == "medium":
            txt = "Mild fatigue detected — consider a break"
            ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            ax = (w - ts[0]) // 2
            ay = h - 35
            draw_panel(frame, ax-15, ay-28, ts[0]+30, 45, 0.55)
            put(frame, txt, (ax, ay), COL_YELLOW, 0.65, 2)
        elif tracker.microsleep_count > 0:
            txt = f"MICROSLEEP DETECTED ({tracker.microsleep_count}x) — Stay alert!"
            ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            ax = (w - ts[0]) // 2
            ay = h - 35
            draw_panel(frame, ax-15, ay-28, ts[0]+30, 45, 0.55)
            put(frame, txt, (ax, ay), COL_ORANGE, 0.65, 2)

        put(frame, "Q:quit  D:mesh  S:snapshot", (10, h-10), COL_GREY, 0.38)

        cv2.imshow("SafeDrive — Fatigue Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            show_mesh = not show_mesh
            print(f"Mesh overlay: {'ON' if show_mesh else 'OFF'}")
        elif key == ord("s"):
            fn = f"safedrive_snap_{int(time.time())}.jpg"
            cv2.imwrite(fn, frame)
            print(f"Saved: {fn}")

    # ═══════════════════════════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════════════════════════
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - tracker.start_time
    print()
    print("=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"Duration:        {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Avg FPS:         {frame_count/elapsed:.1f}")
    print(f"Blinks:          {tracker.blink_count} ({tracker.blink_rate:.1f}/min)")
    print(f"Blink CV:        {tracker.blink_variability:.2f}")
    print(f"Microsleeps:     {tracker.microsleep_count}")
    print(f"Yawns:           {tracker.yawn_count}")
    print(f"Head Nods:       {tracker.nod_count}")
    print(f"Final PERCLOS:   {tracker.perclos:.1%}")
    print(f"Fatigue Score:   {tracker.fatigue_score:.0f}/100 [{tracker.risk_level.upper()}]")
    print("=" * 50)


if __name__ == "__main__":
    main()