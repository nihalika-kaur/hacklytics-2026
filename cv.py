"""
SafeDrive - CV Prototype: Face & Emotion Detection
====================================================
Dependencies:
    pip install opencv-python mediapipe deepface tf-keras numpy scipy

Run:
    python face_emotion_detection.py

Controls:
    Q  - Quit
    S  - Save snapshot
    D  - Toggle debug landmarks overlay
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from deepface import DeepFace
import time
import threading

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
WEBCAM_INDEX        = 0
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720

# EAR (Eye Aspect Ratio) threshold for blink / closed-eye detection
EAR_THRESHOLD       = 0.22
EAR_CONSEC_FRAMES   = 2        # frames below threshold = blink

# Emotion re-analysis every N frames (DeepFace is slow)
EMOTION_INTERVAL    = 15

# Colours (BGR)
COL_GREEN   = (0,   210,  90)
COL_YELLOW  = (0,   220, 230)
COL_RED     = (0,    50, 240)
COL_WHITE   = (255, 255, 255)
COL_DARK    = ( 20,  20,  20)
COL_BLUE    = (230, 120,  30)

# ─────────────────────────────────────────────
#  MediaPipe FaceMesh indices
# ─────────────────────────────────────────────
# Left eye  landmarks (FaceMesh 468-point model)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye landmarks
RIGHT_EYE = [33,  160, 158,  133, 153, 144]
# Mouth outer landmarks
MOUTH     = [61,  291, 39,  181, 0,   17,  269, 405]

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def eye_aspect_ratio(landmarks, indices, w, h):
    """Compute EAR for a set of 6 eye landmark indices."""
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(landmarks, indices, w, h):
    """Compute MAR (Mouth Aspect Ratio) as a yawn proxy."""
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    vertical1 = distance.euclidean(pts[2], pts[6])
    vertical2 = distance.euclidean(pts[3], pts[5])
    horizontal = distance.euclidean(pts[0], pts[1])
    return (vertical1 + vertical2) / (2.0 * horizontal)


def draw_bar(frame, x, y, w, h, value, max_val, color, label):
    """Draw a labelled horizontal progress bar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), COL_DARK, -1)
    fill = int((value / max_val) * w)
    fill = min(fill, w)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), COL_WHITE, 1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1)


def overlay_panel(frame, lines, x=10, y=10, line_h=22, alpha=0.55):
    """Semi-transparent dark panel with text lines."""
    max_len = max(len(l) for l in lines) if lines else 1
    panel_w  = max_len * 9 + 16
    panel_h  = len(lines) * line_h + 12
    overlay  = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), COL_DARK, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + 8, y + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_WHITE, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  EMOTION WORKER (runs in a background thread)
# ─────────────────────────────────────────────

class EmotionWorker:
    def __init__(self):
        self.emotion   = "Initialising…"
        self.scores    = {}
        self._lock     = threading.Lock()
        self._frame    = None
        self._running  = True
        self._thread   = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update_frame(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def get_result(self):
        with self._lock:
            return self.emotion, dict(self.scores)

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            with self._lock:
                frame = self._frame.copy() if self._frame is not None else None
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                res = result[0] if isinstance(result, list) else result
                with self._lock:
                    self.emotion = res["dominant_emotion"]
                    self.scores  = res["emotion"]
            except Exception:
                pass
            time.sleep(0.1)   # throttle


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    emotion_worker = EmotionWorker()

    blink_count       = 0
    blink_consec      = 0
    frame_idx         = 0
    show_landmarks    = False
    fps_time          = time.time()

    # ── Emotion colour map ──────────────────────
    emotion_color = {
        "happy":     COL_GREEN,
        "sad":       COL_BLUE,
        "angry":     COL_RED,
        "surprise":  COL_YELLOW,
        "fear":      (130, 0, 200),
        "disgust":   (0, 140, 80),
        "neutral":   COL_WHITE,
    }

    print("[SafeDrive CV] Starting — press Q to quit, D to toggle landmarks, S to save snapshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Send frame to emotion worker every N frames ──
        if frame_idx % EMOTION_INTERVAL == 0:
            emotion_worker.update_frame(frame)

        emotion_label, emotion_scores = emotion_worker.get_result()
        emo_color = emotion_color.get(emotion_label.lower(), COL_WHITE)

        # ── MediaPipe face mesh ──────────────────────────
        result = face_mesh.process(rgb)

        ear_val  = 0.0
        mar_val  = 0.0
        face_det = False

        if result.multi_face_landmarks:
            face_det  = True
            lm        = result.multi_face_landmarks[0].landmark

            ear_l = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            ear_r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear_val   = (ear_l + ear_r) / 2.0
            mar_val   = mouth_aspect_ratio(lm, MOUTH, w, h)

            # ── Blink detection ──────────────────────
            if ear_val < EAR_THRESHOLD:
                blink_consec += 1
            else:
                if blink_consec >= EAR_CONSEC_FRAMES:
                    blink_count += 1
                blink_consec = 0

            # ── Draw facial landmarks (optional) ──────
            if show_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    result.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    frame,
                    result.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                )

            # ── Eye bounding boxes ─────────────────────
            for eye_idx in [LEFT_EYE, RIGHT_EYE]:
                pts = np.array(
                    [[int(lm[i].x * w), int(lm[i].y * h)] for i in eye_idx]
                )
                x1, y1 = pts.min(axis=0) - 6
                x2, y2 = pts.max(axis=0) + 6
                eye_col = COL_RED if ear_val < EAR_THRESHOLD else COL_GREEN
                cv2.rectangle(frame, (x1, y1), (x2, y2), eye_col, 1)

            # ── Mouth bounding box ────────────────────
            mouth_pts = np.array(
                [[int(lm[i].x * w), int(lm[i].y * h)] for i in MOUTH]
            )
            mx1, my1 = mouth_pts.min(axis=0) - 6
            mx2, my2 = mouth_pts.max(axis=0) + 6
            mouth_col = COL_YELLOW if mar_val > 0.6 else COL_GREEN
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), mouth_col, 1)

        # ── Emotion badge ──────────────────────────────
        badge_text = f"  {emotion_label.upper()}  "
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        bx = w - tw - 20
        by = 20
        cv2.rectangle(frame, (bx - 8, by - th - 6), (bx + tw + 8, by + 8), emo_color, -1)
        cv2.putText(frame, badge_text, (bx, by), cv2.FONT_HERSHEY_DUPLEX, 0.8, COL_DARK, 2)

        # ── Emotion score bars ─────────────────────────
        bar_x = w - 200
        bar_y_start = by + 30
        for i, (emo, score) in enumerate(
            sorted(emotion_scores.items(), key=lambda x: -x[1])
        ):
            col = emotion_color.get(emo, COL_WHITE)
            draw_bar(frame, bar_x, bar_y_start + i * 32, 180, 16,
                     score, 100.0, col, emo)

        # ── Info panel (top-left) ──────────────────────
        fps = 1.0 / (time.time() - fps_time + 1e-9)
        fps_time = time.time()

        face_status = "DETECTED" if face_det else "NOT FOUND"
        eye_status  = "CLOSED" if ear_val < EAR_THRESHOLD else "OPEN"
        yawn_status = "YAWN" if mar_val > 0.6 else "Normal"

        info_lines = [
            f"FPS      : {fps:4.1f}",
            f"Face     : {face_status}",
            f"EAR      : {ear_val:.3f}  Eyes: {eye_status}",
            f"MAR      : {mar_val:.3f}  Mouth: {yawn_status}",
            f"Blinks   : {blink_count}",
            f"Emotion  : {emotion_label}",
            f"Landmarks: {'ON' if show_landmarks else 'OFF'}  [D]",
        ]
        overlay_panel(frame, info_lines, x=10, y=10)

        # ── Alert overlay ──────────────────────────────
        if not face_det:
            cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 160, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, COL_RED, 3)

        # ── Show frame ────────────────────────────────
        cv2.imshow("SafeDrive — Face & Emotion Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            show_landmarks = not show_landmarks
        elif key == ord("s"):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[SafeDrive] Snapshot saved → {fname}")

    emotion_worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[SafeDrive CV] Stopped.")


if __name__ == "__main__":
    main()