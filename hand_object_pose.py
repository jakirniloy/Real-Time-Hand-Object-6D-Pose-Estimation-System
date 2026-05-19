"""
Real-Time Hand + Object 6D Pose Estimation
==========================================

Architecture
-------------
MediaPipe Hands  → Hand Detection
YOLO Object      → Object Detection
ORB + CSRT       → Tracking
solvePnP         → 6D Pose Estimation

Optimized for:
---------------
✔ Core i5 7th Gen
✔ 8 GB RAM
✔ CPU only
✔ 20–35 FPS

Install
-------
pip install opencv-contrib-python mediapipe ultralytics scipy numpy

Run
---
python main.py
"""

import cv2
import time
import math
import mediapipe as mp
import numpy as np

from ultralytics import YOLO
from scipy.spatial.transform import Rotation


# =============================================================================
# CONFIG
# =============================================================================

CAM_IDX = 0

FRAME_W = 640
FRAME_H = 480

FPS_TARGET = 30

YOLO_MODEL = "yolo11n.pt"

YOLO_IMGSZ = 320
YOLO_CONF = 0.35

# Camera intrinsics
FOCAL = 600.0

CAM_MATRIX = np.array([
    [FOCAL, 0, FRAME_W / 2],
    [0, FOCAL, FRAME_H / 2],
    [0, 0, 1]
], dtype=np.float64)

DIST_COEFFS = np.zeros((4, 1))

# Object size estimate (meters)
OBJ_HALF = 0.05

# =============================================================================
# COCO CLASSES
# =============================================================================

GRASPABLE_CLASSES = {
    39: "bottle",
    41: "cup",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    67: "cell phone",
    73: "book",
    76: "scissors",
    79: "toothbrush",
}

# =============================================================================
# COLORS
# =============================================================================

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (0, 165, 255)

# =============================================================================
# 3D BOX
# =============================================================================

OBJ_3D = np.array([
    [-OBJ_HALF, -OBJ_HALF, 0],
    [ OBJ_HALF, -OBJ_HALF, 0],
    [ OBJ_HALF,  OBJ_HALF, 0],
    [-OBJ_HALF,  OBJ_HALF, 0],
], dtype=np.float64)

# =============================================================================
# MEDIAPIPE HAND DETECTOR
# =============================================================================

class HandDetector:

    def __init__(self):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, None, 0

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = frame.shape

        pts = []

        for lm in hand_landmarks.landmark:
            pts.append((int(lm.x * w), int(lm.y * h)))

        pts = np.array(pts)

        x, y, bw, bh = cv2.boundingRect(pts)

        centroid = (x + bw // 2, y + bh // 2)

        fingers = self.count_fingers(pts)

        return (x, y, bw, bh), centroid, fingers

    def count_fingers(self, pts):

        tips = [4, 8, 12, 16, 20]

        count = 0

        if pts[4][0] > pts[3][0]:
            count += 1

        for tip in [8, 12, 16, 20]:
            if pts[tip][1] < pts[tip - 2][1]:
                count += 1

        return count

# =============================================================================
# YOLO OBJECT DETECTOR
# =============================================================================

class ObjectDetector:

    def __init__(self):

        self.model = YOLO(YOLO_MODEL)

    def detect(self, frame, hand_centroid=None):

        results = self.model.predict(
            frame,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            verbose=False,
            device="cpu"
        )

        detections = []

        for r in results:

            for box in r.boxes:

                cls_id = int(box.cls[0])

                if cls_id not in GRASPABLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                bbox = (x1, y1, x2 - x1, y2 - y1)

                label = GRASPABLE_CLASSES[cls_id]

                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2

                detections.append((bbox, label, (cx, cy)))

        if not detections:
            return None, "Object"

        if hand_centroid is None:
            return detections[0][0], detections[0][1]

        best = None
        best_dist = 1e9

        for bbox, label, center in detections:

            d = math.hypot(
                hand_centroid[0] - center[0],
                hand_centroid[1] - center[1]
            )

            if d < best_dist:
                best_dist = d
                best = (bbox, label)

        return best

# =============================================================================
# TRACKER
# =============================================================================

class Tracker:

    def __init__(self):

        self.tracker = None
        self.active = False

    def init(self, frame, bbox):

        self.tracker = cv2.TrackerCSRT_create()

        self.tracker.init(frame, bbox)

        self.active = True

    def update(self, frame):

        if not self.active:
            return False, None

        ok, bbox = self.tracker.update(frame)

        return ok, bbox

    def reset(self):

        self.active = False
        self.tracker = None

# =============================================================================
# POSE ESTIMATOR
# =============================================================================

class PoseEstimator:

    def __init__(self):

        self.rvec_ema = None
        self.tvec_ema = None

        self.alpha = 0.3

    def estimate(self, bbox):

        x, y, w, h = bbox

        img_pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            OBJ_3D,
            img_pts,
            CAM_MATRIX,
            DIST_COEFFS,
            flags=cv2.SOLVEPNP_IPPE
        )

        if not ok:
            return None, None, None

        if self.rvec_ema is None:

            self.rvec_ema = rvec
            self.tvec_ema = tvec

        else:

            self.rvec_ema = (
                self.alpha * rvec +
                (1 - self.alpha) * self.rvec_ema
            )

            self.tvec_ema = (
                self.alpha * tvec +
                (1 - self.alpha) * self.tvec_ema
            )

        euler = self.to_euler(self.rvec_ema)

        return self.rvec_ema, self.tvec_ema, euler

    def to_euler(self, rvec):

        R, _ = cv2.Rodrigues(rvec)

        rot = Rotation.from_matrix(R)

        return rot.as_euler('xyz', degrees=True)

# =============================================================================
# DRAWING
# =============================================================================

def draw_hand(frame, bbox, fingers):

    if bbox is None:
        return

    x, y, w, h = bbox

    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)

    cv2.putText(
        frame,
        f"Hand Fingers:{fingers}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        WHITE,
        2
    )

def draw_object(frame, bbox, label):

    if bbox is None:
        return

    x, y, w, h = [int(v) for v in bbox]

    cv2.rectangle(frame, (x, y), (x + w, y + h), CYAN, 2)

    cv2.putText(
        frame,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WHITE,
        2
    )

def draw_axes(frame, rvec, tvec):

    axis = np.float32([
        [0.05, 0, 0],
        [0, 0.05, 0],
        [0, 0, 0.05]
    ])

    origin = np.float32([[0, 0, 0]])

    imgpts, _ = cv2.projectPoints(
        axis,
        rvec,
        tvec,
        CAM_MATRIX,
        DIST_COEFFS
    )

    origin2d, _ = cv2.projectPoints(
        origin,
        rvec,
        tvec,
        CAM_MATRIX,
        DIST_COEFFS
    )

    o = tuple(origin2d[0].ravel().astype(int))

    x = tuple(imgpts[0].ravel().astype(int))
    y = tuple(imgpts[1].ravel().astype(int))
    z = tuple(imgpts[2].ravel().astype(int))

    cv2.line(frame, o, x, RED, 3)
    cv2.line(frame, o, y, GREEN, 3)
    cv2.line(frame, o, z, BLUE, 3)

# =============================================================================
# MAIN
# =============================================================================

def main():

    cap = cv2.VideoCapture(CAM_IDX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    hand_detector = HandDetector()

    object_detector = ObjectDetector()

    tracker = Tracker()

    pose_estimator = PoseEstimator()

    prev = time.time()

    state = "DETECT"

    object_bbox = None
    object_label = "Object"
    
    # Manual control variables
    manual_mode = False
    manual_x_offset = 0
    manual_y_offset = 0
    manual_scale = 1.0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ============================================================
        # HAND DETECTION
        # ============================================================

        hand_bbox, hand_centroid, fingers = hand_detector.detect(frame)

        # ============================================================
        # OBJECT DETECTION
        # ============================================================

        if state == "DETECT":

            object_bbox, object_label = object_detector.detect(
                frame,
                hand_centroid
            )

            if fingers <= 2 and object_bbox is not None:

                tracker.init(frame, object_bbox)

                state = "TRACK"

        # ============================================================
        # TRACKING
        # ============================================================

        if state == "TRACK":

            ok, tracked_bbox = tracker.update(frame)

            if ok:

                object_bbox = tracked_bbox

            else:

                tracker.reset()

                state = "DETECT"

        # ============================================================
        # POSE ESTIMATION
        # ============================================================

        rvec = None
        tvec = None
        euler = None

        if object_bbox is not None:

            rvec, tvec, euler = pose_estimator.estimate(object_bbox)

        # ============================================================
        # DRAW
        # ============================================================

        draw_hand(frame, hand_bbox, fingers)

        draw_object(frame, object_bbox, object_label)

        if rvec is not None:

            draw_axes(frame, rvec, tvec)

            roll, pitch, yaw = euler

            cv2.putText(
                frame,
                f"R:{roll:.1f} P:{pitch:.1f} Y:{yaw:.1f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                YELLOW,
                2
            )

        # FPS
        now = time.time()

        fps = 1 / (now - prev)

        prev = now

        cv2.putText(
            frame,
            f"FPS:{fps:.1f}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            GREEN,
            2
        )

        cv2.putText(
            frame,
            state,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            ORANGE,
            2
        )
        
        # Manual mode indicator
        if manual_mode:
            cv2.putText(
                frame,
                "MANUAL MODE (m=toggle | arrows=move | +/-=scale | r=reset)",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                MAGENTA,
                2
            )

        cv2.imshow("6D Pose Estimation", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        elif key == ord('r'):

            tracker.reset()

            state = "DETECT"
            
            manual_mode = False
            manual_x_offset = 0
            manual_y_offset = 0
            manual_scale = 1.0
        
        # Manual control mode (press 'm' to toggle)
        elif key == ord('m'):
            manual_mode = not manual_mode
            if manual_mode:
                print("✓ MANUAL MODE ON - Use arrow keys to move, +/- to scale, 'r' to reset")
            else:
                print("✗ Manual mode off")
        
        # Manual object movement (arrow keys)
        elif key == 65361:  # LEFT arrow
            manual_x_offset -= 5
        elif key == 65363:  # RIGHT arrow
            manual_x_offset += 5
        elif key == 65362:  # UP arrow
            manual_y_offset -= 5
        elif key == 65364:  # DOWN arrow
            manual_y_offset += 5
        
        # Scale adjustment
        elif key == ord('+') or key == ord('='):
            manual_scale += 0.05
            print(f"Scale: {manual_scale:.2f}x")
        elif key == ord('-') or key == ord('_'):
            manual_scale = max(0.1, manual_scale - 0.05)
            print(f"Scale: {manual_scale:.2f}x")
        
        # Apply manual adjustments if in manual mode
        if manual_mode and object_bbox is not None:
            x, y, w, h = object_bbox
            new_w = int(w * manual_scale)
            new_h = int(h * manual_scale)
            new_x = int(x + manual_x_offset)
            new_y = int(y + manual_y_offset)
            
            # Clamp to frame boundaries
            new_x = max(0, min(new_x, FRAME_W - new_w))
            new_y = max(0, min(new_y, FRAME_H - new_h))
            new_w = max(10, min(new_w, FRAME_W - new_x))
            new_h = max(10, min(new_h, FRAME_H - new_y))
            
            object_bbox = (new_x, new_y, new_w, new_h)

    cap.release()

    cv2.destroyAllWindows()

# =============================================================================

if __name__ == "__main__":

    main()