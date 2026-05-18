"""
Real-Time Hand & Object 6D Pose Estimation System
==================================================
Stack  : Python · OpenCV (contrib) · NumPy · SciPy
Models : Zero-download – skin-colour hand segmentation,
         ORB feature tracking, CSRT tracker, solvePnP
Target : Core i5 7th-Gen CPU · 8 GB RAM · 30+ FPS
"""

import cv2
import numpy as np
import time
import collections
import math
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CAM_IDX          = 0
FRAME_W, FRAME_H = 640, 480
FPS_TARGET       = 30

# Camera intrinsics (generic laptop webcam – adjust if you calibrate)
FOCAL            = 600.0
CAM_MATRIX       = np.array([[FOCAL, 0,     FRAME_W / 2],
                              [0,     FOCAL, FRAME_H / 2],
                              [0,     0,     1.0        ]], dtype=np.float64)
DIST_COEFFS      = np.zeros((4, 1), dtype=np.float64)

# Object half-size (metres) used for 3-D model points
OBJ_HALF = 0.05   # 10 cm cube

# 3-D box corners (8 pts) centred at origin
OBJ_3D = np.array([
    [-OBJ_HALF, -OBJ_HALF, -OBJ_HALF],
    [ OBJ_HALF, -OBJ_HALF, -OBJ_HALF],
    [ OBJ_HALF,  OBJ_HALF, -OBJ_HALF],
    [-OBJ_HALF,  OBJ_HALF, -OBJ_HALF],
    [-OBJ_HALF, -OBJ_HALF,  OBJ_HALF],
    [ OBJ_HALF, -OBJ_HALF,  OBJ_HALF],
    [ OBJ_HALF,  OBJ_HALF,  OBJ_HALF],
    [-OBJ_HALF,  OBJ_HALF,  OBJ_HALF],
], dtype=np.float64)

# 4 front-face corners for solvePnP (stable subset)
OBJ_3D_FRONT = OBJ_3D[:4]

# Axis length for pose axes drawing
AXIS_LEN = OBJ_HALF * 1.5


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────

C_GREEN   = (0,   230,  80)
C_CYAN    = (0,   220, 220)
C_ORANGE  = (0,   160, 255)
C_RED     = (0,    40, 220)
C_YELLOW  = (0,   230, 230)
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,     0,   0)
C_PURPLE  = (200,  40, 200)
C_BLUE    = (230, 100,   0)
C_MAGENTA = (200,   0, 200)


# ─────────────────────────────────────────────────────────────────────────────
# SKIN MASK  (HSV-based, fast)
# ─────────────────────────────────────────────────────────────────────────────

def skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Return binary mask of skin-coloured pixels (robust across tones)."""
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ycr  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)

    # HSV range for skin
    m1 = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
    # YCrCb range for skin
    m2 = cv2.inRange(ycr, (0, 133, 77), (255, 173, 127))

    mask = cv2.bitwise_and(m1, m2)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel7)
    mask = cv2.medianBlur(mask, 5)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# HAND DETECTOR  (contour + convex hull)
# ─────────────────────────────────────────────────────────────────────────────

class HandDetector:
    """Lightweight skin-colour hand detector with finger-count estimation."""

    MIN_AREA    = 5_000    # px²
    MAX_AREA    = 80_000

    def detect(self, frame_bgr: np.ndarray):
        """
        Returns:
            hand_bbox  : (x,y,w,h) or None
            centroid   : (cx,cy) or None
            fingers    : estimated finger count (0-5)
            hull_pts   : convex hull points (for drawing)
            contour    : largest skin contour
            mask       : skin mask
        """
        mask = skin_mask(frame_bgr)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None, 0, None, None, mask

        # pick the largest contour inside our area band
        valid = [c for c in cnts
                 if self.MIN_AREA < cv2.contourArea(c) < self.MAX_AREA]
        if not valid:
            return None, None, 0, None, None, mask

        cnt  = max(valid, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        M  = cv2.moments(cnt)
        if M["m00"] == 0:
            return None, None, 0, None, cnt, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        fingers = self._count_fingers(cnt, cx, cy, x, y, w, h)
        return (x, y, w, h), (cx, cy), fingers, hull, cnt, mask

    # ------------------------------------------------------------------ #
    def _count_fingers(self, cnt, cx, cy, x, y, w, h):
        """Rough finger count via convexity-defects (no model needed)."""
        try:
            hull_idx = cv2.convexHull(cnt, returnPoints=False)
            if hull_idx is None or len(hull_idx) < 3:
                return 0
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is None:
                return 0
            count = 0
            for d in defects:
                s, e, f, depth = d[0]
                depth /= 256.0
                if depth > 20:          # threshold in pixels
                    count += 1
            # defect count → finger count (heuristic)
            return min(count + 1, 5)
        except Exception:
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# ORB FEATURE TRACKER  (tracks object via keypoints)
# ─────────────────────────────────────────────────────────────────────────────

class ORBObjectTracker:
    """
    Detect + track an object using ORB features + BFMatcher.
    First frame when grab is detected → memorise reference descriptors.
    Subsequent frames → match + estimate homography → get 2-D corners.
    """

    def __init__(self):
        self.orb     = cv2.ORB_create(nfeatures=300, scaleFactor=1.2,
                                       nlevels=6, edgeThreshold=15)
        self.bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # CSRT for stable bounding-box tracking
        self.tracker = None
        self.state   = "idle"   # idle | tracking
        self._ref_kp    = None
        self._ref_desc  = None
        self._ref_bbox  = None   # (x,y,w,h) of reference frame
        self.bbox       = None   # current tracked bbox
        self.track_ok   = False
        self._lost_cnt  = 0

    # ------------------------------------------------------------------ #
    def init_tracking(self, frame_gray: np.ndarray, frame_bgr: np.ndarray,
                      obj_bbox):
        """Call when a grab is first detected to lock onto the object."""
        x, y, w, h = [int(v) for v in obj_bbox]
        roi = frame_gray[y:y+h, x:x+w]
        kp, desc = self.orb.detectAndCompute(roi, None)
        if desc is None or len(kp) < 6:
            return False

        self._ref_kp   = kp
        self._ref_desc = desc
        self._ref_bbox = (x, y, w, h)
        self.bbox      = (x, y, w, h)

        # Also start CSRT for robust tracking fallback
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame_bgr, (x, y, w, h))

        self.state    = "tracking"
        self.track_ok = True
        self._lost_cnt = 0
        return True

    # ------------------------------------------------------------------ #
    def update(self, frame_gray: np.ndarray, frame_bgr: np.ndarray):
        """Update tracker; returns (ok, bbox, matched_pts_img)."""
        if self.state != "tracking" or self.tracker is None:
            return False, None, None

        # Primary: CSRT
        ok, box = self.tracker.update(frame_bgr)
        if ok:
            x, y, w, h = [int(v) for v in box]
            self.bbox   = (x, y, w, h)
            self.track_ok = True
            self._lost_cnt = 0

            # ORB re-match inside current bbox for better accuracy
            matched_pts = self._orb_match_in_roi(frame_gray, x, y, w, h)
            return True, self.bbox, matched_pts
        else:
            self._lost_cnt += 1
            if self._lost_cnt > 10:
                self.state    = "idle"
                self.track_ok = False
            return False, self.bbox, None

    # ------------------------------------------------------------------ #
    def _orb_match_in_roi(self, gray, x, y, w, h):
        """Return ~4-8 matched image points for solvePnP."""
        if self._ref_desc is None:
            return None
        roi  = gray[max(0,y):y+h, max(0,x):x+w]
        kp2, desc2 = self.orb.detectAndCompute(roi, None)
        if desc2 is None or len(kp2) < 4:
            return None
        matches = self.bf.match(self._ref_desc, desc2)
        matches = sorted(matches, key=lambda m: m.distance)[:12]
        if len(matches) < 4:
            return None
        # Map back to full-frame coords
        pts = []
        for m in matches:
            kp = kp2[m.trainIdx]
            pts.append((kp.pt[0] + x, kp.pt[1] + y))
        return np.array(pts, dtype=np.float32)

    def reset(self):
        self.state    = "idle"
        self.tracker  = None
        self.track_ok = False
        self.bbox     = None
        self._lost_cnt = 0


# ─────────────────────────────────────────────────────────────────────────────
# 6-D POSE ESTIMATOR  (PnP)
# ─────────────────────────────────────────────────────────────────────────────

class PoseEstimator6D:
    """
    Estimates 6-D pose (rotation + translation) from a bounding box.
    Two modes:
      • bbox_mode   – fast IPPE-SQUARE from 4 bbox corners
      • keypoint_mode – solvePnPRansac from ORB matches
    Both produce rvec, tvec, and Euler angles.
    """

    def __init__(self, cam_matrix, dist_coeffs):
        self.K    = cam_matrix
        self.dist = dist_coeffs
        # Smoothing (exponential moving average)
        self._rvec_ema = None
        self._tvec_ema = None
        self.alpha      = 0.35   # EMA weight for new measurement

    # ------------------------------------------------------------------ #
    def estimate_from_bbox(self, bbox):
        """Use 4 bbox corners → solvePnP(IPPE_SQUARE) for planar pose."""
        x, y, w, h = bbox
        img_pts = np.array([
            [x,     y    ],
            [x + w, y    ],
            [x + w, y + h],
            [x,     y + h],
        ], dtype=np.float64)

        # Square planar target (front face of box)
        ret, rvec, tvec = cv2.solvePnP(
            OBJ_3D_FRONT, img_pts, self.K, self.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)

        if not ret:
            return None, None, None
        return self._smooth(rvec, tvec)

    # ------------------------------------------------------------------ #
    def estimate_from_keypoints(self, obj_pts_3d, img_pts_2d):
        """
        Use N matched keypoints.
        obj_pts_3d: (N,3) – 3-D points on the reference surface (rough).
        img_pts_2d: (N,2) – corresponding 2-D image points.
        """
        if len(img_pts_2d) < 4:
            return None, None, None
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts_3d, img_pts_2d, self.K, self.dist,
            iterationsCount=50, reprojectionError=6.0,
            flags=cv2.SOLVEPNP_EPNP)
        if not ret or inliers is None or len(inliers) < 4:
            return None, None, None
        return self._smooth(rvec, tvec)

    # ------------------------------------------------------------------ #
    def _smooth(self, rvec, tvec):
        if self._rvec_ema is None:
            self._rvec_ema = rvec.copy()
            self._tvec_ema = tvec.copy()
        else:
            self._rvec_ema = self.alpha * rvec + (1 - self.alpha) * self._rvec_ema
            self._tvec_ema = self.alpha * tvec + (1 - self.alpha) * self._tvec_ema

        euler = self._rvec_to_euler(self._rvec_ema)
        return self._rvec_ema, self._tvec_ema, euler

    def reset_smoothing(self):
        self._rvec_ema = None
        self._tvec_ema = None

    @staticmethod
    def _rvec_to_euler(rvec):
        R_mat, _ = cv2.Rodrigues(rvec)
        r = Rotation.from_matrix(R_mat)
        return r.as_euler('xyz', degrees=True)    # roll, pitch, yaw

    def project_axis(self, rvec, tvec, length=AXIS_LEN):
        """Project 3-D axis origin + 3 tips → 4 image points."""
        axis_pts = np.float64([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length],
        ])
        pts2d, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.K, self.dist)
        return pts2d.reshape(-1, 2).astype(int)

    def project_box(self, rvec, tvec):
        """Project 8 box corners → image."""
        pts2d, _ = cv2.projectPoints(OBJ_3D, rvec, tvec, self.K, self.dist)
        return pts2d.reshape(-1, 2).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# OBJECT DETECTOR  (motion-based + colour histogram)
# ─────────────────────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Detect the object held by the hand using:
      1. Look just below/around the hand centroid
      2. Find the dominant non-skin rectangular region
      3. Return a bounding box for the object
    """

    def __init__(self):
        self.bg_sub  = cv2.createBackgroundSubtractorMOG2(
                            history=200, varThreshold=40, detectShadows=False)
        self._last_bbox = None

    # ------------------------------------------------------------------ #
    def detect_near_hand(self, frame_bgr, hand_bbox, skin_mask_):
        """Return an object bbox near the hand (heuristic search)."""
        if hand_bbox is None:
            return None

        hx, hy, hw, hh = hand_bbox
        H, W = frame_bgr.shape[:2]

        # Search zone: expand hand bbox slightly downward
        sx  = max(0, hx - 20)
        sy  = max(0, hy - 20)
        ex  = min(W, hx + hw + 20)
        ey  = min(H, hy + hh + 40)

        roi_bgr  = frame_bgr[sy:ey, sx:ex]
        roi_skin = skin_mask_[sy:ey, sx:ex]

        # Non-skin region within ROI
        non_skin = cv2.bitwise_not(roi_skin)
        gray     = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Edge-based detection
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.bitwise_and(edges, non_skin)

        # Find contours in non-skin edge region
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return self._last_bbox

        # filter by area, pick largest
        valid = [c for c in cnts if 800 < cv2.contourArea(c) < 50_000]
        if not valid:
            return self._last_bbox

        best  = max(valid, key=cv2.contourArea)
        ox, oy, ow, oh = cv2.boundingRect(best)
        # Back to full-frame coords
        full_bbox = (sx + ox, sy + oy, ow, oh)
        self._last_bbox = full_bbox
        return full_bbox


# ─────────────────────────────────────────────────────────────────────────────
# GRAB DETECTOR  – decides whether the hand is grabbing
# ─────────────────────────────────────────────────────────────────────────────

class GrabDetector:
    """
    Heuristic grab detection:
      • Few fingers visible (fingers ≤ 2) AND object near hand centroid.
    Uses temporal smoothing to avoid flicker.
    """

    GRAB_FINGERS = 2      # ≤ this → "closed hand"
    PROXIMITY_PX = 80     # max distance hand–object centre

    def __init__(self, smoothing=5):
        self._history = collections.deque(maxlen=smoothing)

    def update(self, fingers, hand_centroid, obj_bbox):
        vote = False
        if fingers <= self.GRAB_FINGERS and hand_centroid and obj_bbox:
            ox, oy, ow, oh = obj_bbox
            obj_cx = ox + ow // 2
            obj_cy = oy + oh // 2
            dx = hand_centroid[0] - obj_cx
            dy = hand_centroid[1] - obj_cy
            dist = math.hypot(dx, dy)
            vote = dist < self.PROXIMITY_PX

        self._history.append(vote)
        # majority vote
        return sum(self._history) > len(self._history) // 2


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_hand(frame, contour, hull, centroid, fingers, grabbing):
    """Draw hand contour, hull, centroid, finger count."""
    if contour is not None:
        colour = C_ORANGE if grabbing else C_GREEN
        cv2.drawContours(frame, [contour], -1, colour, 2)
    if hull is not None:
        colour = C_YELLOW if grabbing else C_CYAN
        cv2.drawContours(frame, [hull], -1, colour, 2)
    if centroid:
        cx, cy = centroid
        cv2.circle(frame, (cx, cy), 8, C_WHITE, -1)
        cv2.circle(frame, (cx, cy), 8, C_GREEN, 2)
        label = f"Fingers: {fingers}"
        if grabbing:
            label += "  [GRAB]"
        cv2.putText(frame, label, (cx + 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 2, cv2.LINE_AA)


def draw_object_box(frame, bbox, colour=C_CYAN, label="Object"):
    """Draw a labelled bounding box."""
    if bbox is None:
        return
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    # Corner accents
    lc = 12
    thick = 3
    for (rx, ry) in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
        sx = 1 if rx == x else -1
        sy = 1 if ry == y else -1
        cv2.line(frame, (rx, ry), (rx + sx * lc, ry),       colour, thick)
        cv2.line(frame, (rx, ry), (rx, ry + sy * lc),       colour, thick)

    # Label
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), colour, -1)
    cv2.putText(frame, label, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_BLACK, 1, cv2.LINE_AA)


def draw_pose_axes(frame, pts):
    """Draw 3 coloured pose axes. pts = [origin, x_tip, y_tip, z_tip]."""
    if pts is None:
        return
    o, xp, yp, zp = pts
    H, W = frame.shape[:2]

    def clamp(pt):
        return (int(np.clip(pt[0], 0, W-1)), int(np.clip(pt[1], 0, H-1)))

    cv2.arrowedLine(frame, clamp(o), clamp(xp), (0,   0, 255), 3, tipLength=0.3)
    cv2.arrowedLine(frame, clamp(o), clamp(yp), (0, 200,   0), 3, tipLength=0.3)
    cv2.arrowedLine(frame, clamp(o), clamp(zp), (255, 0,   0), 3, tipLength=0.3)


def draw_pose_wire_box(frame, corners2d, colour=C_PURPLE):
    """Draw projected 3-D bounding box wireframe."""
    if corners2d is None:
        return
    H, W = frame.shape[:2]
    edges = [(0,1),(1,2),(2,3),(3,0),    # back face
             (4,5),(5,6),(6,7),(7,4),    # front face
             (0,4),(1,5),(2,6),(3,7)]    # connecting

    def cp(i):
        x = int(np.clip(corners2d[i][0], 0, W-1))
        y = int(np.clip(corners2d[i][1], 0, H-1))
        return (x, y)

    for a, b in edges:
        cv2.line(frame, cp(a), cp(b), colour, 1, cv2.LINE_AA)


def draw_info_panel(frame, fps, state, euler, tvec):
    """Overlay HUD panel with system state and pose values."""
    H, W = frame.shape[:2]
    panel_h = 150
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - panel_h), (320, H), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y0 = H - panel_h + 18
    dy = 22

    def txt(msg, row, col=C_WHITE, scale=0.48):
        cv2.putText(frame, msg, (10, y0 + row * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)

    txt(f"FPS: {fps:4.1f}    State: {state.upper()}", 0,
        C_GREEN if state == "tracking" else C_ORANGE, 0.52)

    if euler is not None:
        r, p, w_ = euler
        txt(f"Roll : {r:+7.1f} deg", 1, C_CYAN)
        txt(f"Pitch: {p:+7.1f} deg", 2, C_CYAN)
        txt(f"Yaw  : {w_:+7.1f} deg", 3, C_CYAN)

    if tvec is not None:
        x_, y_, z_ = tvec.flatten()
        txt(f"T  ({x_*100:+5.1f}, {y_*100:+5.1f}, {z_*100:+5.1f}) cm", 4,
            C_YELLOW)

    # Mini legend
    legend_y = 20
    for col, label in [(C_RED, "X axis"), (C_GREEN, "Y axis"), (C_BLUE, "Z axis")]:
        cv2.circle(frame, (W - 120, legend_y), 5, col, -1)
        cv2.putText(frame, label, (W - 110, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        legend_y += 18


def draw_fps_bar(frame, fps, target=FPS_TARGET):
    """Top-right mini performance bar."""
    W = frame.shape[1]
    pct  = min(fps / target, 1.0)
    col  = C_GREEN if pct > 0.7 else (C_YELLOW if pct > 0.4 else C_RED)
    bw   = 80
    cv2.rectangle(frame, (W - bw - 10, 8), (W - 10, 22), (40, 40, 40), -1)
    cv2.rectangle(frame, (W - bw - 10, 8), (W - bw - 10 + int(bw * pct), 22), col, -1)
    cv2.putText(frame, f"{fps:.0f}fps", (W - bw - 10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# 3-D OBJECT POINTS FOR KEYPOINT-BASED PnP
# (map 2-D matched points → approx 3-D positions on front face)
# ─────────────────────────────────────────────────────────────────────────────

def build_obj_pts_for_matches(img_pts_2d, ref_bbox):
    """
    Map N 2-D image points on the reference bounding-box face
    to their approximate 3-D positions on OBJ_3D_FRONT plane.
    """
    rx, ry, rw, rh = ref_bbox
    # Normalise to [−half, +half] in X,Y  ;  Z = 0 (planar assumption)
    obj_pts = []
    for (ix, iy) in img_pts_2d:
        nx = (ix - (rx + rw / 2)) / (rw / 2) * OBJ_HALF
        ny = (iy - (ry + rh / 2)) / (rh / 2) * OBJ_HALF
        obj_pts.append([nx, ny, 0.0])
    return np.array(obj_pts, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(CAM_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          FPS_TARGET)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAM_IDX in config.")
        return

    hand_det  = HandDetector()
    obj_det   = ObjectDetector()
    orb_track = ORBObjectTracker()
    pose_est  = PoseEstimator6D(CAM_MATRIX, DIST_COEFFS)
    grab_det  = GrabDetector()

    # State
    state        = "detecting"   # detecting | grabbing | tracking
    obj_bbox_det = None          # detection-level bbox
    rvec         = None
    tvec         = None
    euler        = None
    fps          = 0.0
    fps_alpha    = 0.2

    print("=" * 58)
    print("  Hand + Object 6D Pose Estimation – Real-Time System")
    print("=" * 58)
    print("  Controls:")
    print("    R     – reset tracker")
    print("    Q/ESC – quit")
    print("  Tip: Hold an object in your fist to trigger pose.")
    print("=" * 58)

    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed – retrying…")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)   # mirror for natural interaction
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── FPS ──────────────────────────────────────────────────── #
        t_now = time.perf_counter()
        fps   = fps_alpha / (t_now - t_prev) + (1 - fps_alpha) * fps
        t_prev = t_now

        # ── HAND DETECTION ───────────────────────────────────────── #
        hand_bbox, centroid, fingers, hull, contour, smask = \
            hand_det.detect(frame)

        # ── OBJECT DETECTION ─────────────────────────────────────── #
        obj_bbox_det = obj_det.detect_near_hand(frame, hand_bbox, smask)

        # ── GRAB DETECTION ───────────────────────────────────────── #
        grabbing = grab_det.update(fingers, centroid, obj_bbox_det)

        # ── STATE MACHINE ────────────────────────────────────────── #
        if state == "detecting":
            if grabbing and obj_bbox_det is not None:
                ok = orb_track.init_tracking(gray, frame, obj_bbox_det)
                if ok:
                    state = "tracking"
                    pose_est.reset_smoothing()
                    print("[INFO] Grab detected – tracking started.")

        elif state == "tracking":
            if not grabbing:
                # Hand released
                orb_track.reset()
                pose_est.reset_smoothing()
                rvec, tvec, euler = None, None, None
                state = "detecting"
            else:
                ok, tracked_bbox, matched_pts = orb_track.update(gray, frame)

                if ok and tracked_bbox is not None:
                    obj_bbox_det = tracked_bbox

                    # ── 6-D POSE ─────────────────────────────── #
                    if (matched_pts is not None and
                            len(matched_pts) >= 4 and
                            orb_track._ref_bbox is not None):
                        obj3d = build_obj_pts_for_matches(
                            matched_pts, orb_track._ref_bbox)
                        rv, tv, eu = pose_est.estimate_from_keypoints(
                            obj3d, matched_pts)
                        if rv is not None:
                            rvec, tvec, euler = rv, tv, eu
                        else:
                            rv, tv, eu = pose_est.estimate_from_bbox(tracked_bbox)
                            if rv is not None:
                                rvec, tvec, euler = rv, tv, eu
                    else:
                        rv, tv, eu = pose_est.estimate_from_bbox(tracked_bbox)
                        if rv is not None:
                            rvec, tvec, euler = rv, tv, eu
                else:
                    # Tracker lost
                    orb_track.reset()
                    rvec, tvec, euler = None, None, None
                    state = "detecting"

        # ── DRAWING ──────────────────────────────────────────────── #

        # 1. Hand
        draw_hand(frame, contour, hull, centroid, fingers, grabbing)

        # 2. Object bounding box
        if obj_bbox_det is not None:
            col   = C_ORANGE if grabbing else C_CYAN
            label = "Object [grabbed]" if grabbing else "Object"
            draw_object_box(frame, obj_bbox_det, col, label)

        # 3. 6-D Pose: projected wire box + axes
        if rvec is not None and tvec is not None:
            corners2d = pose_est.project_box(rvec, tvec)
            draw_pose_wire_box(frame, corners2d)

            axis_pts = pose_est.project_axis(rvec, tvec)
            draw_pose_axes(frame, axis_pts)

        # 4. State badge
        badge_col = {
            "detecting": C_YELLOW,
            "tracking":  C_GREEN,
        }.get(state, C_WHITE)
        badge_txt = {
            "detecting": "DETECTING HAND & OBJECT",
            "tracking":  "TRACKING – 6D POSE ACTIVE",
        }.get(state, state.upper())

        cv2.putText(frame, badge_txt, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_col, 2, cv2.LINE_AA)

        # Hint text
        if state == "detecting":
            cv2.putText(frame, "Close your fist around an object to grab",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1, cv2.LINE_AA)

        # 5. Info panel (bottom-left)
        draw_info_panel(frame, fps, state, euler, tvec)

        # 6. FPS bar (top-right)
        draw_fps_bar(frame, fps)

        cv2.imshow("Hand & Object 6D Pose — press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            orb_track.reset()
            pose_est.reset_smoothing()
            rvec, tvec, euler = None, None, None
            state = "detecting"
            print("[INFO] Tracker reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
