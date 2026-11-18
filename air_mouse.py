"""Air mouse: control your computer with hand gestures through the webcam.

Modes (only the active mode acts on hand data):
  1 = MOUSE   index finger moves the cursor, pinch thumb + index to click / drag
  2 = VOLUME  pinch to grab a virtual knob, rotate your hand to turn it (b = volume/brightness)
  3 = DRAW    placeholder, not implemented yet

Keys (with the OpenCV window focused):  1/2/3 = switch mode, p = pause/resume, q or Esc = quit.
Emergency stop: slam the real mouse/trackpad cursor into a screen corner (pyautogui FAILSAFE).
"""

import csv
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# =============================================================================
# CONFIG: everything you might want to tune lives here
# =============================================================================

# --- Modes ---
START_MODE = "MOUSE"
MODE_KEYS = {"1": "MOUSE", "2": "VOLUME", "3": "DRAW", "0": "SCROLL"}  # key -> mode name
# The gesture menu picks the Nth mode in this order: 1-3 fingers, or 4 (open palm) = SCROLL.

# --- Camera ---
CAMERA_INDEX = 0            # 0 = built-in webcam; try 1, 2 for external / Continuity cameras
CAMERA_WIDTH = 640          # lower resolution = faster detection
CAMERA_HEIGHT = 480
MIRROR = True               # flip horizontally so moving right moves the cursor right

# --- Active region (fraction of the frame trimmed off each side) ---
# The inner rectangle is mapped to the full screen, so you can reach the
# edges without your fingertip leaving the camera view.
REGION_MARGIN_X = 0.18      # 0.18 on each side -> 64% of the frame width is active
REGION_MARGIN_TOP = 0.12    # less margin on top: hands rarely go above the frame
REGION_MARGIN_BOTTOM = 0.28 # more margin at the bottom: your wrist/arm sits there
# These margins are the default. Calibration (press k in MOUSE mode) replaces them
# with a region measured from your own reach, saved in calibration.json.

# --- Calibration (MOUSE mode: k = calibrate, r = reset to defaults, Esc = cancel) ---
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
CAL_CORNER_S = 2.5          # time spent on each of the four corners
CAL_IGNORE_S = 0.5          # ignore this much at the start of each corner (time to move there)
CAL_MIN_SAMPLES = 10        # fewer hand samples than this in a corner window = retry ("no hand")
CAL_MAX_SCATTER = 0.05      # 80% of samples must lie within this distance of their median
                            # (fraction of frame height), or the corner is retried ("hold still")
CAL_PADDING = 0.05          # shrink the measured region by this fraction of its size on each side,
                            # so you hit the screen edge slightly before your calibration point
CAL_MIN_SIZE = 0.15         # reject a region smaller than this fraction of the frame in either axis
LOCK_ASPECT_RATIO = False   # True = grow the region around its center to match the screen's shape

# --- Clutch (MOUSE mode: make a fist to "lift the mouse"; x toggles) ---
# Fist = cursor freezes. Open the hand = the cursor continues from where it froze,
# via an offset added to the fingertip mapping (so it never teleports).
# In MOUSE mode with the clutch on, a fist on the controlling hand is the clutch,
# not the quit gesture; quit with the other hand's fist or q.
CLUTCH_ENABLED = True
CLUTCH_CURL_RATIO = 1.0     # a finger is curled when |wrist->tip| < |wrist->PIP| * this; fist = all 4 curled
CLUTCH_OPEN_RATIO = 1.1     # the hand counts as open again once any finger is beyond this ratio
CLUTCH_ENGAGE_FRAMES = 3    # fist must hold this many frames to engage
CLUTCH_RELEASE_FRAMES = 3   # open hand must hold this many frames to release
CLUTCH_BLEED_RATE = 0.0008  # while the hand moves toward the side the offset cut off, that axis of the
                            # offset shrinks by exp(-rate * px moved): the cursor runs a little faster
                            # that way until the edge is reachable again. Moving the other way is 1:1.
                            # 0 = keep the offset until a reset.
CLUTCH_LOST_RESET_S = 2.0   # hand gone longer than this = drop the offset and the clutch

# --- Scroll (MOUSE mode: peace sign = scroll) ---
# Hold index + middle up (ring + pinky curled) and move the hand vertically.
# The cursor holds still while you scroll, then continues from the same spot.
SCROLL_ENABLED = True
SCROLL_SPEED = 60.0         # scroll lines per full frame-height of hand travel
SCROLL_DIRECTION = 1        # 1 = hand up scrolls down, hand down scrolls up; -1 = reverse
SCROLL_DEADZONE = 0.002     # ignore per-frame hand movement smaller than this (fraction of frame height)
SCROLL_SMOOTH_ALPHA = 0.5   # EMA weight for the hand position while scrolling
# (SCROLL_SPEED / DIRECTION / DEADZONE / SMOOTH_ALPHA also drive manual scrolling in SCROLL mode.)

# --- SCROLL mode (key 0, or an open palm on the menu hand) ---
# Index finger only. Tilted up = auto-scroll up, tilted down = auto-scroll down,
# roughly level = manual (move the hand: up scrolls down, down scrolls up).
AUTO_SCROLL_ANGLE = 35.0    # tilt above/below level (degrees) that starts auto-scrolling
AUTO_SCROLL_EXIT_ANGLE = 25.0  # tilt back under this to return to manual (hysteresis)
AUTO_SCROLL_MIN_SPEED = 4.0 # lines/s right at AUTO_SCROLL_ANGLE
AUTO_SCROLL_MAX_SPEED = 30.0  # lines/s pointing straight up or down
AUTO_SCROLL_DIRECTION = 1   # 1 = pointing up scrolls up; -1 = reverse

# --- Fatigue tracker (all modes; f = on/off, n = snooze) ---
# "Hand detected" stands in for "arm raised" (clutch and PAUSED count as raised).
FATIGUE_ENABLED = True
FATIGUE_GRACE_S = 3.0       # tracking dropouts shorter than this still count as raised
REST_RESET_SECONDS = 20.0   # hand gone this long = a real rest: the timer and nudges reset
NUDGE_LEVEL1_S = 10 * 60    # first (gentle) notification - nothing before this
NUDGE_LEVEL2_S = 15 * 60    # firmer notification
NUDGE_LEVEL3_S = 20 * 60    # persistent reminder, repeated every NUDGE_REPEAT_S after
NUDGE_REPEAT_S = 5 * 60
SNOOZE_S = 5 * 60           # n snoozes nudges this long
NUDGE_SOUND = False         # play a sound with desktop notifications
FATIGUE_DEBUG_SPEEDUP = 1   # divides REST_RESET_SECONDS, the nudge times and SNOOZE_S (e.g. 30 to test
                            # in seconds). Not the grace period: that's about camera dropouts.
FATIGUE_MIN_LOG_S = 5.0     # sessions with less raised time than this aren't logged (stray detections)
FATIGUE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fatigue_log.csv")

# --- Smoothing (adaptive exponential moving average) ---
# alpha = weight of the new sample. Low alpha = smoother but laggier.
# The alpha slides from SMOOTH_ALPHA_SLOW (hand nearly still -> precise) to
# SMOOTH_ALPHA_FAST (hand moving quickly -> responsive).
SMOOTH_ALPHA_SLOW = 0.12
SMOOTH_ALPHA_FAST = 0.65
SPEED_SLOW = 150.0          # screen px/sec at or below which ALPHA_SLOW applies
SPEED_FAST = 2500.0         # screen px/sec at or above which ALPHA_FAST applies
DEADZONE_PX = 1.5           # ignore smoothed moves smaller than this (kills sub-pixel shimmer)

# --- Pinch detection (distances are normalized by hand size) ---
# hand size = distance from wrist (0) to middle-finger knuckle (9)
PINCH_START = 0.28          # thumb-index distance below this -> pinch begins
PINCH_RELEASE = 0.40        # distance above this -> pinch ends (must be > PINCH_START)

# --- Click / drag timing ---
DRAG_DELAY_MS = 300         # hold a pinch longer than this to start a drag
RELEASE_FREEZE_MS = 120     # keep the cursor frozen briefly after a click release
DRAG_ALPHA_SCALE = 0.5      # while dragging, multiply alpha by this (extra damping)

# --- Knob (VOLUME mode) ---
KNOB_TARGET = "volume"      # "volume" or "brightness"; press b in VOLUME mode to toggle
KNOB_DEGREES_FOR_FULL = 180.0  # hand rotation that sweeps 0 -> 100%
KNOB_DIRECTION = 1          # 1 = rotating clockwise (as seen in the window) turns it up; -1 = reverse
KNOB_ANGLE_ALPHA = 0.35     # EMA weight for the hand angle; lower = smoother, laggier knob
KNOB_MIN_STEP = 1.0         # only send a new value when it changes by at least this many %
KNOB_POLL_S = 1.0           # while released, re-read the value this often (catches keyboard changes)

# --- Circle gesture (VOLUME mode) ---
# Draw circles in the air with only your index finger up: clockwise (as seen in
# the window) = up, counter-clockwise = down. No pinch needed.
CIRCLE_PERCENT_PER_TURN = 20.0  # one full circle changes the value by this much
CIRCLE_DIRECTION = 1        # 1 = clockwise turns it up; -1 = reverse
CIRCLE_WINDOW_S = 0.8       # the circle's center is the average fingertip position over this window
CIRCLE_MIN_RADIUS = 0.025   # fingertip must be this far from that center (fraction of frame height)
CIRCLE_DEADBAND_DEG = 45.0  # ignore the first this-many degrees of rotation (filters out wobble)
CIRCLE_MAX_STEP_DEG = 60.0  # ignore single-frame jumps bigger than this (tracking glitches)
CIRCLE_TIP_ALPHA = 0.5      # EMA weight for the fingertip before measuring the circle

# --- Screen ---
# None = use the primary display size reported by pyautogui.
# Otherwise (left, top, width, height) in pyautogui coordinates, e.g. to target
# a specific monitor or a corrected size on a misreporting high-DPI setup.
SCREEN_REGION = None
EDGE_INSET_PX = 2           # keep the hand-driven cursor off the exact corners so it
                            # never trips FAILSAFE by itself (only you can)

# --- Finger poses (DRAW mode and the gesture menu) ---
# A finger counts as extended when its tip is this much farther from the wrist
# than its middle (PIP) joint. Distances from the wrist don't change when the
# hand rotates, so this works for tilted and sideways hands too.
FINGER_EXTEND_RATIO = 1.15
POSE_STABLE_FRAMES = 3      # a new pose must hold this many frames before it counts

# --- Gesture menu (switch modes with your other hand) ---
# With two hands in view, the hand you've been using stays the controlling hand.
# Hold up 1, 2, 3 fingers (index, +middle, +ring) or an open palm (4) on the other hand to pick
# the mode with that number in MODE_KEYS. Either hand can be the menu hand.
GESTURE_MENU = True
MENU_HOLD_S = 0.8           # hold the finger count this long to switch (progress ring shows it)

# --- Quit gesture ---
# Make a fist (all four fingers curled) with either hand and hold it to quit.
# Ignored while the controlling hand is pinching (click / drag / knob grab) and while paused.
QUIT_GESTURE = True
QUIT_HOLD_S = 1.5           # hold the fist this long (a red countdown ring shows it)
QUIT_GRACE_S = 0.15         # a fist that drops out for less than this doesn't restart the countdown

# --- Draw (DRAW mode) ---
# Pen down = only the index finger up. Pen up = index + middle (peace sign).
# Open palm held still for CLEAR_HOLD_S = clear the canvas.
DRAW_COLORS = {             # key -> (name, BGR)
    "4": ("RED", (40, 40, 255)),
    "5": ("GREEN", (60, 220, 60)),
    "6": ("BLUE", (255, 120, 30)),
    "7": ("YELLOW", (0, 230, 255)),
    "8": ("WHITE", (255, 255, 255)),
}
DRAW_START_COLOR = "4"
DRAW_BRUSH_SIZE = 6         # px; [ and ] change it
DRAW_BRUSH_MIN, DRAW_BRUSH_MAX, DRAW_BRUSH_STEP = 2, 40, 2
DRAW_SPEED_SCALE = 2.4      # frame px are ~2.4x smaller than screen px; scales the
                            # adaptive smoothing's speed thresholds to match MOUSE feel
DRAW_MIN_SEGMENT_PX = 1.5   # skip points closer than this to the last one
DRAW_OPACITY = 0.9          # how strongly strokes cover the webcam feed (0-1)
CLEAR_HOLD_S = 1.0
CLEAR_MAX_MOVE = 0.05       # palm drifting more than this (fraction of frame) restarts the countdown
DRAW_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drawings")

# --- Debug window ---
SHOW_DEBUG_WINDOW = True    # False = small status-only window (still needed for keys)
NUM_HANDS = 2               # 2 = needed for the gesture menu; 1 = slightly faster
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

# =============================================================================

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
WINDOW_NAME = "Air Mouse"
GREY, WHITE = (128, 128, 128), (255, 255, 255)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # default is 0.1 s after every call, which would cap us at ~10 fps


# ----------------------------------------------------------------------------- setup

def enable_windows_dpi_awareness():
    """On Windows with display scaling, make pyautogui see real pixels."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass


def get_screen_region():
    if SCREEN_REGION is not None:
        return SCREEN_REGION
    w, h = pyautogui.size()
    return (0, 0, w, h)


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading hand landmark model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera():
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    ok, _ = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        msg = f"Could not read from camera {CAMERA_INDEX}."
        if sys.platform == "darwin":
            msg += (" On macOS, grant Camera access to your terminal app in System Settings >"
                    " Privacy & Security > Camera, then restart the terminal.")
        sys.exit(msg)
    return cap


def get_hands(landmarker, frame_bgr, timestamp_ms):
    """Return a list (0 to NUM_HANDS long) of 21-landmark lists, normalized coords."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return list(landmarker.detect_for_video(image, timestamp_ms).hand_landmarks)


class HandRouter:
    """Split detected hands into (controlling hand, menu hand).

    One hand: it controls. Two hands: the one nearest where the controlling hand
    was last frame keeps control, so either hand can be the menu hand.
    """

    def __init__(self):
        self.last_wrist = None

    def split(self, hands):
        if not hands:
            self.last_wrist = None
            return None, None
        if len(hands) == 1 or self.last_wrist is None:
            primary = hands[0]
        else:
            lx, ly = self.last_wrist
            primary = min(hands, key=lambda h: math.hypot(h[WRIST].x - lx, h[WRIST].y - ly))
        other = next((h for h in hands if h is not primary), None)
        self.last_wrist = (primary[WRIST].x, primary[WRIST].y)
        return primary, other


# ----------------------------------------------------------------------------- shared helpers

def default_region():
    """The margin-based region as normalized (x1, y1, x2, y2)."""
    return (REGION_MARGIN_X, REGION_MARGIN_TOP, 1 - REGION_MARGIN_X, 1 - REGION_MARGIN_BOTTOM)


def active_region_px(frame_w, frame_h, region=None):
    if region is not None:
        x1, y1, x2, y2 = region
        return int(frame_w * x1), int(frame_h * y1), int(frame_w * x2), int(frame_h * y2)
    x1 = int(frame_w * REGION_MARGIN_X)
    x2 = int(frame_w * (1 - REGION_MARGIN_X))
    y1 = int(frame_h * REGION_MARGIN_TOP)
    y2 = int(frame_h * (1 - REGION_MARGIN_BOTTOM))
    return x1, y1, x2, y2


def map_to_screen(nx, ny, screen, region=None):
    """Map a normalized frame point inside the active region to screen coordinates.

    region = normalized (x1, y1, x2, y2); None = the default margins.
    """
    left, top, sw, sh = screen
    if region is not None:
        x1, y1, x2, y2 = region
        rx = (nx - x1) / (x2 - x1)
        ry = (ny - y1) / (y2 - y1)
    else:
        rx = (nx - REGION_MARGIN_X) / (1 - 2 * REGION_MARGIN_X)
        ry = (ny - REGION_MARGIN_TOP) / (1 - REGION_MARGIN_TOP - REGION_MARGIN_BOTTOM)
    rx = min(max(rx, 0.0), 1.0)
    ry = min(max(ry, 0.0), 1.0)
    x = left + EDGE_INSET_PX + rx * (sw - 1 - 2 * EDGE_INSET_PX)
    y = top + EDGE_INSET_PX + ry * (sh - 1 - 2 * EDGE_INSET_PX)
    return x, y


def adaptive_alpha(speed_px_s):
    t = (speed_px_s - SPEED_SLOW) / (SPEED_FAST - SPEED_SLOW)
    t = min(max(t, 0.0), 1.0)
    return SMOOTH_ALPHA_SLOW + t * (SMOOTH_ALPHA_FAST - SMOOTH_ALPHA_SLOW)


def smooth(prev, target, dt, alpha_scale=1.0, speed_scale=1.0):
    """Adaptive EMA: more smoothing when slow, less when fast.

    speed_scale converts the caller's units to screen px for the speed thresholds.
    """
    if prev is None:
        return target
    dx, dy = target[0] - prev[0], target[1] - prev[1]
    speed = math.hypot(dx, dy) / max(dt, 1e-3) * speed_scale
    a = adaptive_alpha(speed) * alpha_scale
    return (prev[0] + a * dx, prev[1] + a * dy)


def pinch_distance(lm):
    """Thumb-index distance divided by hand size, so it's independent of camera distance."""
    def dist(a, b):
        return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)
    hand_size = dist(WRIST, MIDDLE_MCP)
    if hand_size < 1e-6:
        return None
    return dist(THUMB_TIP, INDEX_TIP) / hand_size


def detect_pinch(dist, was_pinched):
    """Hysteresis: separate thresholds for entering and leaving the pinch."""
    if dist is None:
        return was_pinched
    if was_pinched:
        return dist < PINCH_RELEASE
    return dist < PINCH_START


def hand_angle(lm):
    """Angle in degrees of the wrist -> middle-knuckle vector (image coords, y down).

    Increases as the hand rotates clockwise on screen. Range (-180, 180].
    """
    return math.degrees(math.atan2(lm[MIDDLE_MCP].y - lm[WRIST].y, lm[MIDDLE_MCP].x - lm[WRIST].x))


def wrap_degrees(d):
    """Shortest signed difference, so 179 -> -179 counts as +2, not -358."""
    return (d + 180.0) % 360.0 - 180.0


FINGER_JOINTS = ((8, 6), (12, 10), (16, 14), (20, 18))  # (tip, PIP) for index, middle, ring, pinky


def extended_fingers(lm, aspect=4 / 3):
    """(index, middle, ring, pinky) extended flags. Rotation-invariant: compares
    wrist->tip with wrist->PIP distance. aspect = frame width / height, so x and
    y distances are in the same units. The thumb is ignored (too unreliable)."""
    def from_wrist(i):
        return math.hypot((lm[i].x - lm[WRIST].x) * aspect, lm[i].y - lm[WRIST].y)
    return tuple(from_wrist(tip) > from_wrist(pip) * FINGER_EXTEND_RATIO for tip, pip in FINGER_JOINTS)


POSES = {
    (True, False, False, False): "POINT",   # index only
    (True, True, False, False): "PEACE",    # index + middle
    (True, True, True, False): "THREE",     # index + middle + ring
    (True, True, True, True): "PALM",       # all four
    (False, False, False, False): "FIST",   # none
}


def finger_ratios(lm, aspect=4 / 3):
    """|wrist->tip| / |wrist->PIP| for index, middle, ring, pinky. Rotation-invariant."""
    def from_wrist(i):
        return math.hypot((lm[i].x - lm[WRIST].x) * aspect, lm[i].y - lm[WRIST].y)
    return tuple(from_wrist(tip) / max(from_wrist(pip), 1e-6) for tip, pip in FINGER_JOINTS)


def is_fist(lm, aspect=4 / 3):
    return all(r < CLUTCH_CURL_RATIO for r in finger_ratios(lm, aspect))


def is_open_hand(lm, aspect=4 / 3):
    return any(r > CLUTCH_OPEN_RATIO for r in finger_ratios(lm, aspect))


class ClutchDetector:
    """Debounced fist detector with separate engage / release conditions (hysteresis)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.engaged = False
        self._count = 0

    def update(self, lm, aspect):
        """Return "engage", "release" or None."""
        if not self.engaged:
            self._count = self._count + 1 if is_fist(lm, aspect) else 0
            if self._count >= CLUTCH_ENGAGE_FRAMES:
                self.engaged, self._count = True, 0
                return "engage"
        else:
            self._count = self._count + 1 if is_open_hand(lm, aspect) else 0
            if self._count >= CLUTCH_RELEASE_FRAMES:
                self.engaged, self._count = False, 0
                return "release"
        return None


def classify_pose(lm, aspect=4 / 3):
    return POSES.get(extended_fingers(lm, aspect), "OTHER")


class PoseTracker:
    """Debounce: report a new pose only after it holds for POSE_STABLE_FRAMES frames."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.stable, self._candidate, self._count = "NONE", None, 0

    def update(self, pose):
        if pose == self.stable:
            self._candidate, self._count = None, 0
        elif pose == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = pose, 1
        if self._count >= POSE_STABLE_FRAMES:
            self.stable, self._candidate, self._count = pose, None, 0
        return self.stable


# ----------------------------------------------------------------------------- system volume / brightness
# Each backend is a (get_volume, set_volume) pair working in 0-100. Swap one out here.

def _applescript_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def send_notification(title, body, sound=False):
    """Show a desktop notification. Raises with a reason if it can't. Never steals
    focus or moves the cursor (these are passive OS notifications)."""
    if sys.platform == "darwin":
        script = f"display notification {_applescript_str(body)} with title {_applescript_str(title)}"
        if sound:
            script += ' sound name "Glass"'
        _run(["osascript", "-e", script])
    elif sys.platform == "win32":
        try:
            from plyer import notification
            notification.notify(title=title, message=body, app_name="Air Mouse", timeout=8)
        except ImportError:
            from win10toast import ToastNotifier
            ToastNotifier().show_toast(title, body, duration=8, threaded=True)
        if sound:
            import winsound
            winsound.MessageBeep()
    elif sys.platform.startswith("linux"):
        if not shutil.which("notify-send"):
            raise RuntimeError("notify-send not found (install libnotify-bin)")
        _run(["notify-send", "-a", "Air Mouse", title, body])
        if sound and shutil.which("canberra-gtk-play"):
            _run(["canberra-gtk-play", "-i", "message"])
    else:
        raise RuntimeError(f"no notification method for {sys.platform}")


def notification_problem():
    """Cheap startup check (sends nothing). Returns a reason string, or None if it should work."""
    if sys.platform == "darwin":
        return None if shutil.which("osascript") else "osascript not found"
    if sys.platform == "win32":
        for mod in ("plyer", "win10toast"):
            try:
                __import__(mod)
                return None
            except ImportError:
                pass
        return "neither plyer nor win10toast is installed (pip install plyer)"
    if sys.platform.startswith("linux"):
        return None if shutil.which("notify-send") else "notify-send not found (install libnotify-bin)"
    return f"no notification method for {sys.platform}"


class Notifier:
    """Sends notifications on a background thread so the camera loop never waits."""

    def __init__(self, sender=send_notification):
        self.sender = sender
        self.error = notification_problem() if sender is send_notification else None
        if self.error:
            print(f"Desktop notifications unavailable: {self.error}. Nudges will show in the window only.")
        self.sent = 0
        self._q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def notify(self, title, body):
        if not self.error:
            self._q.put((title, body))     # returns immediately

    def _worker(self):
        while True:
            title, body = self._q.get()
            try:
                self.sender(title, body, NUDGE_SOUND)
                self.sent += 1
            except Exception as e:
                if not self.error:
                    print(f"Desktop notification failed: {e}. Nudges will show in the window only.")
                self.error = str(e) or type(e).__name__


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=True).stdout


def _macos_volume():
    def get_volume():
        return float(_run(["osascript", "-e", "output volume of (get volume settings)"]).strip())

    def set_volume(v):
        _run(["osascript", "-e", f"set volume output volume {int(round(v))}"])
    return get_volume, set_volume


def _windows_volume():
    from pycaw.pycaw import AudioUtilities

    def endpoint():
        speakers = AudioUtilities.GetSpeakers()
        if hasattr(speakers, "EndpointVolume"):          # pycaw >= 20240210
            return speakers.EndpointVolume
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import IAudioEndpointVolume
        iface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(iface, POINTER(IAudioEndpointVolume))

    def get_volume():
        return endpoint().GetMasterVolumeLevelScalar() * 100

    def set_volume(v):
        endpoint().SetMasterVolumeLevelScalar(v / 100, None)
    return get_volume, set_volume


def _linux_volume():
    if shutil.which("pactl"):
        def get_volume():
            return float(re.search(r"(\d+)%", _run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])).group(1))

        def set_volume(v):
            _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(round(v))}%"])
        return get_volume, set_volume
    if shutil.which("amixer"):
        def get_volume():
            return float(re.search(r"\[(\d+)%\]", _run(["amixer", "get", "Master"])).group(1))

        def set_volume(v):
            _run(["amixer", "-q", "set", "Master", f"{int(round(v))}%"])
        return get_volume, set_volume
    raise RuntimeError("neither pactl nor amixer found")


def make_volume_backend():
    """Return (get_volume, set_volume, description) or raise with the reason it's unavailable."""
    if sys.platform == "darwin":
        return (*_macos_volume(), "macOS osascript")
    if sys.platform == "win32":
        return (*_windows_volume(), "Windows pycaw")
    if sys.platform.startswith("linux"):
        get, set_ = _linux_volume()
        return get, set_, "Linux " + ("pactl" if shutil.which("pactl") else "amixer")
    raise RuntimeError(f"unsupported OS {sys.platform}")


def make_brightness_backend():
    """screen_brightness_control on Windows/Linux. Deliberately disabled on macOS."""
    if sys.platform == "darwin":
        raise RuntimeError("screen_brightness_control doesn't support macOS")
    import screen_brightness_control as sbc

    def get_brightness():
        return float(sbc.get_brightness(display=0)[0])

    def set_brightness(v):
        sbc.set_brightness(int(round(v)), display=0)
    get_brightness()  # fail now (no supported display, permissions) rather than mid-turn
    return get_brightness, set_brightness, "screen_brightness_control"


class KnobTarget:
    """A 0-100 system value. Writes and polls run on a worker thread so a slow
    backend (osascript takes ~150 ms) never stalls the camera loop. Only the
    latest pending write is applied."""

    def __init__(self, name, factory):
        self.name = name
        self.value = None
        self.available = False
        self.reason = ""
        try:
            self._get, self._set, self.backend = factory()
            self.value = self._get()
            self.available = True
        except Exception as e:
            self.reason = str(e) or type(e).__name__
            return
        self._pending = None
        self._refresh = False
        self._busy = False
        self._cv = threading.Condition()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        if sys.platform == "win32":
            import comtypes
            comtypes.CoInitialize()
        while True:
            with self._cv:
                while self._pending is None and not self._refresh:
                    self._cv.wait()
                value, self._pending = self._pending, None
                refresh, self._refresh = self._refresh, False
                self._busy = True
            try:
                if value is not None:
                    self._set(value)
                elif refresh:
                    self.value = self._get()
            except Exception as e:
                print(f"{self.name} backend error: {e}")
            with self._cv:
                self._busy = False
                self._cv.notify_all()

    def set_async(self, v):
        self.value = v
        with self._cv:
            self._pending = v
            self._cv.notify_all()

    def request_refresh(self):
        with self._cv:
            self._refresh = True
            self._cv.notify_all()

    def read_now(self):
        """Wait for queued writes to land, then read the real current value."""
        with self._cv:
            while self._pending is not None or self._busy:
                self._cv.wait(timeout=1)
        try:
            self.value = self._get()
        except Exception as e:
            print(f"{self.name} backend error: {e}")
        return self.value


# ----------------------------------------------------------------------------- modes

class Mode:
    """Interface every mode implements. Only the active mode gets update() calls.

    update(landmarks, frame, now)  act on this frame's hand (landmarks is None if no hand)
    preview(landmarks)             observe while paused: refresh readouts, never act
    reset()                        called on switch-away and pause: release anything held
    on_key(key)                    a key the main loop didn't handle (only sent to the active mode)
    engaged()                      True while the hand is mid-action (pinching); blocks the quit fist
    busy()                         True mid-action (pinch, drag, stroke...); fatigue nudges wait for False
    on_escape()                    return True to use Esc (cancel) instead of quitting
    status()                       (label, BGR color) for the status line
    draw_overlay(frame)            draw mode-specific guides onto the debug frame
    """
    name = "BASE"
    color = GREY

    def update(self, landmarks, frame, now):
        pass

    def preview(self, landmarks):
        pass

    def reset(self):
        pass

    def on_key(self, key):
        pass

    def engaged(self):
        return False

    def busy(self):
        return False

    def on_escape(self):
        """Return True to swallow Esc (e.g. to cancel something) instead of quitting."""
        return False

    def status(self):
        return "IDLE", GREY

    def draw_overlay(self, frame):
        pass


def lock_aspect(region, frame_aspect, screen_aspect):
    """Grow the region around its center until its shape (in pixels) matches the screen.
    It's capped at the frame edges, so a region that would overflow ends up as close as fits."""
    x1, y1, x2, y2 = region
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    if w * frame_aspect / h < screen_aspect:
        w = h * screen_aspect / frame_aspect
    else:
        h = w * frame_aspect / screen_aspect
    w, h = min(w, 1.0), min(h, 1.0)
    cx = min(max(cx, w / 2), 1 - w / 2)
    cy = min(max(cy, h / 2), 1 - h / 2)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def region_from_points(points):
    """Bounding box of all four corner points, shrunk by CAL_PADDING per side.
    Returns (region, None) or (None, reason)."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    px, py = (x2 - x1) * CAL_PADDING, (y2 - y1) * CAL_PADDING
    x1, x2, y1, y2 = x1 + px, x2 - px, y1 + py, y2 - py
    if x2 - x1 < CAL_MIN_SIZE or y2 - y1 < CAL_MIN_SIZE:
        return None, (f"Range too small ({(x2 - x1) * 100:.0f}% x {(y2 - y1) * 100:.0f}% of the frame, "
                      f"need {CAL_MIN_SIZE * 100:.0f}%). Press k and reach further.")
    return (x1, y1, x2, y2), None


def load_calibration(path=None):
    """Return the saved region, or None (defaults) if the file is missing or invalid."""
    path = path or CALIBRATION_FILE
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            region = tuple(float(v) for v in json.load(f)["region"])
        x1, y1, x2, y2 = region
        if not (len(region) == 4 and 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"region out of range: {region}")
        print(f"Loaded calibration from {os.path.basename(path)}: "
              f"x {x1:.2f}-{x2:.2f}, y {y1:.2f}-{y2:.2f}")
        return region
    except Exception as e:
        print(f"Ignoring {os.path.basename(path)} ({e}); using the default region.")
        return None


def save_calibration(region, points, path=None):
    path = path or CALIBRATION_FILE
    data = {"region": [round(v, 4) for v in region],
            "corner_points": [[round(x, 4), round(y, 4)] for x, y in points],
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)  # atomic: a crash mid-write never leaves a half-written file


class Calibrator:
    """Guided four-corner calibration. Feed it the index fingertip each frame."""
    CORNERS = (("TOP-LEFT", (0, 0)), ("TOP-RIGHT", (1, 0)),
               ("BOTTOM-RIGHT", (1, 1)), ("BOTTOM-LEFT", (0, 1)))

    def __init__(self):
        self.index = 0
        self.points = []           # median fingertip per finished corner, normalized
        self.samples = []
        self.started = None        # start of the current corner window
        self.progress = 0.0
        self.tip = None
        self.note = ""             # retry reason shown on screen

    @property
    def corner(self):
        return self.CORNERS[self.index]

    def update(self, lm, now, aspect):
        """Return None while running, ("done", region, points) or ("failed", reason)."""
        if self.started is None:
            self.started = now
        elapsed = now - self.started
        self.tip = (lm[INDEX_TIP].x, lm[INDEX_TIP].y) if lm is not None else None
        if self.tip is not None and elapsed >= CAL_IGNORE_S:
            self.samples.append(self.tip)
        self.progress = min(elapsed / CAL_CORNER_S, 1.0)
        if elapsed < CAL_CORNER_S:
            return None
        return self.finish_corner(now, aspect)

    def finish_corner(self, now, aspect):
        name = self.corner[0]
        point, problem = self.measure(aspect)
        self.samples, self.started, self.progress = [], now, 0.0
        if problem:
            self.note = f"{name}: {problem} - try that corner again"
            return None
        self.note = ""
        self.points.append(point)
        self.index += 1
        if self.index < len(self.CORNERS):
            return None
        region, reason = region_from_points(self.points)
        return ("done", region, self.points) if region else ("failed", reason)

    def measure(self, aspect):
        if len(self.samples) < CAL_MIN_SAMPLES:
            return None, "no hand seen"
        xs = sorted(p[0] for p in self.samples)
        ys = sorted(p[1] for p in self.samples)
        med = (xs[len(xs) // 2], ys[len(ys) // 2]) if len(xs) % 2 else \
              ((xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2, (ys[len(ys) // 2 - 1] + ys[len(ys) // 2]) / 2)
        dists = sorted(math.hypot((x - med[0]) * aspect, y - med[1]) for x, y in self.samples)
        if dists[int(len(dists) * 0.8) - 1] > CAL_MAX_SCATTER:
            return None, "hand moved too much, hold still"
        return med, None

    def draw_overlay(self, frame):
        h, w = frame.shape[:2]
        name, (cxn, cyn) = self.corner
        # Big marker in the targeted corner of the window (inset below the mode banner),
        # with an arrow pointing into the corner.
        corner = (int(cxn * (w - 1)), 37 if cyn == 0 else h - 1)
        marker = (60 if cxn == 0 else w - 61, 37 + 60 if cyn == 0 else h - 61)
        cv2.circle(frame, marker, 34, (0, 230, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, marker, 42, WHITE, 3, cv2.LINE_AA)
        cv2.arrowedLine(frame, marker, corner, (0, 230, 255), 5, cv2.LINE_AA, tipLength=0.35)
        for i, (px, py) in enumerate(self.points):
            cv2.circle(frame, (int(px * w), int(py * h)), 8, (0, 200, 0), -1)
            cv2.putText(frame, str(i + 1), (int(px * w) + 10, int(py * h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        if self.tip is not None:
            tip = (int(self.tip[0] * w), int(self.tip[1] * h))
            cv2.circle(frame, tip, 10, (0, 230, 255), 2, cv2.LINE_AA)
            # Progress ring around the fingertip: grey while you get there, yellow while sampling.
            sampling = self.progress * CAL_CORNER_S >= CAL_IGNORE_S
            cv2.ellipse(frame, tip, (26, 26), -90, 0, 360 * self.progress,
                        (0, 230, 255) if sampling else GREY, 5, cv2.LINE_AA)
        # Prompt box in the middle of the frame.
        box_y = h // 2 - 50
        cv2.rectangle(frame, (20, box_y), (w - 20, box_y + 100), (30, 30, 30), -1)
        cv2.putText(frame, f"Point to {name}", (36, box_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 2)
        cv2.putText(frame, "of your comfortable range and hold still", (36, box_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        bar_w = w - 72
        cv2.rectangle(frame, (36, box_y + 74), (36 + bar_w, box_y + 86), (80, 80, 80), -1)
        cv2.rectangle(frame, (36, box_y + 74), (36 + int(bar_w * self.progress), box_y + 86), (0, 230, 255), -1)
        cv2.putText(frame, f"corner {self.index + 1}/4   Esc = cancel", (w - 230, box_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
        if self.note:
            cv2.putText(frame, self.note, (24, box_y + 124), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


@dataclass
class MouseState:
    pinched: bool = False          # hysteresis output
    dragging: bool = False
    pinch_started_at: float = 0.0
    frozen_until: float = 0.0      # cursor ignores the hand until this time
    smooth_pos: tuple | None = None
    last_time: float | None = None
    pinch_dist: float | None = None


class MouseMode(Mode):
    """Index fingertip drives the cursor; pinch to click, hold the pinch to drag."""
    name = "MOUSE"
    color = (0, 200, 0)

    def __init__(self, screen):
        self.screen = screen
        self.state = MouseState()
        self.calibrated = load_calibration()   # normalized region, or None = default margins
        self.calibrator = None                 # set while a calibration is running
        self.frame_aspect = CAMERA_WIDTH / CAMERA_HEIGHT
        self.message, self.message_until = "", 0.0
        # Clutch state
        self.clutch_enabled = CLUTCH_ENABLED
        self.clutch = ClutchDetector()
        self.raw_fist = False          # this frame looks like a fist (before debounce)
        self.frozen_cursor = None      # cursor position held while clutched
        self.offset = (0.0, 0.0)       # added to the mapped fingertip position
        self.ghost = None              # mapped position with no offset (debug marker)
        self._prev_raw = None
        self.pinch_armed = True        # after unclutching, wait for the fingers to stay apart before clicking
        self._arm_frames = 0
        self.lost_since = None
        self._reanchor = False         # clutch switched off with x while engaged: re-anchor next frame
        # Scroll state
        self.scroll_pose = PoseTracker()
        self.scrolling = False
        self.scroll_y = None           # smoothed palm height while scrolling (normalized)
        self.scroll_accum = 0.0        # fractional scroll lines not sent yet
        self.scroll_dir = 0            # last scroll direction, for the overlay (-1 down, +1 up)
        self.scroll_palm = None        # palm position in frame px, for the overlay

    @property
    def region(self):
        """Region used for mapping: calibrated or default, optionally aspect-locked.
        None means the default margins (kept as the exact original code path)."""
        region = self.calibrated
        if LOCK_ASPECT_RATIO:
            region = lock_aspect(region or default_region(), self.frame_aspect, self.screen[2] / self.screen[3])
        return region

    def update(self, landmarks, frame, now):
        s = self.state
        dt = now - s.last_time if s.last_time is not None else 1 / 30
        s.last_time = now
        if frame is not None:
            self.frame_aspect = frame.shape[1] / frame.shape[0]
        if self.calibrator is not None:
            self.update_calibration(landmarks, now)   # the mouse never moves or clicks here
            return
        if landmarks is None:
            self.release()
            s.pinch_dist = None
            self.raw_fist = False
            if self.scrolling:                 # hand lost mid-scroll: end it, don't jump-scroll on return
                self.scrolling = False
                self.scroll_pose.reset()
                self.frozen_cursor = None
            self.lost_since = self.lost_since if self.lost_since is not None else now
            if now - self.lost_since > CLUTCH_LOST_RESET_S and (self.clutch.engaged or self.offset != (0.0, 0.0)):
                self.clutch.reset()
                self.frozen_cursor = None
                self.reset_offset("hand lost")
            return
        self.lost_since = None
        s.pinch_dist = pinch_distance(landmarks)
        if self._reanchor:
            self._reanchor = False
            self.release_clutch(landmarks)
        # Clutch runs before pinch detection and overrides it: a fist can look like a pinch.
        if self.clutch_enabled and self.handle_clutch(landmarks):
            return
        if SCROLL_ENABLED and self.handle_scroll(landmarks):
            return
        if not self.pinch_armed:
            # Just unclutched: opening a fist can pass through a pinch-like pose, so a pinch
            # only counts after thumb and index have been apart for CLUTCH_RELEASE_FRAMES frames.
            apart = s.pinch_dist is None or s.pinch_dist > PINCH_RELEASE
            self._arm_frames = self._arm_frames + 1 if apart else 0
            self.pinch_armed = self._arm_frames >= CLUTCH_RELEASE_FRAMES
            pinched_now = False
        else:
            pinched_now = detect_pinch(s.pinch_dist, s.pinched)
        self.handle_click_logic(pinched_now, now)
        self.move_cursor(landmarks, now, dt)

    # --- clutch -----------------------------------------------------------------
    def handle_clutch(self, lm):
        """Update the clutch. Return True when this frame must not move or click."""
        self.raw_fist = is_fist(lm, self.frame_aspect)
        change = self.clutch.update(lm, self.frame_aspect)
        if change == "engage":
            self.engage_clutch()
        elif change == "release":
            self.release_clutch(lm)
        if self.clutch.engaged:
            self.ghost = map_to_screen(lm[INDEX_TIP].x, lm[INDEX_TIP].y, self.screen, self.region)
            return True
        return self.raw_fist   # fist forming but not confirmed yet: hold still, don't pinch

    def hold_cursor(self):
        """Freeze the cursor where it is for a gesture (clutch, scroll); never click."""
        s = self.state
        if s.dragging:
            pyautogui.mouseUp()        # release the button before lifting the "mouse"
        s.dragging = False
        s.pinched = False              # a pinch that started as the gesture formed is not a click
        s.frozen_until = 0.0
        self.frozen_cursor = s.smooth_pos

    def reanchor(self, lm):
        """Continue from the held cursor: offset = held position - mapped fingertip."""
        s = self.state
        mapped = map_to_screen(lm[INDEX_TIP].x, lm[INDEX_TIP].y, self.screen, self.region)
        if self.frozen_cursor is not None:
            self.offset = (self.frozen_cursor[0] - mapped[0], self.frozen_cursor[1] - mapped[1])
        self.frozen_cursor = None
        s.smooth_pos = None            # first frame after release starts exactly at the frozen point
        s.frozen_until = 0.0
        self._prev_raw = mapped
        self.pinch_armed, self._arm_frames = False, 0

    def engage_clutch(self):
        self.hold_cursor()
        print("Clutch engaged")

    def release_clutch(self, lm):
        self.reanchor(lm)
        print(f"Clutch released, offset ({self.offset[0]:+.0f}, {self.offset[1]:+.0f}) px")

    def handle_scroll(self, lm):
        """Peace sign = scroll by moving the hand vertically. Return True when this
        frame must not move the cursor or click."""
        raw = classify_pose(lm, self.frame_aspect)
        peace = self.scroll_pose.update(raw) == "PEACE"
        palm_y = lm[MIDDLE_MCP].y
        if peace and not self.scrolling:
            self.hold_cursor()
            self.scrolling, self.scroll_y, self.scroll_accum, self.scroll_dir = True, palm_y, 0.0, 0
        elif not peace and self.scrolling:
            self.scrolling = False
            self.reanchor(lm)
        if not self.scrolling:
            return raw == "PEACE"          # peace sign forming: hold still, don't pinch
        prev = self.scroll_y
        self.scroll_y = prev + SCROLL_SMOOTH_ALPHA * (palm_y - prev)
        dy = self.scroll_y - prev          # < 0 = hand moved up
        if abs(dy) >= SCROLL_DEADZONE:
            # pyautogui: positive = scroll up. Hand up (dy < 0) -> negative -> page scrolls down.
            self.scroll_accum += dy * SCROLL_SPEED * SCROLL_DIRECTION
        lines = int(self.scroll_accum)
        if lines:
            pyautogui.scroll(lines)
            self.scroll_accum -= lines
            self.scroll_dir = 1 if lines > 0 else -1
        self.scroll_palm = (lm[MIDDLE_MCP].x, palm_y)
        self.ghost = map_to_screen(lm[INDEX_TIP].x, lm[INDEX_TIP].y, self.screen, self.region)
        return True

    def reset_offset(self, why=""):
        if self.offset != (0.0, 0.0):
            print(f"Offset reset{' (' + why + ')' if why else ''}")
        self.offset = (0.0, 0.0)
        self._prev_raw = None
        self.ghost = None

    def clamp_to_screen(self, x, y):
        left, top, sw, sh = self.screen
        return (min(max(x, left), left + sw - 1), min(max(y, top), top + sh - 1))

    def toggle_clutch(self):
        if self.clutch.engaged:
            self.clutch.reset()
            self._reanchor = True      # continue from the frozen cursor, don't jump
        self.clutch_enabled = not self.clutch_enabled
        self.flash(f"Clutch {'ON' if self.clutch_enabled else 'OFF (fist quits again)'}")
        print(f"Clutch {'enabled' if self.clutch_enabled else 'disabled'}")

    def preview(self, landmarks):
        self.state.pinch_dist = pinch_distance(landmarks) if landmarks is not None else None

    def reset(self):
        """Mode switch / pause: cancel calibration, drop the clutch and the offset."""
        if self.calibrator is not None:
            self.cancel_calibration()
        self.release()
        self.clutch.reset()
        self.frozen_cursor = None
        self.raw_fist = False
        self.pinch_armed = True
        self.lost_since = None
        self._reanchor = False
        self.scrolling = False
        self.scroll_pose.reset()
        self.reset_offset()

    def release(self):
        """Never leave the mouse button stuck down, and don't jump on the next frame."""
        s = self.state
        if s.dragging:
            pyautogui.mouseUp()
        s.dragging = False
        s.pinched = False
        s.frozen_until = 0.0
        s.smooth_pos = None

    def handle_click_logic(self, pinched_now, now):
        """Turn pinch transitions into click / drag mouse events."""
        s = self.state
        if pinched_now and not s.pinched:
            # Pinch began: freeze the cursor where it is so the pinch motion can't misaim.
            s.pinch_started_at = now
            s.frozen_until = float("inf")
        elif pinched_now and s.pinched and not s.dragging:
            if (now - s.pinch_started_at) * 1000 >= DRAG_DELAY_MS:
                pyautogui.mouseDown()
                s.dragging = True
                s.frozen_until = 0.0      # let the cursor move (damped) while dragging
        elif not pinched_now and s.pinched:
            if s.dragging:
                pyautogui.mouseUp()
                s.dragging = False
            else:
                pyautogui.click()
            s.frozen_until = now + RELEASE_FREEZE_MS / 1000
        s.pinched = pinched_now

    def move_cursor(self, lm, now, dt):
        s = self.state
        target = map_to_screen(lm[INDEX_TIP].x, lm[INDEX_TIP].y, self.screen, self.region)
        if self.offset != (0.0, 0.0):
            target = self.apply_offset(target, frozen=now < s.frozen_until)
        if now < s.frozen_until:
            # Keep the filter anchored on the frozen point so there's no jump afterwards.
            return
        scale = DRAG_ALPHA_SCALE if s.dragging else 1.0
        new = smooth(s.smooth_pos, target, dt, scale)
        if s.smooth_pos is not None and math.hypot(new[0] - s.smooth_pos[0],
                                                   new[1] - s.smooth_pos[1]) < DEADZONE_PX:
            return
        s.smooth_pos = new
        pyautogui.moveTo(int(new[0]), int(new[1]))

    def apply_offset(self, raw, frozen=False):
        """raw mapped position -> clamped position with the clutch offset.

        An offset of +dx hides the screen strip [0, dx) (the hand can't map below 0).
        So per axis, while the hand moves toward the hidden side, bleed that axis of
        the offset toward zero in proportion to the movement. The hidden strip comes
        back into reach; movement away from it stays exactly 1:1."""
        self.ghost = raw
        if not frozen and self._prev_raw is not None and CLUTCH_BLEED_RATE > 0:
            off = list(self.offset)
            for i in (0, 1):
                d = raw[i] - self._prev_raw[i]
                if d * off[i] < 0:                      # moving toward the hidden side
                    off[i] *= math.exp(-CLUTCH_BLEED_RATE * abs(d))
                    if abs(off[i]) < 10:            # snap the last few px so nothing stays hidden
                        off[i] = 0.0
            self.offset = (off[0], off[1])
        self._prev_raw = raw
        return self.clamp_to_screen(raw[0] + self.offset[0], raw[1] + self.offset[1])

    def busy(self):
        return self.state.pinched or self.state.dragging or self.calibrator is not None or self.scrolling

    def engaged(self):
        """Blocks the quit fist: pinching, calibrating, or the fist is the clutch."""
        return (self.state.pinched or self.calibrator is not None
                or (self.clutch_enabled and (self.clutch.engaged or self.raw_fist)))

    # --- calibration ----------------------------------------------------------
    def start_calibration(self):
        self.release()
        self.clutch.reset()
        self.frozen_cursor = None
        self.reset_offset("calibration")
        self.calibrator = Calibrator()
        print("Calibration started: point to each corner of your comfortable range.")

    def update_calibration(self, landmarks, now):
        result = self.calibrator.update(landmarks, now, self.frame_aspect)
        if result is None:
            return
        self.calibrator = None
        if result[0] == "failed":
            self.flash(f"Calibration rejected: {result[1]}", 6.0)
            print(f"Calibration rejected: {result[1]}")
            return
        _, region, points = result
        self.calibrated = region
        self.reset_offset("calibration")
        try:
            save_calibration(region, points)
            where = f"saved to {os.path.basename(CALIBRATION_FILE)}"
        except OSError as e:
            where = f"NOT saved ({e})"
        x1, y1, x2, y2 = region
        self.flash(f"Calibrated: x {x1:.2f}-{x2:.2f}, y {y1:.2f}-{y2:.2f} ({where})", 5.0)
        print(f"Calibration done: x {x1:.3f}-{x2:.3f}, y {y1:.3f}-{y2:.3f}, {where}")

    def cancel_calibration(self):
        self.calibrator = None
        self.flash("Calibration cancelled - region unchanged")
        print("Calibration cancelled.")

    def reset_calibration(self):
        self.calibrated = None
        self.reset_offset("r")
        try:
            if os.path.exists(CALIBRATION_FILE):
                os.remove(CALIBRATION_FILE)
        except OSError as e:
            print(f"Couldn't delete {CALIBRATION_FILE}: {e}")
        self.flash("Region reset to defaults")
        print("Region reset to defaults.")

    def on_key(self, key):
        if key == ord("k") and self.calibrator is None:
            self.start_calibration()
        elif key == ord("r") and self.calibrator is None:
            self.reset_calibration()
        elif key == ord("x") and self.calibrator is None:
            self.toggle_clutch()

    def on_escape(self):
        if self.calibrator is None:
            return False
        self.cancel_calibration()
        return True

    def flash(self, text, seconds=3.0):
        self.message, self.message_until = text, time.monotonic() + seconds

    CLUTCH_COLOR = (255, 0, 255)

    def status(self):
        if self.calibrator is not None:
            return "CALIBRATING", (0, 230, 255)
        if self.clutch_enabled and self.clutch.engaged:
            return "CLUTCHED", self.CLUTCH_COLOR
        if self.scrolling:
            way = {1: "UP", -1: "DOWN"}.get(self.scroll_dir, "")
            return f"SCROLLING {way}".strip(), (255, 160, 0)
        if self.state.dragging:
            return "DRAGGING", (0, 0, 255)
        if self.state.pinched:
            return "PINCHED", (0, 255, 255)
        return "TRACKING", (0, 200, 0)

    def draw_overlay(self, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = active_region_px(w, h, self.region)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        label = ("calibrated" if self.calibrated else "default") + (" + aspect lock" if LOCK_ASPECT_RATIO else "")
        cv2.putText(frame, label, (x1 + 4, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        if self.calibrator is not None:
            self.calibrator.draw_overlay(frame)
            return
        d = "--" if self.state.pinch_dist is None else f"{self.state.pinch_dist:.2f}"
        cv2.putText(frame, f"pinch {d} (<{PINCH_START} / >{PINCH_RELEASE})  k=calibrate r=reset "
                    f"x=clutch {'on' if self.clutch_enabled else 'OFF'}",
                    (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
        self.draw_clutch_markers(frame, (x1, y1, x2, y2))
        if self.message and time.monotonic() < self.message_until:
            cv2.putText(frame, self.message, (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1)

    def draw_scroll_indicator(self, frame):
        if not self.scrolling or self.scroll_palm is None:
            return
        h, w = frame.shape[:2]
        x, y = int(self.scroll_palm[0] * w) + 60, int(self.scroll_palm[1] * h)
        color = (255, 160, 0)
        cv2.line(frame, (x, y - 45), (x, y + 45), (90, 90, 90), 6, cv2.LINE_AA)
        if self.scroll_dir:
            # Arrow shows which way the PAGE moves.
            tip = (x, y + 45) if self.scroll_dir < 0 else (x, y - 45)
            cv2.arrowedLine(frame, (x, y), tip, color, 6, cv2.LINE_AA, tipLength=0.4)
        cv2.putText(frame, "scroll", (x - 22, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def screen_to_frame(self, pt, region_px):
        """Inverse of map_to_screen, for drawing a screen position inside the region box."""
        left, top, sw, sh = self.screen
        x1, y1, x2, y2 = region_px
        rx = (pt[0] - left - EDGE_INSET_PX) / max(sw - 1 - 2 * EDGE_INSET_PX, 1)
        ry = (pt[1] - top - EDGE_INSET_PX) / max(sh - 1 - 2 * EDGE_INSET_PX, 1)
        return int(x1 + rx * (x2 - x1)), int(y1 + ry * (y2 - y1))

    def draw_clutch_markers(self, frame, region_px):
        """Cursor (filled) vs ghost = where it would be with no offset (hollow)."""
        self.draw_scroll_indicator(frame)
        clutched = (self.clutch_enabled and self.clutch.engaged) or self.scrolling
        cursor = self.frozen_cursor if clutched else self.state.smooth_pos
        if self.ghost is None or cursor is None or not (clutched or self.offset != (0.0, 0.0)):
            return
        c = self.screen_to_frame(cursor, region_px)
        g = self.screen_to_frame(self.ghost, region_px)
        cv2.line(frame, g, c, GREY, 1, cv2.LINE_AA)
        cv2.circle(frame, g, 9, GREY, 2, cv2.LINE_AA)
        cv2.putText(frame, "ghost", (g[0] + 11, g[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)
        cv2.circle(frame, c, 7, self.CLUTCH_COLOR, -1, cv2.LINE_AA)
        cv2.putText(frame, "cursor", (c[0] + 10, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLUTCH_COLOR, 1)
        dx, dy = (self.offset if not clutched else (cursor[0] - self.ghost[0], cursor[1] - self.ghost[1]))
        cv2.putText(frame, f"offset {dx:+.0f}, {dy:+.0f} px", (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    self.CLUTCH_COLOR, 1)


class CircleTracker:
    """Measures how far the fingertip has turned around the center of its recent path.

    The center is the mean fingertip position over the last CIRCLE_WINDOW_S. While
    the finger circles, the center -> tip vector turns with it; a straight swipe or
    a still finger keeps it pointing one way (or too short to count).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.trail = deque()       # (time, (x, y)) in aspect-corrected units (frame height = 1)
        self.tip = None
        self.center = None
        self.radius = 0.0
        self.prev_angle = None
        self.active = False

    def update(self, tip, now):
        """Feed a fingertip position; return the rotation this frame in degrees
        (positive = clockwise as seen in the window), 0 when not circling."""
        if self.tip is None:
            self.tip = tip
        else:
            self.tip = (self.tip[0] + CIRCLE_TIP_ALPHA * (tip[0] - self.tip[0]),
                        self.tip[1] + CIRCLE_TIP_ALPHA * (tip[1] - self.tip[1]))
        self.trail.append((now, self.tip))
        while self.trail and now - self.trail[0][0] > CIRCLE_WINDOW_S:
            self.trail.popleft()
        if len(self.trail) < 8:
            return 0.0
        xs, ys = zip(*(p for _, p in self.trail))
        self.center = (sum(xs) / len(xs), sum(ys) / len(ys))
        vx, vy = self.tip[0] - self.center[0], self.tip[1] - self.center[1]
        self.radius = math.hypot(vx, vy)
        if self.radius < CIRCLE_MIN_RADIUS:
            self.prev_angle, self.active = None, False
            return 0.0
        angle = math.degrees(math.atan2(vy, vx))
        d = 0.0 if self.prev_angle is None else wrap_degrees(angle - self.prev_angle)
        self.prev_angle = angle
        if abs(d) > CIRCLE_MAX_STEP_DEG:
            return 0.0
        self.active = True
        return d


class VolumeMode(Mode):
    """Virtual knob: pinch to grab, rotate the hand to turn, release to let go.
    Or, without pinching, circle the index finger: clockwise = up, counter-clockwise = down.

    Relative control: both start from the current system value, so it never jumps.
    """
    name = "VOLUME"
    color = (255, 140, 0)
    GRAB_COLOR = (0, 215, 255)

    def __init__(self, targets):
        self.targets = targets
        self.target_name = KNOB_TARGET
        self.message = ""
        if not self.target.available and self.target_name != "volume":
            self.message = f"{self.target_name} unavailable ({self.target.reason}); using volume"
            print(self.message)
            self.target_name = "volume"
        self.pinched = False
        self.grabbed = False
        self.pinch_dist = None
        self.angle = None        # smoothed, unwrapped hand angle (degrees)
        self._unwrapped = None   # raw angle with +/-180 wraparound removed
        self._raw_prev = None
        self.grab_angle = 0.0
        self.grab_value = 0.0
        self.knob_value = None
        self.last_sent = None
        self.last_poll = 0.0
        self.pose = PoseTracker()
        self.circle = CircleTracker()
        self.circling = False      # past the deadband and changing the value
        self.circle_accum = 0.0    # rotation since the circle tracker (re)started
        self.circle_value = None
        self.circle_dir = 0

    @property
    def target(self):
        return self.targets[self.target_name]

    def update(self, landmarks, frame, now):
        if landmarks is None:
            self.reset()
            self.pinch_dist = None
            return
        self.pinch_dist = pinch_distance(landmarks)
        self.track_angle(hand_angle(landmarks))
        pinched_now = detect_pinch(self.pinch_dist, self.pinched)
        if pinched_now and not self.pinched:
            self.grab()
        elif not pinched_now and self.pinched:
            self.grabbed = False           # let go: the value stays where it is
        self.pinched = pinched_now
        h, w = frame.shape[:2] if frame is not None else (3, 4)
        pose = self.pose.update(classify_pose(landmarks, w / h))
        if self.grabbed:
            self.turn()
            self.stop_circle()
        elif pose == "POINT" and not self.pinched and self.target.available:
            tip = (landmarks[INDEX_TIP].x * w / h, landmarks[INDEX_TIP].y)
            self.apply_circle(self.circle.update(tip, now))
        else:
            self.stop_circle()
        if not self.grabbed and not self.circling and self.target.available \
                and now - self.last_poll >= KNOB_POLL_S:
            self.last_poll = now
            self.target.request_refresh()

    def preview(self, landmarks):
        self.pinch_dist = pinch_distance(landmarks) if landmarks is not None else None

    def reset(self):
        self.grabbed = False
        self.pinched = False
        self.angle = self._unwrapped = self._raw_prev = None
        self.pose.reset()
        self.stop_circle()

    def engaged(self):
        return self.pinched

    def busy(self):
        return self.grabbed or self.circling

    def stop_circle(self):
        self.circle.reset()
        self.circling = False
        self.circle_accum = 0.0
        self.circle_dir = 0

    def apply_circle(self, d):
        if not self.circle.active:
            self.circling, self.circle_accum = False, 0.0
            return
        self.circle_accum += d
        if not self.circling:
            if abs(self.circle_accum) < CIRCLE_DEADBAND_DEG:
                return
            self.circling = True
            self.circle_value = self.target.value if self.target.value is not None else 0.0
            self.last_sent = round(self.circle_value)
            d = self.circle_accum - math.copysign(CIRCLE_DEADBAND_DEG, self.circle_accum)
        if d:
            self.circle_dir = 1 if d > 0 else -1
        change = d * CIRCLE_DIRECTION * CIRCLE_PERCENT_PER_TURN / 360.0
        self.circle_value = min(max(self.circle_value + change, 0.0), 100.0)
        r = round(self.circle_value)
        if abs(r - self.last_sent) >= KNOB_MIN_STEP:
            self.target.set_async(r)
            self.last_sent = r

    def track_angle(self, raw):
        """Unwrap across +/-180, then EMA-smooth."""
        if self._raw_prev is None:
            self._unwrapped = self.angle = raw
        else:
            self._unwrapped += wrap_degrees(raw - self._raw_prev)
            self.angle += KNOB_ANGLE_ALPHA * (self._unwrapped - self.angle)
        self._raw_prev = raw

    def grab(self):
        if not self.target.available:
            return
        start = self.target.read_now()
        if start is None:
            return
        self.grabbed = True
        self.grab_angle = self.angle
        self.grab_value = self.knob_value = start
        self.last_sent = round(start)

    def turn(self):
        delta = (self.angle - self.grab_angle) * KNOB_DIRECTION * 100.0 / KNOB_DEGREES_FOR_FULL
        v = self.grab_value + delta
        # At a stop, move the anchor so turning back responds immediately, like a real knob.
        if v > 100:
            self.grab_value -= v - 100
            v = 100.0
        elif v < 0:
            self.grab_value -= v
            v = 0.0
        self.knob_value = v
        r = round(v)
        if abs(r - self.last_sent) >= KNOB_MIN_STEP:
            self.target.set_async(r)
            self.last_sent = r

    def on_key(self, key):
        if key != ord("b"):
            return
        other = "brightness" if self.target_name == "volume" else "volume"
        if not self.targets[other].available:
            self.message = f"{other} unavailable: {self.targets[other].reason}"
            print(self.message)
            return
        self.reset()
        self.target_name = other
        self.message = ""
        print(f"Knob target: {other}")

    def status(self):
        if not self.target.available:
            return "UNAVAILABLE", GREY
        if self.grabbed:
            return "GRABBED", self.GRAB_COLOR
        if self.circling:
            return ("CIRCLING CW (up)" if self.circle_dir * CIRCLE_DIRECTION >= 0
                    else "CIRCLING CCW (down)"), self.GRAB_COLOR
        return "RELEASED", WHITE

    def draw_overlay(self, frame):
        w = frame.shape[1]
        cx, cy, r = w - 95, 150, 58
        if self.grabbed:
            value = self.knob_value
        elif self.circling:
            value = self.circle_value
        else:
            value = self.target.value
        ring = self.GRAB_COLOR if self.grabbed or self.circling else GREY
        self.draw_circle_trail(frame)
        if self.grabbed:
            cv2.circle(frame, (cx, cy), r, (40, 40, 40), -1)
        cv2.circle(frame, (cx, cy), r, ring, 5 if self.grabbed else 2)
        # Level arc: a 270-degree gauge starting bottom-left.
        cv2.ellipse(frame, (cx, cy), (r + 10, r + 10), 0, 135, 405, (70, 70, 70), 4)
        if value is not None:
            cv2.ellipse(frame, (cx, cy), (r + 10, r + 10), 0, 135, 135 + 270 * value / 100, self.color, 4)
        if self.angle is not None:
            a = math.radians(self.angle)
            tip = (int(cx + (r - 8) * math.cos(a)), int(cy + (r - 8) * math.sin(a)))
            cv2.line(frame, (cx, cy), tip, ring, 4 if self.grabbed else 2)
            cv2.circle(frame, tip, 6 if self.grabbed else 4, ring, -1)
        text = "N/A" if value is None else f"{value:.0f}%"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(frame, text, (cx - tw // 2, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        label = self.target_name.upper() + (" - GRABBED" if self.grabbed else
                                            " - CIRCLING" if self.circling else "")
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, label, (cx - tw // 2, cy + r + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ring, 1)
        hint = "pinch + rotate hand = knob | circle index finger: CW up, CCW down | b"
        cv2.putText(frame, hint, (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        if self.message:
            cv2.putText(frame, self.message, (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    def draw_circle_trail(self, frame):
        """Fingertip path and its center while circling, so you can see what's measured."""
        if not self.circle.trail or self.circle.center is None:
            return
        h = frame.shape[0]
        pts = np.array([(int(x * h), int(y * h)) for _, (x, y) in self.circle.trail], np.int32)
        color = self.GRAB_COLOR if self.circling else GREY
        cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)
        cx, cy = int(self.circle.center[0] * h), int(self.circle.center[1] * h)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        if self.circling:
            sign = "+" if self.circle_dir * CIRCLE_DIRECTION >= 0 else "-"
            cv2.putText(frame, sign, (cx - 8, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


class DrawMode(Mode):
    """Draw in the air with the index fingertip onto a canvas blended over the feed.

    Pen down = POINT pose, pen up = PEACE (or anything else), PALM held = clear.
    Strokes are kept as point lists so undo can re-render the canvas.
    """
    name = "DRAW"
    color = (200, 0, 200)

    def __init__(self):
        self.strokes = []          # [{"color": bgr, "size": px, "points": [(x, y), ...]}]
        self.canvas = None         # BGRA, same size as the frame; alpha 0 = transparent
        self.color_key = DRAW_START_COLOR
        self.size = DRAW_BRUSH_SIZE
        self.pose = PoseTracker()
        self.pen_down = False
        self.cursor = None         # smoothed fingertip, frame px
        self.last_time = None
        self._recent_lengths = []  # stroke length at each recent frame, to trim the pose-lag tail
        self.clear_started = None
        self.clear_anchor = None
        self.clear_progress = 0.0
        self.clear_latched = False
        self.palm_center = None
        self.message, self.message_until = "", 0.0

    # --- canvas ---------------------------------------------------------------
    def ensure_canvas(self, frame):
        h, w = frame.shape[:2]
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 4), np.uint8)
            self.render()

    def render(self):
        if self.canvas is None:
            return
        self.canvas[:] = 0
        for s in self.strokes:
            pts = s["points"]
            for p, q in zip(pts, pts[1:]):
                self.draw_segment(s, p, q)
            if len(pts) == 1:
                self.draw_segment(s, pts[0], pts[0])

    def draw_segment(self, stroke, p, q):
        color = (*stroke["color"], 255)
        p, q = (int(p[0]), int(p[1])), (int(q[0]), int(q[1]))
        if p == q:
            cv2.circle(self.canvas, p, max(stroke["size"] // 2, 1), color, -1, cv2.LINE_AA)
        else:
            cv2.line(self.canvas, p, q, color, stroke["size"], cv2.LINE_AA)

    # --- strokes --------------------------------------------------------------
    def begin_stroke(self):
        self.strokes.append({"color": DRAW_COLORS[self.color_key][1], "size": self.size,
                             "points": [self.cursor]})
        self.draw_segment(self.strokes[-1], self.cursor, self.cursor)
        self.pen_down = True
        self._recent_lengths = []

    def extend_stroke(self):
        s = self.strokes[-1]
        last = s["points"][-1]
        if math.hypot(self.cursor[0] - last[0], self.cursor[1] - last[1]) >= DRAW_MIN_SEGMENT_PX:
            s["points"].append(self.cursor)
            self.draw_segment(s, last, self.cursor)
        self._recent_lengths.append(len(s["points"]))
        del self._recent_lengths[:-POSE_STABLE_FRAMES]

    def end_stroke(self, trim=True):
        """Pen up. The pose change was confirmed POSE_STABLE_FRAMES late, so drop
        the points drawn during that lag (that's where the peace sign wobbles)."""
        if self.pen_down and trim and len(self._recent_lengths) >= POSE_STABLE_FRAMES:
            keep = self._recent_lengths[-POSE_STABLE_FRAMES]   # length just before the lag frames
            if keep < len(self.strokes[-1]["points"]):
                del self.strokes[-1]["points"][keep:]
                self.render()
        self.pen_down = False
        self._recent_lengths = []

    def clear(self):
        self.end_stroke(trim=False)
        self.strokes = []
        self.render()
        self.flash("Canvas cleared")

    def undo(self):
        self.end_stroke(trim=False)
        if self.strokes:
            self.strokes.pop()
            self.render()
            self.flash(f"Undo ({len(self.strokes)} strokes left)")

    def save(self):
        if not self.strokes or self.canvas is None:
            self.flash("Nothing to save")
            return
        os.makedirs(DRAW_SAVE_DIR, exist_ok=True)
        path = os.path.join(DRAW_SAVE_DIR, time.strftime("drawing_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(path, self.canvas)  # BGRA -> PNG with a transparent background
        print(f"Saved {path}")
        self.flash(f"Saved {os.path.basename(path)}")

    def flash(self, text, seconds=2.0):
        self.message, self.message_until = text, time.monotonic() + seconds

    # --- Mode interface -------------------------------------------------------
    def update(self, landmarks, frame, now):
        self.ensure_canvas(frame)
        dt = now - self.last_time if self.last_time is not None else 1 / 30
        self.last_time = now
        if landmarks is None:
            self.reset()
            return
        h, w = frame.shape[:2]
        pose = self.pose.update(classify_pose(landmarks, w / h))
        target = (landmarks[INDEX_TIP].x * w, landmarks[INDEX_TIP].y * h)
        self.cursor = smooth(self.cursor, target, dt, speed_scale=DRAW_SPEED_SCALE)
        self.palm_center = (int(landmarks[MIDDLE_MCP].x * w), int(landmarks[MIDDLE_MCP].y * h))
        self.handle_clear_gesture(pose, landmarks, now)
        if pose == "POINT" and not self.pen_down:
            self.begin_stroke()           # new stroke each time, so gaps never connect
        elif pose != "POINT" and self.pen_down:
            self.end_stroke()
        if self.pen_down:
            self.extend_stroke()

    def handle_clear_gesture(self, pose, lm, now):
        if pose != "PALM":
            self.clear_started, self.clear_progress, self.clear_latched = None, 0.0, False
            return
        if self.clear_latched:            # already cleared; lower the palm to arm it again
            return
        wrist = (lm[WRIST].x, lm[WRIST].y)
        if self.clear_started is None or math.hypot(wrist[0] - self.clear_anchor[0],
                                                    wrist[1] - self.clear_anchor[1]) > CLEAR_MAX_MOVE:
            self.clear_started, self.clear_anchor = now, wrist   # moved: restart the countdown
        self.clear_progress = min((now - self.clear_started) / CLEAR_HOLD_S, 1.0)
        if self.clear_progress >= 1.0:
            self.clear()
            self.clear_started, self.clear_progress, self.clear_latched = None, 0.0, True

    def reset(self):
        self.end_stroke(trim=False)
        self.pose.reset()
        self.cursor = None
        self.palm_center = None
        self.clear_started, self.clear_progress, self.clear_latched = None, 0.0, False

    def busy(self):
        return self.pen_down

    def on_key(self, key):
        ch = chr(key)
        if ch == "c":
            self.clear()
        elif ch == "z":
            self.undo()
        elif ch == "s":
            self.save()
        elif ch in DRAW_COLORS:
            self.end_stroke(trim=False)   # the next stroke uses the new color
            self.color_key = ch
            self.flash(f"Color: {DRAW_COLORS[ch][0]}")
        elif ch in "[]":
            step = DRAW_BRUSH_STEP if ch == "]" else -DRAW_BRUSH_STEP
            self.end_stroke(trim=False)
            self.size = min(max(self.size + step, DRAW_BRUSH_MIN), DRAW_BRUSH_MAX)
            self.flash(f"Brush: {self.size}px")

    def status(self):
        if self.clear_progress > 0:
            return "CLEARING...", (0, 165, 255)
        if self.pen_down:
            return "PEN DOWN", DRAW_COLORS[self.color_key][1]
        return f"PEN UP ({self.pose.stable.lower()})", WHITE

    def draw_overlay(self, frame):
        self.ensure_canvas(frame)
        if self.strokes:
            alpha = self.canvas[:, :, 3:4].astype(np.float32) * (DRAW_OPACITY / 255.0)
            frame[:] = (frame * (1 - alpha) + self.canvas[:, :, :3] * alpha).astype(np.uint8)
        brush = DRAW_COLORS[self.color_key][1]
        if self.cursor is not None:
            c = (int(self.cursor[0]), int(self.cursor[1]))
            if self.pen_down:
                cv2.circle(frame, c, max(self.size // 2, 4), brush, -1, cv2.LINE_AA)
                cv2.circle(frame, c, max(self.size // 2, 4) + 2, WHITE, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, c, max(self.size // 2, 4), GREY, 2, cv2.LINE_AA)
        if self.clear_progress > 0 and self.palm_center is not None:
            cv2.ellipse(frame, self.palm_center, (45, 45), -90, 0, 360 * self.clear_progress,
                        (0, 165, 255), 6, cv2.LINE_AA)
            left = CLEAR_HOLD_S * (1 - self.clear_progress)
            cv2.putText(frame, f"CLEAR in {left:.1f}s", (self.palm_center[0] - 55, self.palm_center[1] - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        # Palette + brush size, top right under the banner.
        w = frame.shape[1]
        x = w - 20 * len(DRAW_COLORS) - 70
        for k, (_, bgr) in DRAW_COLORS.items():
            cv2.rectangle(frame, (x, 46), (x + 16, 62), bgr, -1)
            if k == self.color_key:
                cv2.rectangle(frame, (x - 2, 44), (x + 18, 64), WHITE, 2)
            cv2.putText(frame, k, (x + 4, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
            x += 20
        cv2.putText(frame, f"{self.size}px", (x + 4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        hint = "1 finger = draw, peace = lift, palm 1s = clear | c z s 4-8 [ ]"
        cv2.putText(frame, hint, (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
        if self.message and time.monotonic() < self.message_until:
            cv2.putText(frame, self.message, (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)


def index_elevation(lm, aspect=4 / 3):
    """Angle of the index finger (knuckle -> tip) above level, in degrees:
    +90 straight up, -90 straight down, 0 sideways (either side)."""
    dx = (lm[8].x - lm[5].x) * aspect
    dy = lm[8].y - lm[5].y
    return math.degrees(math.atan2(-dy, abs(dx)))


class ScrollMode(Mode):
    """Index finger only. Tilt up / down = auto-scroll (faster the steeper);
    roughly level = manual scroll by moving the hand. The mouse never moves here."""
    name = "SCROLL"
    color = (255, 160, 0)

    def __init__(self):
        self.pose = PoseTracker()
        self.reset()

    def reset(self):
        self.pose.reset()
        self.zone = "IDLE"         # IDLE, MANUAL, AUTO_UP, AUTO_DOWN
        self.elevation = None
        self.speed = 0.0           # current auto speed, lines/s (signed: + = up)
        self.accum = 0.0
        self.palm_y = None
        self.last_time = None
        self.tip = self.knuckle = None
        self.last_dir = 0

    def update(self, landmarks, frame, now):
        dt = now - self.last_time if self.last_time is not None else 1 / 30
        self.last_time = now
        if landmarks is None:
            self.reset()
            return
        h, w = frame.shape[:2] if frame is not None else (3, 4)
        aspect = w / h
        if self.pose.update(classify_pose(landmarks, aspect)) != "POINT":
            self.zone, self.speed, self.accum, self.palm_y, self.elevation = "IDLE", 0.0, 0.0, None, None
            self.tip = self.knuckle = None
            return
        self.tip, self.knuckle = (landmarks[8].x, landmarks[8].y), (landmarks[5].x, landmarks[5].y)
        self.elevation = e = index_elevation(landmarks, aspect)
        self.zone = self.next_zone(e)
        if self.zone == "MANUAL":
            self.manual(landmarks[MIDDLE_MCP].y)
        else:
            self.palm_y = None           # restart manual tracking when you come back to level
            self.auto(e, dt)
        lines = int(self.accum)
        if lines:
            pyautogui.scroll(lines)
            self.accum -= lines
            self.last_dir = 1 if lines > 0 else -1

    def next_zone(self, e):
        """Hysteresis between the auto zones and the manual (level) zone."""
        if self.zone == "AUTO_UP":
            return "AUTO_UP" if e > AUTO_SCROLL_EXIT_ANGLE else ("AUTO_DOWN" if e < -AUTO_SCROLL_ANGLE else "MANUAL")
        if self.zone == "AUTO_DOWN":
            return "AUTO_DOWN" if e < -AUTO_SCROLL_EXIT_ANGLE else ("AUTO_UP" if e > AUTO_SCROLL_ANGLE else "MANUAL")
        if e > AUTO_SCROLL_ANGLE:
            return "AUTO_UP"
        if e < -AUTO_SCROLL_ANGLE:
            return "AUTO_DOWN"
        return "MANUAL"

    def auto(self, e, dt):
        t = (abs(e) - AUTO_SCROLL_ANGLE) / max(90.0 - AUTO_SCROLL_ANGLE, 1e-6)
        t = min(max(t, 0.0), 1.0)
        speed = AUTO_SCROLL_MIN_SPEED + t * (AUTO_SCROLL_MAX_SPEED - AUTO_SCROLL_MIN_SPEED)
        self.speed = speed * (1 if self.zone == "AUTO_UP" else -1) * AUTO_SCROLL_DIRECTION
        self.accum += self.speed * min(dt, 0.1)   # cap dt so a stall can't dump a burst of lines

    def manual(self, palm_y):
        """Same feel as the MOUSE-mode peace-sign scroll: hand up = page down."""
        self.speed = 0.0
        if self.palm_y is None:
            self.palm_y, self.accum = palm_y, 0.0
            return
        prev = self.palm_y
        self.palm_y = prev + SCROLL_SMOOTH_ALPHA * (palm_y - prev)
        dy = self.palm_y - prev
        if abs(dy) >= SCROLL_DEADZONE:
            self.accum += dy * SCROLL_SPEED * SCROLL_DIRECTION

    def status(self):
        if self.zone == "AUTO_UP":
            return f"AUTO UP {abs(self.speed):.0f}/s", self.color
        if self.zone == "AUTO_DOWN":
            return f"AUTO DOWN {abs(self.speed):.0f}/s", self.color
        if self.zone == "MANUAL":
            return "MANUAL - move hand", WHITE
        return "IDLE - point your index", GREY

    def draw_overlay(self, frame):
        h, w = frame.shape[:2]
        # Tilt gauge on the right: auto-up zone, manual zone, auto-down zone, and a needle.
        cx, cy, r = w - 70, h // 2, 55
        cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, -AUTO_SCROLL_ANGLE, (0, 120, 255), 8)   # up zone
        cv2.ellipse(frame, (cx, cy), (r, r), 0, -AUTO_SCROLL_ANGLE, AUTO_SCROLL_ANGLE, (90, 90, 90), 8)
        cv2.ellipse(frame, (cx, cy), (r, r), 0, AUTO_SCROLL_ANGLE, 90, (0, 120, 255), 8)     # down zone
        cv2.putText(frame, "up", (cx - 8, cy - r - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        cv2.putText(frame, "down", (cx - 18, cy + r + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        cv2.putText(frame, "manual", (cx - r - 62, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
        if self.elevation is not None:
            a = math.radians(-self.elevation)
            tip = (int(cx + (r - 6) * math.cos(a)), int(cy + (r - 6) * math.sin(a)))
            color = self.color if self.zone.startswith("AUTO") else WHITE
            cv2.line(frame, (cx, cy), tip, color, 4, cv2.LINE_AA)
            cv2.circle(frame, tip, 6, color, -1, cv2.LINE_AA)
        if self.tip is not None:
            p = (int(self.tip[0] * w), int(self.tip[1] * h))
            k = (int(self.knuckle[0] * w), int(self.knuckle[1] * h))
            cv2.arrowedLine(frame, k, p, self.color, 3, cv2.LINE_AA, tipLength=0.3)
        hint = "index up = auto up, down = auto down, level + move hand = manual"
        cv2.putText(frame, hint, (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)


def create_knob_targets():
    targets = {"volume": KnobTarget("volume", make_volume_backend),
               "brightness": KnobTarget("brightness", make_brightness_backend)}
    for t in targets.values():
        print(f"Knob {t.name}: " + (f"{t.backend} (now {t.value:.0f}%)" if t.available
                                    else f"DISABLED - {t.reason}"))
    return targets


def create_modes(screen):
    modes = [MouseMode(screen), VolumeMode(create_knob_targets()), DrawMode(), ScrollMode()]
    return {m.name: m for m in modes}


class GestureMenu:
    """The non-controlling hand picks a mode by holding up 1, 2, 3 fingers or an open palm (4).

    Guards against accidental switches: it needs a second hand in view, an exact
    finger pattern (index / +middle / +ring, others curled), POSE_STABLE_FRAMES
    of debounce, then MENU_HOLD_S of holding, with a progress ring shown.
    """
    COUNTS = {"POINT": 1, "PEACE": 2, "THREE": 3, "PALM": 4}
    COLOR = (255, 255, 0)

    def __init__(self):
        self.pose = PoseTracker()
        self.reset()

    def reset(self):
        self.pose.reset()
        self.target = None
        self.started = None
        self.progress = 0.0
        self.anchor = None         # menu hand wrist, frame px, for drawing

    def update(self, lm, frame_shape, now, current_mode):
        """Return a mode name to switch to, or None."""
        if lm is None:
            self.reset()
            return None
        h, w = frame_shape[:2]
        self.anchor = (int(lm[WRIST].x * w), int(lm[WRIST].y * h))
        count = self.COUNTS.get(self.pose.update(classify_pose(lm, w / h)))
        names = list(MODE_KEYS.values())
        target = names[count - 1] if count and count <= len(names) else None
        if target is None or target == current_mode:
            self.target, self.progress, self.started = target, 0.0, None
            return None
        if target != self.target or self.started is None:
            self.target, self.started = target, now
        self.progress = min((now - self.started) / MENU_HOLD_S, 1.0)
        if self.progress >= 1.0:
            self.progress = 0.0
            return target
        return None

    def draw_overlay(self, frame, current_mode):
        if self.anchor is None:
            return
        x, y = self.anchor
        cv2.putText(frame, "MENU HAND", (x - 45, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR, 1)
        if self.target is None:
            return
        if self.target == current_mode:
            cv2.putText(frame, f"{self.target} (current)", (x - 55, y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)
            return
        cv2.ellipse(frame, (x, y), (32, 32), -90, 0, 360 * self.progress, self.COLOR, 5, cv2.LINE_AA)
        cv2.putText(frame, f"-> {self.target}", (x - 45, y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR, 2)


class QuitGesture:
    """Either hand holds a fist for QUIT_HOLD_S -> quit. Shows a red countdown ring."""
    COLOR = (0, 0, 255)

    def __init__(self):
        self.reset()

    def reset(self):
        self.started = None
        self.last_seen = None
        self.progress = 0.0
        self.anchor = None

    def update(self, hands, frame_shape, now):
        """hands: landmark lists allowed to quit this frame. Return True to quit."""
        h, w = frame_shape[:2]
        fist = next((lm for lm in hands if classify_pose(lm, w / h) == "FIST"), None)
        if fist is None:
            if self.last_seen is None or now - self.last_seen > QUIT_GRACE_S:
                self.reset()
            return False
        self.anchor = (int(fist[MIDDLE_MCP].x * w), int(fist[MIDDLE_MCP].y * h))
        self.last_seen = now
        if self.started is None:
            self.started = now
        self.progress = min((now - self.started) / QUIT_HOLD_S, 1.0)
        return self.progress >= 1.0

    def draw_overlay(self, frame):
        if self.progress <= 0 or self.anchor is None:
            return
        x, y = self.anchor
        cv2.ellipse(frame, (x, y), (50, 50), -90, 0, 360 * self.progress, self.COLOR, 7, cv2.LINE_AA)
        left = QUIT_HOLD_S * (1 - self.progress)
        cv2.putText(frame, f"QUIT in {left:.1f}s", (x - 60, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR, 2)


LOG_FIELDS = ["session_start", "session_end", "duration_s", "raised_s",
              "longest_stretch_s", "nudge_fired", "max_nudge_level"]


def fz(seconds):
    """A fatigue time threshold, divided by FATIGUE_DEBUG_SPEEDUP for testing."""
    return seconds / max(FATIGUE_DEBUG_SPEEDUP, 1e-9)


class FatigueTracker:
    """Estimates how long the arm has been raised (hand visible) and nudges you to rest.

    Session = from the first detection until a real rest (hand gone REST_RESET_SECONDS).
    raised time counts visible time plus dropouts shorter than FATIGUE_GRACE_S.
    The main loop calls update(hand_detected, is_busy) once per frame; it's cheap.
    """
    GREEN, YELLOW, RED = (0, 200, 0), (0, 220, 255), (0, 0, 255)

    def __init__(self, notifier=None, log_path=None):
        self.enabled = FATIGUE_ENABLED
        self.notifier = notifier if notifier is not None else Notifier()
        self.log_path = log_path or FATIGUE_LOG
        self.day = date.today()
        self.total_today = 0.0
        self.rests_today = 0
        self.snooze_until = 0.0
        self.banner, self.banner_until = "", 0.0
        self._log_error = None
        self._now = 0.0
        self.new_session()

    def new_session(self):
        self.start_wall = None         # wall-clock time of the first detection
        self.end_wall = None           # wall-clock time of the last detection
        self.last_seen = None          # monotonic time of the last detection
        self.continuous = 0.0          # raised time this session (drives the nudges)
        self.stretch = 0.0             # current uninterrupted stretch (no gap > grace)
        self.longest_stretch = 0.0
        self.fired_level = 0           # highest nudge level delivered this session
        self.pending = 0               # level waiting for a good moment / end of snooze

    # --- thresholds ------------------------------------------------------------
    @staticmethod
    def rest_reset_s():
        return max(fz(REST_RESET_SECONDS), FATIGUE_GRACE_S)

    @staticmethod
    def level_for(raised):
        """0 = none, 1, 2, then 3, 4, 5... for the level-3 reminder and each repeat."""
        if raised < fz(NUDGE_LEVEL1_S):
            return 0
        if raised < fz(NUDGE_LEVEL2_S):
            return 1
        if raised < fz(NUDGE_LEVEL3_S):
            return 2
        return 3 + int((raised - fz(NUDGE_LEVEL3_S)) // fz(NUDGE_REPEAT_S))

    # --- per frame ---------------------------------------------------------------
    def update(self, hand_detected, is_busy, now=None):
        if not self.enabled:
            return
        now = time.monotonic() if now is None else now
        self._now = now
        self.roll_day()
        if hand_detected:
            if self.last_seen is None:
                if self.start_wall is None:
                    self.start_wall = time.time()
            else:
                gap = now - self.last_seen
                if gap <= FATIGUE_GRACE_S:     # visible, or a short dropout: still raised
                    self.continuous += gap
                    self.total_today += gap
                    self.stretch += gap
                    self.longest_stretch = max(self.longest_stretch, self.stretch)
                else:                          # lowered for a bit, but not long enough to rest
                    self.stretch = 0.0
            self.last_seen = now
            self.end_wall = time.time()
        elif self.last_seen is not None and now - self.last_seen >= self.rest_reset_s():
            self.end_session(rested=True)
        self.check_nudges(is_busy, now)

    def check_nudges(self, is_busy, now):
        due = self.level_for(self.continuous)
        if due > self.fired_level:
            self.pending = due
        if self.pending and not is_busy and now >= self.snooze_until:
            self.deliver(self.pending)
            self.fired_level, self.pending = self.pending, 0

    def deliver(self, level):
        minutes = self.continuous * FATIGUE_DEBUG_SPEEDUP / 60
        if level == 1:
            title, body = "Air Mouse", f"Arm up for {minutes:.0f} min, maybe give it a rest."
        elif level == 2:
            title, body = "Air Mouse: rest your arm", f"Arm up for {minutes:.0f} min. Lower it for a bit."
        else:
            title = "Air Mouse: please rest"
            body = (f"Arm up for {minutes:.0f} min without a real break. "
                    f"Lower your arm for {REST_RESET_SECONDS:.0f} s.")
        if FATIGUE_DEBUG_SPEEDUP != 1:
            body += f" [speedup x{FATIGUE_DEBUG_SPEEDUP:g}]"
        self.notifier.notify(title, body)
        # Level 3+ stays on screen until a rest or a snooze.
        self.banner = body
        self.banner_until = float("inf") if level >= 3 else self._now + 8.0
        print(f"Nudge level {level}: {body}")

    # --- sessions / day ------------------------------------------------------------
    def roll_day(self):
        today = date.today()
        if today != self.day:
            self.day, self.total_today, self.rests_today = today, 0.0, 0

    def end_session(self, rested):
        if self.start_wall is not None and self.continuous >= FATIGUE_MIN_LOG_S:
            self.write_log_row()
        if rested and self.start_wall is not None:
            self.rests_today += 1
            print(f"Rest detected - fatigue timer reset (rests today: {self.rests_today})")
        if self.banner_until == float("inf"):
            self.banner, self.banner_until = "", 0.0
        self.new_session()

    def write_log_row(self):
        row = [datetime.fromtimestamp(self.start_wall).isoformat(timespec="seconds"),
               datetime.fromtimestamp(self.end_wall or time.time()).isoformat(timespec="seconds"),
               f"{(self.end_wall or time.time()) - self.start_wall:.1f}",
               f"{self.continuous:.1f}", f"{self.longest_stretch:.1f}",
               "yes" if self.fired_level else "no", str(self.fired_level)]
        try:
            new_file = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0
            with open(self.log_path, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(LOG_FIELDS)
                w.writerow(row)
            self._log_error = None
        except OSError as e:
            if self._log_error is None:
                print(f"Couldn't write {os.path.basename(self.log_path)} ({e}); fatigue log skipped.")
            self._log_error = str(e)

    # --- user actions ----------------------------------------------------------------
    def snooze(self, now=None):
        now = time.monotonic() if now is None else now
        self.snooze_until = now + fz(SNOOZE_S)
        self.banner, self.banner_until = "", 0.0
        print(f"Fatigue nudges snoozed for {fz(SNOOZE_S) / 60:.1f} min")

    def toggle(self):
        if self.enabled:
            self.end_session(rested=False)     # log what we have
            self.banner, self.banner_until = "", 0.0
        else:
            self.new_session()
        self.enabled = not self.enabled
        print(f"Fatigue tracker {'on' if self.enabled else 'off'}")

    def close(self):
        """On quit: log the open session."""
        if self.enabled:
            self.end_session(rested=False)

    # --- debug window ------------------------------------------------------------------
    def draw_overlay(self, frame):
        h, w = frame.shape[:2]
        x2, y2 = w - 10, h - 10
        x1, y1 = x2 - 170, y2 - 34
        if not self.enabled:
            cv2.putText(frame, "fatigue OFF (f)", (x1, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
            return
        c = self.continuous
        color = self.GREEN if c < fz(NUDGE_LEVEL1_S) else self.YELLOW if c < fz(NUDGE_LEVEL2_S) else self.RED
        fill = min(c / fz(NUDGE_LEVEL1_S), 1.0)
        cv2.rectangle(frame, (x1, y2 - 10), (x2, y2), (60, 60, 60), -1)
        cv2.rectangle(frame, (x1, y2 - 10), (x1 + int((x2 - x1) * fill), y2), color, -1)
        shown = c * FATIGUE_DEBUG_SPEEDUP
        today = self.total_today * FATIGUE_DEBUG_SPEEDUP
        text = f"arm up {int(shown // 60)}:{int(shown % 60):02d}  today {int(today // 60)}m  rests {self.rests_today}"
        cv2.putText(frame, text, (x1 - 40, y1 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        if self._now < self.snooze_until:
            left = (self.snooze_until - self._now) * FATIGUE_DEBUG_SPEEDUP
            cv2.putText(frame, f"snoozed {int(left // 60)}:{int(left % 60):02d} (n)", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
        if self.banner and self._now < self.banner_until:
            self.draw_banner(frame)

    def draw_banner(self, frame):
        w = frame.shape[1]
        note = ""
        if self.notifier.error:
            note = f"(desktop notifications failed: {self.notifier.error[:40]})"
        top = 150
        cv2.rectangle(frame, (10, top), (w - 10, top + (54 if note else 36)), (0, 0, 160), -1)
        cv2.putText(frame, self.banner[:70], (20, top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        if note:
            cv2.putText(frame, note, (20, top + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1)


# ----------------------------------------------------------------------------- app state + drawing

@dataclass
class App:
    modes: dict
    mode_name: str
    paused: bool = False
    menu: GestureMenu = None
    quitter: QuitGesture = None
    fatigue: FatigueTracker = None

    @property
    def mode(self):
        return self.modes[self.mode_name]


def switch_mode(app, name):
    if name == app.mode_name:
        return
    app.mode.reset()              # e.g. MOUSE releases a held drag before handing over
    app.mode_name = name
    app.mode.reset()
    print(f"Mode: {name}")


def status_label(app, hand_found):
    if app.paused:
        return "PAUSED", (0, 165, 255)
    if not hand_found:
        return "NO HAND", GREY
    return app.mode.status()


def draw_landmarks(frame, lm, bone_color=(200, 200, 200)):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[c.start], pts[c.end], bone_color, 2)
    for i, p in enumerate(pts):
        color = (0, 0, 255) if i in (THUMB_TIP, INDEX_TIP) else (0, 255, 0)
        cv2.circle(frame, p, 6 if i in (THUMB_TIP, INDEX_TIP) else 3, color, -1)
    cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], (255, 0, 255), 1)


def draw_mode_banner(frame, app):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 36), app.mode.color, -1)
    cv2.putText(frame, f"MODE: {app.mode_name}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
    keys = "  ".join(f"{k}={n}" for k, n in MODE_KEYS.items())
    (tw, _), _ = cv2.getTextSize(keys, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, keys, (w - tw - 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)


def draw_hud(frame, app, hand_found, fps):
    h = frame.shape[0]
    draw_mode_banner(frame, app)
    label, color = status_label(app, hand_found)
    cv2.putText(frame, label, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"{fps:4.1f} fps  p=pause q=quit n=snooze f=fatigue", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)


def render_status_only(app, hand_found):
    img = np.zeros((90, 360, 3), dtype="uint8")
    draw_mode_banner(img, app)
    label, color = status_label(app, hand_found)
    cv2.putText(img, label, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img


# ----------------------------------------------------------------------------- main loop

def handle_keys(app):
    """Return False when the user asked to quit."""
    key = cv2.waitKey(1) & 0xFF
    if key == 27 and app.mode.on_escape():
        return True
    if key in (ord("q"), 27):
        return False
    if key == ord("p"):
        app.paused = not app.paused
        app.mode.reset()  # release a held drag on pause; don't jump on resume
        print("Paused" if app.paused else "Resumed")
    elif chr(key) in MODE_KEYS:
        switch_mode(app, MODE_KEYS[chr(key)])
    elif key == ord("n"):
        app.fatigue.snooze()
    elif key == ord("f"):
        app.fatigue.toggle()
    elif key != 255:
        app.mode.on_key(key)
    return True


def run(max_frames=None):
    enable_windows_dpi_awareness()
    ensure_model()
    screen = get_screen_region()
    print(f"Screen region (left, top, w, h): {screen}")
    cap = open_camera()
    landmarker = create_landmarker()
    app = App(modes=create_modes(screen), mode_name=START_MODE, menu=GestureMenu(), quitter=QuitGesture(),
              fatigue=FatigueTracker())
    router = HandRouter()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WINDOW_NAME, 40, 60)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)  # keep it above other apps
    start = last = time.monotonic()
    fps, frames = 0.0, 0
    print(f"Running in {app.mode_name} mode. Focus the 'Air Mouse' window: "
          "1/2/3 = mode, p = pause, q = quit (or hold a fist).")
    try:
        while max_frames is None or frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                continue
            if MIRROR:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-3))
            last = now
            frames += 1

            hands = get_hands(landmarker, frame, int((now - start) * 1000))
            lm, menu_lm = router.split(hands)
            if app.paused:
                app.mode.preview(lm)
                app.menu.reset()
                app.quitter.reset()
            else:
                app.mode.update(lm, frame, now)  # only the active mode touches hand data
                if GESTURE_MENU:
                    picked = app.menu.update(menu_lm, frame.shape, now, app.mode_name)
                    if picked:
                        switch_mode(app, picked)
                if QUIT_GESTURE:
                    quit_hands = [h for h in (None if app.mode.engaged() else lm, menu_lm) if h is not None]
                    if app.quitter.update(quit_hands, frame.shape, now):
                        print("Quit gesture (fist held). Exiting.")
                        break

            # Fatigue: any visible hand counts as raised, including while paused or clutched.
            app.fatigue.update(bool(hands), (not app.paused) and app.mode.busy(), now)

            if SHOW_DEBUG_WINDOW:
                if lm is not None:
                    draw_landmarks(frame, lm)
                if menu_lm is not None and GESTURE_MENU:
                    draw_landmarks(frame, menu_lm, GestureMenu.COLOR)
                app.mode.draw_overlay(frame)
                if GESTURE_MENU:
                    app.menu.draw_overlay(frame, app.mode_name)
                if QUIT_GESTURE:
                    app.quitter.draw_overlay(frame)
                draw_hud(frame, app, lm is not None, fps)
                app.fatigue.draw_overlay(frame)
                cv2.imshow(WINDOW_NAME, frame)
            else:
                cv2.imshow(WINDOW_NAME, render_status_only(app, lm is not None))
            if not handle_keys(app):
                break
    except pyautogui.FailSafeException:
        print("FAILSAFE triggered (cursor hit a screen corner). Exiting.")
    finally:
        app.mode.reset()
        app.fatigue.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
    print(f"Stopped after {frames} frames.")


def fmt_duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m {seconds % 60:02d}s"


def fatigue_report(path=None, days=7, today=None):
    """Print a summary of fatigue_log.csv (standard library only). Returns the text."""
    path = path or FATIGUE_LOG
    today = today or date.today()
    if not os.path.exists(path):
        text = f"No fatigue log yet ({os.path.basename(path)} doesn't exist). Use the air mouse for a while first."
        print(text)
        return text
    sessions, skipped = [], 0
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    start = datetime.fromisoformat(row["session_start"])
                    sessions.append((start, float(row["raised_s"]), row.get("nudge_fired") == "yes"))
                except (KeyError, TypeError, ValueError):
                    skipped += 1
    except (OSError, csv.Error, UnicodeDecodeError) as e:
        text = f"Couldn't read {os.path.basename(path)}: {e}"
        print(text)
        return text
    per_day = defaultdict(float)
    for start, raised, _ in sessions:
        per_day[start.date()] += raised
    lines = [f"Fatigue report ({os.path.basename(path)})", "", f"Raised time, last {days} days:"]
    for i in range(days - 1, -1, -1):
        d = today - timedelta(days=i)
        total = per_day.get(d, 0.0)
        bar = "#" * min(int(total / 300), 40)          # one # per 5 minutes
        lines.append(f"  {d:%a %b %d}  {fmt_duration(total):>8}  {bar}")
    if sessions:
        raised = [r for _, r, _ in sessions]
        longest = max(sessions, key=lambda x: x[1])
        nudged = sum(1 for *_, n in sessions if n)
        lines += ["",
                  f"Sessions logged:        {len(sessions)}  ({nudged} with a nudge)",
                  f"Average session length: {fmt_duration(sum(raised) / len(raised))}",
                  f"Longest session:        {fmt_duration(longest[1])} on {longest[0]:%a %b %d %H:%M}"]
    else:
        lines += ["", "No sessions logged yet."]
    if skipped:
        lines.append(f"(skipped {skipped} unreadable row{'s' if skipped != 1 else ''})")
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    if "--fatigue-report" in sys.argv:
        fatigue_report()
        sys.exit(0)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    frames_arg = int(args[0]) if args else None  # optional: run N frames then exit
    run(frames_arg)
