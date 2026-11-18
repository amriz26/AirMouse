# Air Mouse

Move the cursor with your index fingertip. Pinch thumb and index together to click.

- **Quick pinch and release**: left click
- **Pinch held over 300 ms**: drag (mouse down), release to drop (mouse up)
- **`1` / `2` / `3`**: switch to MOUSE / VOLUME / DRAW mode. The mouse doesn't respond in VOLUME or DRAW.
- **Gesture menu**: with your other hand, hold up 1, 2 or 3 fingers for about 1 s to switch to that mode.
- **DRAW mode**: index finger only = draw, peace sign = lift the pen, open palm held 1 s = clear. Keys: `c` clear, `z` undo, `s` save PNG, `4`–`8` colors, `[` `]` brush size.
- **VOLUME mode**: pinch to grab a virtual knob, rotate your hand to turn it, and open your fingers to let go. Or circle your index finger in the air: clockwise = up, counter-clockwise = down. **`b`** switches the knob between volume and brightness.
- **`k`** (MOUSE mode): calibrate the active region to your reach. **`r`** resets it to the defaults, and **Esc** cancels a calibration that's running.
- **SCROLL mode** (`0`, or the menu hand's open palm): index tilted up or down auto-scrolls, faster the steeper. Index level plus hand movement scrolls manually.
- **Scroll** (MOUSE mode): a peace sign plus vertical hand movement scrolls. Hand up scrolls down, and hand down scrolls up.
- **Clutch** (MOUSE mode): make a fist to freeze the cursor, reposition your arm, then open your hand and carry on from the same spot. **`x`** toggles it.
- **Fist held 1.5 s**: quits. In MOUSE mode with the clutch on, the controlling hand's fist is the clutch, so quit with the other hand's fist or `q`. It works with either hand, but not while the controlling hand is pinching or while paused.
- **`p`**: pause/resume, **`q`** or **Esc**: quit. The OpenCV window must have keyboard focus for these.
- **Emergency stop**: move the real mouse or trackpad cursor into any screen corner. This triggers pyautogui's FAILSAFE and the script exits. Ctrl+C in the terminal also works.

Hand tracking uses the MediaPipe **Tasks API `HandLandmarker`** in VIDEO mode. mediapipe 1.2.0 no longer ships `mp.solutions`, so no fallback is possible. The model file (`models/hand_landmarker.task`, ~8 MB) downloads automatically on first run if it's missing.

## Setup

```bash
python3.11 -m venv .venv          # or: uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt   # or: uv pip install -r requirements.txt
python air_mouse.py
```

`python air_mouse.py 300` runs 300 frames and exits, which is handy as a smoke test.

Install only `opencv-contrib-python`, which mediapipe pulls in. If `opencv-python` is installed alongside it, the two overwrite each other's `cv2` and imports can break.

## macOS permissions

The app that launches Python (Terminal, iTerm, VS Code, Claude) needs both of these:

1. **Camera**: System Settings > Privacy & Security > **Camera**. macOS asks on first run. If you clicked "Don't Allow", turn it on here. Without it the script exits with "Could not read from camera".
2. **Accessibility**: System Settings > Privacy & Security > **Accessibility**. Add your terminal app and turn it on. Without it the script runs and tracks your hand, but **the cursor doesn't move and clicks do nothing**, with no error message.

**Quit and reopen the terminal app** after you change either setting, because macOS only applies the change on restart.

## High-DPI and multiple monitors

- **macOS Retina**: pyautogui works in logical points, not physical pixels (for example 1512x982 on a 14" MacBook Pro), and `moveTo` uses the same units, so the mapping is already correct. At startup the script prints the screen region it's using. Check that it matches your display's "looks like" resolution.
- **Windows with display scaling**: the script calls `SetProcessDpiAwareness(2)` before reading the screen size, so pyautogui sees real pixels. If the cursor stops short of the right or bottom edge, scaling is still distorting the size. Set `SCREEN_REGION` by hand.
- **Multiple monitors**: `pyautogui.size()` reports **only the primary display**. To target another monitor, set `SCREEN_REGION = (left, top, width, height)` in that monitor's coordinates. On macOS, a display to the left of or above the primary one has negative coordinates. To find a monitor's origin, run `python -c "import pyautogui,time; time.sleep(3); print(pyautogui.position())"` and park the mouse in that monitor's top-left corner during the 3 seconds.

## Tuning guide

All values are in the CONFIG block at the top of `air_mouse.py`.

| Symptom | Change |
|---|---|
| Cursor jitters when your hand is still | Lower `SMOOTH_ALPHA_SLOW` (0.12 → 0.08), or raise `DEADZONE_PX` (1.5 → 3) |
| Cursor feels laggy on big, fast moves | Raise `SMOOTH_ALPHA_FAST` (0.65 → 0.8), or lower `SPEED_FAST` so "fast" kicks in sooner |
| Small, precise moves feel sluggish | Raise `SMOOTH_ALPHA_SLOW` a little, or lower `SPEED_SLOW` |
| Can't reach the screen edges or corners | Raise the `REGION_MARGIN_*` values so the active rectangle is smaller |
| Cursor moves too fast, too twitchy overall | Lower the `REGION_MARGIN_*` values so the active rectangle is bigger |
| Clicks fire without you pinching | Lower `PINCH_START` (0.28 → 0.22) |
| You have to squeeze hard to click | Raise `PINCH_START` (0.28 → 0.33). Keep it below `PINCH_RELEASE` |
| Drag drops by itself mid-drag | Raise `PINCH_RELEASE` (0.40 → 0.48) |
| Pinch status flickers on and off | Widen the gap between `PINCH_START` and `PINCH_RELEASE` |
| Quick clicks turn into drags | Raise `DRAG_DELAY_MS` (300 → 400) |
| Cursor jumps right after a click | Raise `RELEASE_FREEZE_MS` (120 → 200) |
| Dragging feels too sluggish | Raise `DRAG_ALPHA_SCALE` toward 1.0 |
| Wrong camera opens | Change `CAMERA_INDEX` (iPhone Continuity Camera often takes 0) |

The debug window shows the live normalized pinch distance. Watch it with your hand open and with a light pinch, then set `PINCH_START` and `PINCH_RELEASE` between those two readings.

## Calibration

Press `k` in MOUSE mode. You then point to the four corners of your comfortable range in order (top-left, top-right, bottom-right, bottom-left), `CAL_CORNER_S` (2.5 s) each. While calibrating, MouseMode sends hand data to the calibrator instead of the cursor, so the mouse can't move or click. The quit fist is blocked too.

- **Sampling**: for each corner, the index fingertip (landmark 8, in mirrored-frame coordinates) is sampled after the first `CAL_IGNORE_S` (0.5 s). The corner's point is the **median** of those samples, so one bad frame can't pull it off.
- **Retries**: the corner repeats, with the reason shown on screen, if there are fewer than `CAL_MIN_SAMPLES` samples ("no hand seen"), or if 80% of the samples aren't within `CAL_MAX_SCATTER` of the median ("hold still").
- **Region**: the min/max of x and y across all four points, then shrunk inward by `CAL_PADDING` (5% of its size per side). The padding means the cursor reaches the screen edge just *before* your fingertip reaches the spot you calibrated. If the result is smaller than `CAL_MIN_SIZE` (15% of the frame) in either direction, it's rejected and the previous region is kept.
- **Saving**: the region is saved atomically to `calibration.json` next to the script and loaded on startup. A missing, corrupt or out-of-range file falls back to the `REGION_MARGIN_*` defaults with a message, and never crashes. `r` deletes the file and goes back to the defaults. Esc cancels without saving.
- **`LOCK_ASPECT_RATIO`** (default `False`): grows the region around its center until its shape matches the screen's, capped at the frame edges. It applies to the default region too. The saved file always holds the unlocked region, so you can toggle this setting without recalibrating.

| Symptom | Change |
|---|---|
| Corners keep retrying with "hold still" | Raise `CAL_MAX_SCATTER` (0.05 → 0.08), or brace your elbow |
| Not enough time to move between corners | Raise `CAL_IGNORE_S` or `CAL_CORNER_S` |
| Still can't quite reach the edges | Raise `CAL_PADDING` (0.05 → 0.1) |
| Edges come too early, cursor feels fast | Lower `CAL_PADDING` |

## SCROLL mode

SCROLL mode only acts on the index-only pose (`POINT`, debounced). The mouse never moves or clicks in this mode.

- **Tilt**: the angle of the index knuckle → tip vector above level, aspect-corrected. +90° is straight up, −90° straight down, and 0° is sideways in either direction.
- **Zones**: above `AUTO_SCROLL_ANGLE` (35°) is auto up, and below −35° is auto down. To leave an auto zone you have to come back under `AUTO_SCROLL_EXIT_ANGLE` (25°), which gives hysteresis. Anything in between is manual.
- **Auto speed**: from `AUTO_SCROLL_MIN_SPEED` (4 lines/s) at 35° to `AUTO_SCROLL_MAX_SPEED` (30 lines/s) pointing straight up or down. `dt` is capped, so a stall can't dump a burst of lines. `AUTO_SCROLL_DIRECTION` flips it.
- **Manual**: the same as the MOUSE peace-sign scroll (`SCROLL_SPEED`, `SCROLL_DIRECTION`, `SCROLL_DEADZONE`, `SCROLL_SMOOTH_ALPHA`). Manual tracking restarts each time you return to level, so the move from tilted to level doesn't scroll.
- **Switching**: key `0` (DRAW's 4–8 color keys are unchanged), or the gesture menu's 4th count (an open palm on the menu hand).

| Symptom | Change |
|---|---|
| Auto-scroll starts too easily | Raise `AUTO_SCROLL_ANGLE` (35 → 45) |
| Auto-scroll too fast or too slow | `AUTO_SCROLL_MAX_SPEED` / `AUTO_SCROLL_MIN_SPEED` |
| Flickers between auto and manual | Widen the gap to `AUTO_SCROLL_EXIT_ANGLE` |

## Scroll (MOUSE mode)

A **peace sign** (index + middle extended, ring + pinky curled) on the controlling hand, debounced over `POSE_STABLE_FRAMES`, starts scrolling. The checks run in this order: clutch, then scroll, then pinch.

- **Holding the cursor**: on start, the cursor holds (a drag gets `mouseUp`, and a half-made pinch is discarded). Frames where the peace sign is still forming also hold still.
- **Scrolling**: while scrolling, the palm's (middle knuckle's) height is EMA-smoothed (`SCROLL_SMOOTH_ALPHA`). Per-frame changes above `SCROLL_DEADZONE` add `dy × SCROLL_SPEED × SCROLL_DIRECTION` lines, which are sent through `pyautogui.scroll` in whole lines. Hand up gives a negative value, which scrolls the page down.
- **Ending**: when you leave the peace sign, the cursor re-anchors with the same offset mechanism as the clutch, so it continues from where it held. Losing the hand ends the scroll without a jump-scroll when it comes back. A mode switch or pause also ends it.
- **Fatigue**: scrolling counts as `busy()`, so fatigue nudges wait until you finish.

| Symptom | Change |
|---|---|
| Scrolls too fast or too slow | Raise or lower `SCROLL_SPEED` (60 lines per frame-height) |
| Direction feels backwards (macOS natural scrolling) | `SCROLL_DIRECTION = -1` |
| Page creeps while your hand is still | Raise `SCROLL_DEADZONE` |
| Turn it off | `SCROLL_ENABLED = False` |

## Clutch (re-anchoring offset)

The cursor is an absolute mapping from fingertip to screen. A clutch that only froze the cursor would make it teleport on release, so the clutch uses an offset instead.

- **Fist**: all four fingers have `|wrist → tip| < |wrist → PIP| × CLUTCH_CURL_RATIO`. These are distances, so it works at any rotation, and the thumb is ignored. The clutch engages after `CLUTCH_ENGAGE_FRAMES` fist frames. It releases after `CLUTCH_RELEASE_FRAMES` frames where any finger is past `CLUTCH_OPEN_RATIO`, which gives hysteresis.
- **Clutch beats pinch**: the clutch check runs before pinch detection. From the first fist-looking frame, the cursor holds and pinches are ignored, because a fist can look like a pinch. On engage, a drag gets `mouseUp`, and a pinch that started as the fist formed is discarded, not clicked. After release, a pinch only counts once thumb and index have stayed apart for `CLUTCH_RELEASE_FRAMES` frames, because opening a fist can pass through a pinch.
- **On release**: `offset = frozen cursor − mapped fingertip`, and every later position is `clamp(mapped + offset)`. The smoothing state is reset, so the first frame lands exactly on the frozen spot.
- **Bleed**: an offset of +dx hides the screen strip from 0 to dx. While the hand moves *toward* the hidden side, that axis of the offset shrinks by `exp(−CLUTCH_BLEED_RATE × px moved)`. The cursor moves a little faster that way until the edge is reachable again, and the last few px snap to zero. Moving away from the hidden side stays exactly 1:1. Set `CLUTCH_BLEED_RATE = 0` to disable bleeding.
- **Offset resets**: calibration start and finish, a mode switch or pause, `r`, and the hand being gone for more than `CLUTCH_LOST_RESET_S`.
- **No fist = no change**: in tests with no fists, MOUSE makes exactly the same mouse calls with the clutch on and off.

| Symptom | Change |
|---|---|
| Clutch engages when you don't mean it | Lower `CLUTCH_CURL_RATIO` (1.0 → 0.9), or raise `CLUTCH_ENGAGE_FRAMES` |
| Fist isn't detected | Raise `CLUTCH_CURL_RATIO` (→ 1.05) |
| Clutch releases too slowly or too easily | Lower or raise `CLUTCH_RELEASE_FRAMES`, or `CLUTCH_OPEN_RATIO` |
| Cursor runs too fast toward the hidden side | Lower `CLUTCH_BLEED_RATE` |

## Fatigue tracker

The camera can only see whether a hand is visible, so "hand detected" (either hand, any mode) stands in for "arm raised". Clutch time and PAUSED time count as raised too.

- **Timer**: raised time adds up while a hand is visible, and also through dropouts shorter than `FATIGUE_GRACE_S` (3 s). A longer gap pauses it without resetting it. Only `REST_RESET_SECONDS` (20 s) with no hand counts as a real rest. That resets the timer and the nudge levels, and adds to the rest count.
- **Nudges**: none before `NUDGE_LEVEL1_S` (10 min). Then level 2 at `NUDGE_LEVEL2_S` (15 min), and level 3 at `NUDGE_LEVEL3_S` (20 min) plus every `NUDGE_REPEAT_S` (5 min) after. Each fires once per session. Level 3 keeps a banner on screen until you rest or snooze.
- **Timing**: a nudge that comes due while the active mode reports `busy()` waits until you're done. Busy means a MOUSE pinch, drag or calibration, a VOLUME knob grab or circling, or a DRAW stroke. The clutch doesn't count as busy. While snoozed (`n`, `SNOOZE_S`), nudges wait. When the snooze ends, only the highest level that came due is sent, so you don't get a backlog.
- **Delivery**: `send_notification()` uses `osascript display notification` on macOS, `plyer` (or `win10toast`) on Windows and `notify-send` on Linux. It runs on a background thread, so `notify()` returns in microseconds. These are passive OS notifications: they don't take keyboard focus or move the cursor. If notifications are unavailable at startup or fail later, the terminal says so clearly, and the in-window banner adds the reason. The banner always shows either way. `NUDGE_SOUND` (off by default) adds a sound.
- **Log**: `fatigue_log.csv` gets one row per session (`session_start, session_end, duration_s, raised_s, longest_stretch_s, nudge_fired, max_nudge_level`). The header is written if the file is missing or empty. Rows are written when a session ends, when you press `f`, and on quit. `duration_s` is wall time including short lowerings, and `longest_stretch_s` is the longest stretch with no gap over the grace period. Sessions under `FATIGUE_MIN_LOG_S` (5 s) are skipped as stray detections. A corrupt, locked or unwritable file prints one warning and never crashes.
- **`--fatigue-report`**: raised time per day for the last 7 days, average session length, and longest session. It uses the standard library only, and unreadable rows are skipped and counted.
- **`FATIGUE_DEBUG_SPEEDUP`**: divides the nudge times, `REST_RESET_SECONDS` and `SNOOZE_S`, but not the grace period, which is about camera dropouts rather than your arm. With a speedup, a rest takes max(20 s / speedup, 3 s). Displayed times and messages are scaled back up, so "5 min" still reads as 5 min.

## Volume knob

- **Hand angle**: `atan2` of the wrist → middle-knuckle vector. Every frame the angle is unwrapped across ±180° and then EMA-smoothed (`KNOB_ANGLE_ALPHA`).
- **Relative control**: on grab, the knob reads the current system value and the current angle. While you hold the pinch, the value is `start + rotation × 100 / KNOB_DEGREES_FOR_FULL`, clamped to 0–100.
- **Stops**: when you hit 0 or 100, the starting point moves with you, so turning back responds immediately, like a real knob with end stops.
- **Fewer updates**: a new value is only sent when it changes by `KNOB_MIN_STEP` (1%) or more.
- **Background thread**: values are applied on a separate thread that always takes the latest one, so a slow backend never stalls the camera.
- **Polling**: while the knob is released, it re-reads the value every `KNOB_POLL_S`, so the display picks up changes you make with the keyboard.

| OS | Volume | Brightness |
|---|---|---|
| macOS | `osascript` (`set volume output volume N`), built in | **Disabled**: `screen_brightness_control` has no macOS support |
| Windows | `pycaw` (master endpoint volume) | `screen_brightness_control` |
| Linux | `pactl` on the default sink, falling back to `amixer` Master | `screen_brightness_control` |

At startup the terminal prints which backend is in use, or why one is disabled. To swap a backend, edit `make_volume_backend()` or `make_brightness_backend()`. Each returns a `(get, set, description)` triple working in 0–100. `KNOB_TARGET` sets which target the knob starts on.

| Symptom | Change |
|---|---|
| Knob jitters while you hold still | Lower `KNOB_ANGLE_ALPHA` (0.35 → 0.2) |
| Knob lags behind your hand | Raise `KNOB_ANGLE_ALPHA` |
| Too sensitive or not sensitive enough | Raise or lower `KNOB_DEGREES_FOR_FULL` |
| Turns the wrong way | Set `KNOB_DIRECTION = -1` |
| Circling changes volume too slowly or too fast | Raise or lower `CIRCLE_PERCENT_PER_TURN` (20) |
| Circles aren't picked up | Lower `CIRCLE_MIN_RADIUS` (make bigger circles first), or lower `CIRCLE_DEADBAND_DEG` |
| Small finger wobbles change the volume | Raise `CIRCLE_DEADBAND_DEG` or `CIRCLE_MIN_RADIUS` |
| Circling direction is reversed | Set `CIRCLE_DIRECTION = -1` |

**Circle gesture:** this only runs when exactly your index finger is up and you're not pinching. The circle's center is the average fingertip position over the last `CIRCLE_WINDOW_S`. The angle of the center → fingertip vector is tracked across ±180°. Positive (clockwise on screen) turns the value up. Straight swipes and a still finger don't rotate that vector, so they don't change the volume. The first `CIRCLE_DEADBAND_DEG` of each gesture is ignored, so a wobble never changes anything.

**Quit gesture:** a fist means all four fingers are curled. Hold it for `QUIT_HOLD_S` to quit. The controlling hand doesn't count while it's pinching (MOUSE click or drag, VOLUME grab), because a tight pinch can look like a fist. The menu hand always counts. Set `QUIT_GESTURE = False` to turn it off.

## Modes and adding a new one

Each mode is a class in `air_mouse.py` that subclasses `Mode` and implements this interface:

| Method | When it's called |
|---|---|
| `update(landmarks, frame, now)` | Every frame while the mode is active and not paused. `landmarks` is `None` when no hand is visible |
| `preview(landmarks)` | Every frame while paused. Update readouts only, never act |
| `reset()` | On switching away from the mode, on switching to it, and on pause/resume. Release anything held (MOUSE releases a drag here) |
| `on_key(key)` | A key the main loop doesn't handle, sent only to the active mode (VOLUME uses `b`) |
| `status()` | Returns `(label, BGR color)` for the status line |
| `draw_overlay(frame)` | Draws the mode's own guides on the debug frame |

The main loop only calls the active mode, so inactive modes never see hand data. To add a fourth mode, register it in `create_modes()` and give it a key in `MODE_KEYS`. A fourth mode is also reachable from the gesture menu only if you add a 4-finger count to `GestureMenu.COUNTS`; right now 4 fingers is the palm, which clears in DRAW.

## Finger poses, DRAW mode and the gesture menu

- **Finger extended**: a finger counts as extended when `|wrist → tip| > |wrist → PIP| × FINGER_EXTEND_RATIO`. Distances from the wrist stay the same when the hand rotates or moves closer, so this works for tilted and sideways hands too. x is scaled by the frame aspect ratio so x and y distances match. The thumb is ignored because it's unreliable.
- **Poses**: index only = `POINT`, index + middle = `PEACE`, plus ring = `THREE`, all four = `PALM`, anything else = `OTHER`. A pose must hold for `POSE_STABLE_FRAMES` frames before it counts, which filters out one-frame glitches.
- **Strokes**: each stroke is a list of points with its own color and size, and the canvas is a BGRA numpy array the size of the frame. Undo removes the last stroke and redraws the canvas. Lifting the pen drops the last few points, the ones drawn while the peace sign was still forming, so strokes don't get a tail. The fingertip uses the same adaptive smoothing as MOUSE, with `DRAW_SPEED_SCALE` to account for frame px vs screen px.
- **Two hands**: `NUM_HANDS = 2`. Whichever hand is closest to where the controlling hand was last frame keeps control. The other hand is the menu hand and is drawn in cyan. A menu switch needs an exact finger pattern, held for `MENU_HOLD_S`, and it ignores the number of the mode you're already in. Set `GESTURE_MENU = False` to turn the menu off.

| Symptom | Change |
|---|---|
| Fingers read as extended when curled, or the reverse | Raise or lower `FINGER_EXTEND_RATIO` (1.15) |
| Pen goes up or down by itself | Raise `POSE_STABLE_FRAMES` (3 → 5) |
| Strokes are wobbly | Lower `SMOOTH_ALPHA_SLOW`, or raise `DRAW_SPEED_SCALE` |
| Strokes lag behind your finger | Lower `DRAW_SPEED_SCALE` |
| Menu switches too easily or too slowly | Raise or lower `MENU_HOLD_S` |
| Clear triggers too easily | Raise `CLEAR_HOLD_S`, or lower `CLEAR_MAX_MOVE` |

## How it works

- The fingertip position inside the active region (blue rectangle) maps linearly to the screen. Anything outside it clamps to the nearest edge. The cursor stays `EDGE_INSET_PX` away from the exact corners so your hand can't trip FAILSAFE by accident.
- Smoothing is an EMA whose weight slides between `SMOOTH_ALPHA_SLOW` and `SMOOTH_ALPHA_FAST` depending on hand speed in screen px/s.
- Pinch distance is `|thumb_tip − index_tip| / |wrist − middle_knuckle|`. Two thresholds give hysteresis, so the state doesn't flicker at the boundary.
- When a pinch starts, the cursor freezes. If the pinch is released within `DRAG_DELAY_MS`, that's a click, and the cursor stays frozen for another `RELEASE_FREEZE_MS`. If the pinch is held longer, that's mouse down, and the cursor follows at damped speed until you release.
- If the hand is lost while dragging, or you pause, the script releases the mouse button so it never gets stuck down.
