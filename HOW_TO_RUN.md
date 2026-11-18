# How to run Air Mouse

Steer the cursor with your index finger and pinch to click, using your webcam.

## 1. Start it

Open Terminal, go to this folder, and run:

```bash
cd ~/Documents/SWE/"Personal SWE Projects"/"air mouse"
.venv/bin/python air_mouse.py
```

A window titled **Air Mouse** opens at the top left of your screen with your camera feed. If you don't see it, click the **Python** icon in the Dock.

You can ignore lines in Terminal that start with `W0000` or `I0000`. They're normal MediaPipe log messages.

## 2. One-time permissions (macOS)

The app you run it from (Terminal, iTerm, VS Code or Claude) needs two permissions, both in **System Settings > Privacy & Security**:

| Permission | Without it |
|---|---|
| **Camera** | The script exits with "Could not read from camera" |
| **Accessibility** | The camera window works, but the cursor doesn't move |

After you turn either one on, quit that app with Cmd+Q and reopen it.

## 3. Use it

The app has three modes. The colored bar at the top of the window shows which one is active. Only the active mode reacts to your hand.

| Key | Mode | What it does |
|---|---|---|
| `1` | **MOUSE** (default) | Moves the cursor, clicks and drags (steps below) |
| `2` | **VOLUME** | A virtual knob (steps below). The mouse doesn't respond |
| `0` | **SCROLL** | Scroll with your index finger: tilt it up or down to auto-scroll, or hold it level and move your hand to scroll manually. The mouse doesn't respond |
| `3` | **DRAW** | Draw in the air (steps below). The mouse doesn't respond |

**Switch modes with your other hand:** while one hand is in use, raise your other hand and hold up 1, 2 or 3 fingers (index, then add middle, then ring), or an open palm for SCROLL, for about 1 second. A cyan ring around that hand fills up, then the mode switches. Either hand can be the menu hand: whichever hand was already in view keeps control. Number keys still work too.

If you switch away from MOUSE during a drag, the mouse button is released.

In MOUSE mode:

1. Hold your hand up so your **index fingertip** is inside the **blue rectangle**. The rectangle maps to your whole screen.
2. **Move**: point and move your index finger. The status reads **TRACKING**.
3. **Click**: pinch thumb and index together and let go quickly. The status flashes **PINCHED**.
4. **Drag**: pinch and hold for about 0.3 s. The status reads **DRAGGING**. Move, then let go to drop.
5. **Stop controlling** without quitting: click the Air Mouse window, then press **`p`**. Press `p` again to resume.

**SCROLL mode (press `0`, or show an open palm with your other hand):** use your index finger only.

- **Auto-scroll:** point it up to scroll up, or down to scroll down. The steeper you point, the faster it scrolls, from about 4 to 30 lines per second.
- **Manual:** hold it roughly level (sideways) and move your hand. Up scrolls the page down, and down scrolls it up.
- The gauge on the right shows your finger's tilt and which zone you're in.
- If auto-scroll goes the wrong way, set `AUTO_SCROLL_DIRECTION = -1`.

**Scroll (MOUSE mode):** hold up a **peace sign** (index + middle fingers) and move your hand. Moving it up scrolls the page down, and moving it down scrolls the page up. The cursor stays still while you scroll, so the page under it scrolls, and it carries on from the same spot when you point again. If the direction feels backwards, set `SCROLL_DIRECTION = -1`.

**Clutch: "lift the mouse" (MOUSE mode):** make a fist and the cursor freezes (status **CLUTCHED**). Move your arm wherever is comfortable, then open your hand. The cursor carries on from where it froze. Use this like picking up a real mouse when you run out of desk.

- A grey "ghost" ring shows where the cursor would be without the offset.
- The offset fades out as you move toward the edge it cut off, so every corner stays reachable.
- Press **`x`** to turn the clutch off and on.
- In MOUSE mode a fist on your controlling hand is the clutch, not quit. To quit, use a fist on your other hand, or `q`. With the clutch off (`x`), a fist quits again.

**Calibrate your reach (MOUSE mode, recommended on first run):** press **`k`** and follow the prompts. For each corner in turn (top-left, top-right, bottom-right, bottom-left), point to the edge of where your hand comfortably reaches and hold still for about 2.5 seconds. The whole flow takes about 10 seconds. The mouse doesn't move during it. If no hand is seen or your hand moves too much, that corner repeats and the window says why.

- The result is saved to `calibration.json` and loaded automatically next time.
- **Esc** cancels without changing anything.
- **`r`** resets to the default region.

In VOLUME mode:

1. **Grab**: pinch thumb and index together. The knob on the right turns yellow and shows **GRABBED**.
2. **Turn**: keep pinching and rotate your hand like turning a dial. Clockwise (as seen in the window) turns it up. A half turn (180°) covers 0 to 100%.
3. **Let go**: open your fingers. The volume stays where it is. The next grab starts from the current volume, so it never jumps.
4. **Or circle your finger**: no pinch needed. Raise only your index finger and draw circles in the air. Clockwise (as seen in the window) turns the volume up, counter-clockwise turns it down. Each full circle is 20%. Your fingertip's path is drawn on screen, and a small wobble of under about ⅛ of a turn is ignored.
5. Press **`b`** to switch the knob between volume and brightness. Brightness is disabled on macOS, and the window says why.

In DRAW mode:

1. **Draw**: raise only your index finger (curl the rest). The dot on your fingertip fills with the brush color.
2. **Lift the pen**: show a peace sign (index + middle). The dot turns into a grey ring, and you can move without drawing.
3. **Clear by gesture**: hold an open palm still for 1 second. An orange countdown ring fills around your palm. Move your hand or close it to cancel.

| DRAW key | Action |
|---|---|
| `c` | Clear the canvas |
| `z` | Undo the last stroke |
| `s` | Save as a transparent PNG in the `drawings/` folder |
| `4` `5` `6` `7` `8` | Red, green, blue, yellow, white |
| `[` `]` | Smaller / bigger brush |

| Control | How |
|---|---|
| Switch mode | `1` MOUSE, `2` VOLUME, `3` DRAW (the Air Mouse window must have focus), or 1/2/3 fingers held up on your other hand |
| Calibrate / reset region | `k` / `r` in MOUSE mode (Esc cancels a calibration) |
| Clutch on/off | `x` in MOUSE mode (fist = clutch while it's on) |
| Snooze rest nudges 5 min | `n` (any mode) |
| Fatigue tracker on/off | `f` (any mode) |
| Pause / resume | `p` (the Air Mouse window must have focus) |
| Quit | `q` or Esc in the window, Ctrl+C in Terminal, or **hold a fist for 1.5 s** (a red countdown ring shows it, and opening your hand cancels). In MOUSE mode with the clutch on, use the other hand's fist |
| Emergency stop | Push the **trackpad** cursor into any screen corner |

Tips: keep your hand 40 to 80 cm from the camera, with your palm facing it. Good lighting helps tracking.

### Rest reminders (fatigue tracker)

The bar in the bottom-right corner shows how long your arm has been up. It's green before 10 min, yellow from 10 min and red from 15 min. Next to it are today's total and how many rests you've taken.

- **Reminders:** nothing before 10 min. At 10 min, 15 min, 20 min and every 5 min after that, you get a desktop notification and a banner in the window. From 20 min on, the banner stays up until you rest.
- **What counts as a rest:** lower your arm (hand out of view) for 20 s, which resets the timer. Shorter breaks don't count.
- **Snooze or turn off:** press `n` to snooze reminders for 5 min, or `f` to turn the tracker off.
- **Timing:** reminders never arrive mid-drag, mid-pinch or mid-stroke. They wait until you're done.

Sessions are logged to `fatigue_log.csv`. For a 7-day summary, run:

```bash
.venv/bin/python air_mouse.py --fatigue-report
```

**Try it quickly:** set `FATIGUE_DEBUG_SPEEDUP = 30` at the top of `air_mouse.py`. Reminders then fire at about 20 s, 30 s and 40 s, and a rest takes 3 s. Set it back to 1 afterwards.

## 4. Tuning

All settings are in the CONFIG block at the top of `air_mouse.py`. Edit a value, save, quit, and run the script again.

- Jittery cursor: lower `SMOOTH_ALPHA_SLOW`
- Laggy cursor: raise `SMOOTH_ALPHA_FAST`
- Clicks happen by accident: lower `PINCH_START`
- Pinch is hard to trigger: raise `PINCH_START`
- Can't reach the screen edges: raise the `REGION_MARGIN_*` values
- Quick clicks turn into drags: raise `DRAG_DELAY_MS`

The debug window shows a live **pinch** number. Note it with your hand open and with a light pinch, then set `PINCH_START` and `PINCH_RELEASE` between those two readings. The full tuning table is in [README.md](README.md).

## 5. If the setup breaks

If `.venv` is missing or broken, rebuild it (this needs [uv](https://docs.astral.sh/uv/)):

```bash
rm -rf .venv
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The hand model (`models/hand_landmarker.task`) downloads automatically if it's missing.
