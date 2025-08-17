#!/usr/bin/env python3
"""
Auto-align template by dragging the screen (hold-left-mouse + move) so that
the template match appears at TARGET = (1158, 774).

Important behaviour change (per user request):
 - DO NOT try to maximize the correlation value.
 - Instead aim to make the template's match LOCATION equal TARGET = (1158, 774).
 - If the match location equals TARGET and the correlation value at TARGET is >= 0.80 -> SUCCESS.
 - Movement acceptance (keep a move) is determined by whether the match location
   moves closer to TARGET (distance decreases). If distance increases, the move
   is reverted and step_limit is reduced.

Other behaviour:
 - capture area: (0,0,1920,1080)
 - start by right-clicking center of the primary FullHD screen
 - press and HOLD left mouse button (do not release) until success or abort
 - repeatedly compute cv2.matchTemplate on the capture and evaluate the location
 - press F10 to abort at any time (will release left button on abort)

Requires: opencv-python, numpy, mss (or pillow fallback), pynput
"""
import sys
import time
import threading
from pathlib import Path
import math

import cv2
import numpy as np

# fast screen capture
try:
    import mss
    _HAS_MSS = True
except Exception:
    _HAS_MSS = False
    from PIL import ImageGrab  # fallback

from pynput import keyboard, mouse
from pynput.keyboard import Key
from pynput.mouse import Button, Controller as MouseController

# ---------------- CONFIG ----------------
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PICTURES = SCRIPT_DIR / "Pictures"
TEMPLATE_NAME = "JOHANNESBURG.png"
TEMPLATE_PATH = PICTURES / TEMPLATE_NAME

# capture area (first monitor) - adjust if needed
CAP_LEFT = 0
CAP_TOP = 0
CAP_WIDTH = 1920
CAP_HEIGHT = 1080

# target location for best match (top-left)
TARGET_X = 1158
TARGET_Y = 774

# tolerances and limits
TOLERANCE = 0  # we require exact match of coordinates (distance 0 between best_loc and target)
MAX_ATTEMPTS = 200            # safety cap of main iterations
INITIAL_STEP_LIMIT = 200      # maximum move attempted at first (will be reduced)
MIN_STEP = 1                  # minimal movement attempt
SLEEP_BETWEEN_CAPTURES = 0.25 # seconds
DRAG_SUBSTEP = 8              # pixels per small move while holding mouse

# success threshold for correlation at target
SUCCESS_CORR_THRESHOLD = 0.80
# ----------------------------------------

stop_event = threading.Event()
mouse_ctrl = MouseController()

def on_press(key):
    if key == Key.f10:
        stop_event.set()
        return False

def load_template_gray():
    if not TEMPLATE_PATH.exists():
        print(f"ERROR: template not found: {TEMPLATE_PATH}", file=sys.stderr)
        sys.exit(1)
    tpl_bgr = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_COLOR)
    if tpl_bgr is None:
        print(f"ERROR: failed to read template: {TEMPLATE_PATH}", file=sys.stderr)
        sys.exit(1)
    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    return tpl_bgr, tpl_gray

def grab_screen_gray():
    if _HAS_MSS:
        with mss.mss() as sct:
            bbox = {"left": CAP_LEFT, "top": CAP_TOP, "width": CAP_WIDTH, "height": CAP_HEIGHT}
            sct_img = sct.grab(bbox)
            arr = np.array(sct_img)  # BGRA or BGR
            if arr.shape[2] == 4:
                bgr = arr[..., :3]
            else:
                bgr = arr
    else:
        pil_img = ImageGrab.grab(bbox=(CAP_LEFT, CAP_TOP, CAP_LEFT + CAP_WIDTH, CAP_TOP + CAP_HEIGHT))
        rgb = np.array(pil_img)
        bgr = rgb[..., ::-1]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray

def compute_match(gray, tpl_gray):
    """
    Compute template matching result array and also return the global maximum
    value and its location.
    Returns: res (2D array), max_val (float), max_loc (x,y)
    """
    res = cv2.matchTemplate(gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return res, max_val, max_loc

def value_at(res, x, y):
    """
    Safely read the correlation value at (x,y) from res.
    res is indexed as res[y, x]. Returns None if (x,y) out of bounds.
    """
    h, w = res.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    return float(res[y, x])

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def move_relative_while_held(dx, dy, hold_time=0.0):
    """
    Move the mouse relatively by dx,dy in small substeps.
    IMPORTANT: This function DOES NOT press or release any mouse buttons.
    Caller must ensure the left button is pressed and held if movement
    should be performed while holding the button.
    """
    steps = max(1, int(max(abs(dx), abs(dy)) / DRAG_SUBSTEP))
    sx = dx / steps
    sy = dy / steps

    try:
        for i in range(steps):
            if stop_event.is_set():
                break
            mouse_ctrl.move(int(round(sx)), int(round(sy)))
            time.sleep(0.01)
        if hold_time > 0:
            time.sleep(hold_time)
    except Exception:
        pass

def attempt_move_strategy(target, tpl_gray, current_best, current_val_at_target, step_limit):
    """
    Attempt to move toward target by at most step_limit (pixels).
    Movement will be executed by move_relative_while_held, so the caller
    should ensure the left mouse button is pressed and held before calling.

    Returns:
      moved_dx, moved_dy,
      new_best (x,y), new_max_val, new_val_at_target (or None if out-of-bounds),
      improved_flag (True if new_best is closer to target than current_best)
    """
    best_x, best_y = current_best
    tx, ty = target
    need_dx = tx - best_x
    need_dy = ty - best_y

    norm = math.hypot(need_dx, need_dy)
    if norm == 0:
        # already at the target location
        return 0, 0, current_best, None, current_val_at_target, True

    scale = 1.0
    if norm > step_limit:
        scale = step_limit / norm
    move_dx = int(round(need_dx * scale))
    move_dy = int(round(need_dy * scale))

    if move_dx == 0 and move_dy == 0:
        move_dx = int(math.copysign(MIN_STEP, need_dx)) if need_dx != 0 else 0
        move_dy = int(math.copysign(MIN_STEP, need_dy)) if need_dy != 0 else 0

    # perform movement while LEFT BUTTON IS HELD by caller
    move_relative_while_held(move_dx, move_dy)

    # capture and evaluate
    _, gray = grab_screen_gray()
    res, new_max_val, new_best = compute_match(gray, tpl_gray)
    new_val_at_target = value_at(res, tx, ty)

    # improvement criteria: match location got closer to target
    improved = (distance(new_best, target) < distance(current_best, target))
    return move_dx, move_dy, new_best, new_max_val, new_val_at_target, improved

def main():
    tpl_bgr, tpl_gray = load_template_gray()
    th, tw = tpl_gray.shape[:2]
    # ensure target fits within match result area
    max_x_allowed = CAP_WIDTH - tw
    max_y_allowed = CAP_HEIGHT - th
    if TARGET_X < 0 or TARGET_Y < 0 or TARGET_X > max_x_allowed or TARGET_Y > max_y_allowed:
        print(f"ERROR: TARGET ({TARGET_X},{TARGET_Y}) is outside allowable match locations (0..{max_x_allowed},0..{max_y_allowed}).", file=sys.stderr)
        sys.exit(1)

    # check template fits capture area
    if th > CAP_HEIGHT or tw > CAP_WIDTH:
        print("ERROR: template size is larger than capture area (1920x1080).", file=sys.stderr)
        sys.exit(1)

    # start keyboard listener for F10
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    target = (TARGET_X, TARGET_Y)
    step_limit = INITIAL_STEP_LIMIT
    attempts = 0
    left_held = False

    # initial right-click at center of primary FullHD screen, then press-and-hold left button
    center_x = CAP_LEFT + CAP_WIDTH // 2
    center_y = CAP_TOP + CAP_HEIGHT // 2
    try:
        mouse_ctrl.position = (center_x, center_y)
        mouse_ctrl.click(Button.right, 1)
        time.sleep(0.05)
        mouse_ctrl.press(Button.left)
        left_held = True
        print(f"RIGHT-CLICK at ({center_x},{center_y}) and HOLDING LEFT BUTTON", flush=True)
    except Exception as e:
        print(f"WARNING: failed to perform initial clicks: {e}", file=sys.stderr)

    # initial capture and match (after holding left)
    bgr, gray = grab_screen_gray()
    res, curr_max_val, curr_best = compute_match(gray, tpl_gray)
    curr_val_at_target = value_at(res, TARGET_X, TARGET_Y)
    print(f"INIT global_max={curr_max_val:.6f} at {curr_best}  value_at_target={None if curr_val_at_target is None else curr_val_at_target:.6f}", flush=True)
    print(f"TARGET = {target}  required_corr_threshold={SUCCESS_CORR_THRESHOLD:.2f}", flush=True)

    try:
        while not stop_event.is_set() and attempts < MAX_ATTEMPTS:
            attempts += 1

            # success condition: best match location equals target AND correlation at target >= threshold
            if curr_best == target and (curr_val_at_target is not None and curr_val_at_target >= SUCCESS_CORR_THRESHOLD):
                print(f"SUCCESS: reached target {target} with corr={curr_val_at_target:.6f} attempts={attempts}", flush=True)
                break

            # attempt move toward target with current step_limit
            moved_dx, moved_dy, new_best, new_max_val, new_val_at_target, improved = attempt_move_strategy(
                target, tpl_gray, curr_best, curr_val_at_target, step_limit
            )

            if stop_event.is_set():
                break

            if improved:
                # accept move because match location moved closer to target
                curr_best = new_best
                curr_max_val = new_max_val
                curr_val_at_target = new_val_at_target
                # optionally enlarge step_limit slowly (bounded)
                step_limit = min(INITIAL_STEP_LIMIT, int(step_limit * 1.2))
                print(f"ITER {attempts}: moved ({moved_dx},{moved_dy}) => global_max={curr_max_val:.6f} at {curr_best}  value_at_target={None if curr_val_at_target is None else curr_val_at_target:.6f}  step_limit={step_limit}", flush=True)
            else:
                # revert: move back by -moved_dx,-moved_dy while left is still held
                move_relative_while_held(-moved_dx, -moved_dy)
                # reduce step and try a smaller move next iteration
                step_limit = max(MIN_STEP, int(step_limit / 2))
                # show worsened value AND coordinates (global max location) of that correlation
                loc_str = f"({new_best[0]},{new_best[1]})" if new_best is not None else "None"
                val_target_str = "None" if new_val_at_target is None else f"{new_val_at_target:.6f}"
                print(f"ITER {attempts}: move ({moved_dx},{moved_dy}) worsened (global_max={new_max_val:.6f} at {loc_str}, value_at_target={val_target_str}), reverted. New step_limit={step_limit}", flush=True)

            # short delay before next capture (broken into small sleeps to allow abort)
            for _ in range(int(max(1, SLEEP_BETWEEN_CAPTURES / 0.05))):
                if stop_event.is_set():
                    break
                time.sleep(0.05)

        else:
            if stop_event.is_set():
                print("ABORTED by F10.", flush=True)
            else:
                print(f"FAILED to converge within {MAX_ATTEMPTS} attempts. Last global_max={curr_max_val:.6f} at {curr_best} value_at_target={None if curr_val_at_target is None else curr_val_at_target:.6f}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted by KeyboardInterrupt.", flush=True)
    finally:
        stop_event.set()
        # release left button if we are holding it
        if left_held:
            try:
                mouse_ctrl.release(Button.left)
                print("Released left mouse button.", flush=True)
            except Exception:
                pass
        try:
            listener.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
