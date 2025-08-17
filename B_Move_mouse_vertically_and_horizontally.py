"""
Hold left mouse at (1000, 740), drag cursor to top, check a pixel change.
If pixel didn't change, perform horizontal alignment by holding left mouse
and dragging horizontally until the template Pictures/JOHANNESBURG.png
matches at TARGET = (1158, 774) with correlation >= SUCCESS_CORR_THRESHOLD.

Abort anytime with F10.

Requirements:
 - python3
 - opencv-python
 - numpy
 - pynput
 - mss (preferred) or Pillow as fallback
"""

import sys
import time
import threading
import math
from pathlib import Path

import cv2
import numpy as np

# fast screen capture
try:
    import mss
    _HAS_MSS = True
except Exception:
    _HAS_MSS = False
    from PIL import ImageGrab

from pynput import keyboard, mouse
from pynput.keyboard import Key
from pynput.mouse import Button, Controller as MouseController

# ---------------- CONFIG ----------------
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PICTURES = SCRIPT_DIR / "Pictures"
TEMPLATE_NAME = "JOHANNESBURG.png"
TEMPLATE_PATH = PICTURES / TEMPLATE_NAME

# fullHD assumptions - absolute screen coords
HOLD_X = 1000
HOLD_Y = 740

CHECK_X = 1600
CHECK_Y = 730

# target location for template match (top-left)
TARGET_X = 1158
TARGET_Y = 774

# dragging params
DRAG_STEP_PIXELS = 10    # pixels per sub-move when smoothing
STEP_DELAY = 0.01        # seconds between sub-moves
VERTICAL_DRAG_TO_Y = 0   # y coordinate to drag to (top of screen)

# horizontal alignment params
INITIAL_STEP_LIMIT = 200
MIN_STEP = 1
MAX_ITERATIONS = 400

# success threshold for correlation at target
SUCCESS_CORR_THRESHOLD = 0.80

# optional safety
MAX_VERTICAL_LOOPS = None  # None = unlimited (until pixel stops changing or aborted)

# assumed screen size (used for clamping large moves)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# ----------------------------------------

stop_event = threading.Event()
mouse_ctrl = MouseController()

def on_press(key):
    """Keyboard handler to set the global stop event on F10."""
    if key == Key.f10:
        stop_event.set()
        return False  # stop listener

def grab_pixel_rgb(x, y):
    """
    Return (r, g, b) tuple for screen pixel at absolute screen coords (x, y).
    Uses mss if available, otherwise PIL ImageGrab.
    Returns None on failure.
    """
    try:
        if _HAS_MSS:
            with mss.mss() as sct:
                bbox = {"left": x, "top": y, "width": 1, "height": 1}
                img = sct.grab(bbox)  # BGRA
                arr = np.array(img)   # shape (1,1,4) or (1,1,3)
                b = int(arr[0,0,0])
                g = int(arr[0,0,1])
                r = int(arr[0,0,2])
                return (r, g, b)
        else:
            im = ImageGrab.grab(bbox=(x, y, x+1, y+1))  # PIL Image in RGB
            arr = im.getpixel((0,0))  # (R, G, B)
            return (int(arr[0]), int(arr[1]), int(arr[2]))
    except Exception as e:
        print(f"Error grabbing pixel at ({x},{y}): {e}", file=sys.stderr)
        return None

def grab_screen_gray():
    """
    Capture the configured screen area and return (bgr, gray).
    bgr is a HxWx3 numpy array in BGR order (uint8).
    """
    if _HAS_MSS:
        with mss.mss() as sct:
            bbox = {"left": 0, "top": 0, "width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
            sct_img = sct.grab(bbox)
            arr = np.array(sct_img)
            if arr.shape[2] == 4:
                bgr = arr[..., :3]
            else:
                bgr = arr
    else:
        pil_img = ImageGrab.grab(bbox=(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        rgb = np.array(pil_img)
        bgr = rgb[..., ::-1]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray

def compute_match(gray, tpl_gray):
    """
    Compute template matching with TM_CCOEFF_NORMED.
    Returns: res (2D array), max_val (float), max_loc (x,y)
    max_loc is the (x,y) coordinate (top-left) of the best match in res coords.
    """
    res = cv2.matchTemplate(gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return res, float(max_val), max_loc

def value_at(res, x, y):
    """
    Safely read the correlation value at (x,y) from res.
    res is indexed as res[y, x]. Returns None if (x,y) out of bounds.
    """
    h, w = res.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    return float(res[y, x])

def move_relative_smooth(dx, dy):
    """
    Move the mouse relatively by dx,dy in small substeps for smooth motion.
    This does not press or release buttons; caller must manage button state.
    """
    steps = max(1, int(max(abs(dx), abs(dy)) / DRAG_STEP_PIXELS))
    sx = dx / steps
    sy = dy / steps
    try:
        for _ in range(steps):
            if stop_event.is_set():
                break
            mouse_ctrl.move(int(round(sx)), int(round(sy)))
            time.sleep(STEP_DELAY)
        # final small pause
        time.sleep(0.01)
    except Exception:
        pass

def press_and_hold_at(x, y):
    """Move cursor to (x,y) and press & hold left mouse button."""
    mouse_ctrl.position = (x, y)
    time.sleep(0.02)
    mouse_ctrl.press(Button.left)
    time.sleep(0.02)

def release_left():
    """Release left mouse button (best-effort)."""
    try:
        mouse_ctrl.release(Button.left)
    except Exception:
        pass

def smooth_drag_to(target_x, target_y):
    """Smooth absolute move to (target_x, target_y) using relative steps."""
    cur_x, cur_y = mouse_ctrl.position
    dx = target_x - cur_x
    dy = target_y - cur_y
    move_relative_smooth(dx, dy)
    # ensure exact final position
    mouse_ctrl.position = (target_x, target_y)
    time.sleep(0.02)

def horizontal_alignment_loop(tpl_gray):
    """
    With the left button already held by the caller at HOLD_X,HOLD_Y,
    perform horizontal moves (only X changes) until the template match
    location equals TARGET and the correlation at TARGET >= threshold.
    If initial correlation at TARGET is below threshold, attempt a large
    horizontal shift of at least half the screen width (first right, then left)
    to bring the template into view.
    Returns True on success, False if aborted or failed to converge.
    """
    th, tw = tpl_gray.shape[:2]
    # check target within allowable match locations
    res_w = SCREEN_WIDTH - tw + 1
    res_h = SCREEN_HEIGHT - th + 1
    if not (0 <= TARGET_X <= res_w - 1 and 0 <= TARGET_Y <= res_h - 1):
        print(f"ERROR: TARGET ({TARGET_X},{TARGET_Y}) is outside match result bounds (0..{res_w-1},0..{res_h-1}).", file=sys.stderr)
        return False

    step_limit = INITIAL_STEP_LIMIT
    attempts = 0

    # initial capture
    _, gray = grab_screen_gray()
    res, curr_max_val, curr_best = compute_match(gray, tpl_gray)
    curr_val_at_target = value_at(res, TARGET_X, TARGET_Y)
    print(f"[HORIZ INIT] global_max={curr_max_val:.6f} at {curr_best}  value_at_target={None if curr_val_at_target is None else curr_val_at_target:.6f}", flush=True)

    # If correlation at target is clearly below threshold, attempt a large half-screen shift
    if curr_val_at_target is None or curr_val_at_target < SUCCESS_CORR_THRESHOLD:
        half_shift = SCREEN_WIDTH // 2
        print(f"[HORIZ INIT] correlation at target {curr_val_at_target} < {SUCCESS_CORR_THRESHOLD}. Attempting large horizontal shifts of {half_shift} pixels.", flush=True)
        # try right first, then left
        tried_improvement = False
        cur_mouse_x, _ = mouse_ctrl.position

        for direction, label in ((half_shift, "right"), (-half_shift, "left")):
            if stop_event.is_set():
                break
            # compute clamped actual dx so we don't go off-screen
            desired_x = cur_mouse_x + direction
            clamped_x = max(0, min(SCREEN_WIDTH - 1, desired_x))
            actual_dx = clamped_x - cur_mouse_x
            if actual_dx == 0:
                print(f"[HORIZ LARGE] cannot move {label}: already at screen edge.", flush=True)
                continue

            print(f"[HORIZ LARGE] attempting move {label} by {actual_dx} pixels.", flush=True)
            move_relative_smooth(actual_dx, 0)

            # recapture and evaluate
            _, gray = grab_screen_gray()
            res2, new_max_val, new_best = compute_match(gray, tpl_gray)
            new_val_at_target = value_at(res2, TARGET_X, TARGET_Y)
            print(f"[HORIZ LARGE] after move {label}: global_max={new_max_val:.6f} at {new_best} value_at_target={None if new_val_at_target is None else new_val_at_target:.6f}", flush=True)

            # accept if value at target improved (or global max improved meaningfully)
            improved = False
            if new_val_at_target is not None and curr_val_at_target is not None:
                improved = new_val_at_target > curr_val_at_target + 1e-6
            else:
                # fallback: compare global max
                improved = new_max_val > curr_max_val + 1e-6

            if improved:
                print(f"[HORIZ LARGE] improvement observed after moving {label}. Accepting position.", flush=True)
                # update current bests and continue regular loop from here
                curr_best = new_best
                curr_max_val = new_max_val
                curr_val_at_target = new_val_at_target
                tried_improvement = True
                # slightly increase step limit after a big successful move
                step_limit = min(INITIAL_STEP_LIMIT, int(step_limit * 1.2))
                break
            else:
                # revert move
                print(f"[HORIZ LARGE] move {label} did not improve. Reverting.", flush=True)
                move_relative_smooth(-actual_dx, 0)
                time.sleep(0.05)

        if not tried_improvement:
            print("[HORIZ LARGE] neither large shift improved the match. Proceeding with incremental alignment.", flush=True)

    while not stop_event.is_set() and attempts < MAX_ITERATIONS:
        attempts += 1

        # success condition: best match location equals target AND corr at target >= threshold
        if curr_best == (TARGET_X, TARGET_Y) and (curr_val_at_target is not None and curr_val_at_target >= SUCCESS_CORR_THRESHOLD):
            print(f"[HORIZ SUCCESS] reached target {(TARGET_X,TARGET_Y)} with corr={curr_val_at_target:.6f} attempts={attempts}", flush=True)
            return True

        # decide horizontal move only (dx)
        need_dx = TARGET_X - curr_best[0]
        if need_dx == 0:
            # vertical mismatch only (should be rare if vertical stage finished). Try small horizontal nudge.
            move_dx = MIN_STEP
        else:
            # limit move magnitude by step_limit
            move_dx = int(math.copysign(min(abs(need_dx), step_limit), need_dx))

        # perform horizontal move while left button is held
        move_relative_smooth(move_dx, 0)

        # recapture and evaluate
        _, gray = grab_screen_gray()
        res, new_max_val, new_best = compute_match(gray, tpl_gray)
        new_val_at_target = value_at(res, TARGET_X, TARGET_Y)

        # decide if move improved (distance to target decreased)
        old_dist = abs(curr_best[0] - TARGET_X)  # horizontal distance only
        new_dist = abs(new_best[0] - TARGET_X)
        improved = (new_dist < old_dist)

        if improved:
            # accept
            curr_best = new_best
            curr_max_val = new_max_val
            curr_val_at_target = new_val_at_target
            # slightly increase step limit (bounded)
            step_limit = min(INITIAL_STEP_LIMIT, int(step_limit * 1.2))
            print(f"[HORIZ ITER {attempts}] moved_dx={move_dx} => global_max={curr_max_val:.6f} at {curr_best} value_at_target={None if curr_val_at_target is None else curr_val_at_target:.6f} step_limit={step_limit}", flush=True)
        else:
            # revert move
            move_relative_smooth(-move_dx, 0)
            # reduce step limit
            step_limit = max(MIN_STEP, int(step_limit / 2))
            print(f"[HORIZ ITER {attempts}] move_dx={move_dx} worsened (global_max={new_max_val:.6f} at {new_best}, value_at_target={None if new_val_at_target is None else new_val_at_target:.6f}), reverted. New step_limit={step_limit}", flush=True)

        # small sleep and continue
        for _ in range(int(max(1, 0.05 / 0.01))):
            if stop_event.is_set():
                break
            time.sleep(0.01)

    # did not converge
    if stop_event.is_set():
        print("[HORIZ] Aborted by user (F10).", flush=True)
    else:
        print(f"[HORIZ] Failed to converge within {MAX_ITERATIONS} iterations. Last global_max={curr_max_val:.6f} at {curr_best} value_at_target={None if curr_val_at_target is None else curr_val_at_target:.6f}", flush=True)
    return False

def main():
    # start keyboard listener for F10 (in separate thread)
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        # VERTICAL PHASE: repeat press-hold-drag-to-top until CHECK pixel unchanged
        loop_count = 0
        while not stop_event.is_set():
            if MAX_VERTICAL_LOOPS is not None and loop_count >= MAX_VERTICAL_LOOPS:
                print(f"Reached MAX_VERTICAL_LOOPS ({MAX_VERTICAL_LOOPS}). Moving on.", flush=True)
                break
            loop_count += 1
            print(f"\n--- VERTICAL ITERATION {loop_count} ---", flush=True)

            before_rgb = grab_pixel_rgb(CHECK_X, CHECK_Y)
            if before_rgb is None:
                print(f"Failed to read BEFORE pixel at ({CHECK_X},{CHECK_Y}). Exiting vertical phase.", file=sys.stderr)
                break
            print(f"BEFORE pixel ({CHECK_X},{CHECK_Y}) RGB = {before_rgb}", flush=True)

            # press & hold at HOLD_X,HOLD_Y
            press_and_hold_at(HOLD_X, HOLD_Y)
            print(f"Held left mouse at ({HOLD_X},{HOLD_Y})", flush=True)

            # drag vertically to top while holding
            smooth_drag_to(HOLD_X, VERTICAL_DRAG_TO_Y)
            print(f"Dragged cursor to ({HOLD_X},{VERTICAL_DRAG_TO_Y}) while holding", flush=True)

            # sample pixel after dragging (still holding)
            after_rgb = grab_pixel_rgb(CHECK_X, CHECK_Y)
            if after_rgb is None:
                print(f"Failed to read AFTER pixel at ({CHECK_X},{CHECK_Y}). Releasing and aborting.", file=sys.stderr)
                release_left()
                break
            print(f"AFTER pixel ({CHECK_X},{CHECK_Y}) RGB = {after_rgb}", flush=True)

            # release left mouse
            release_left()
            print("Released left mouse button", flush=True)

            if after_rgb != before_rgb:
                print("Pixel changed -> repeating vertical drag.", flush=True)
                time.sleep(0.15)
                continue
            else:
                print("Pixel did not change -> vertical stage complete.", flush=True)
                break

        # if aborted during vertical stage, stop
        if stop_event.is_set():
            print("Aborted by user (F10) during vertical stage. Exiting.", flush=True)
            return

        # HORIZONTAL ALIGNMENT PHASE
        # load template
        if not TEMPLATE_PATH.exists():
            print(f"ERROR: template not found: {TEMPLATE_PATH}", file=sys.stderr)
            return
        tpl_bgr = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_COLOR)
        if tpl_bgr is None:
            print(f"ERROR: failed to read template: {TEMPLATE_PATH}", file=sys.stderr)
            return
        tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
        th, tw = tpl_gray.shape[:2]

        # quick bounds check for target location (in match result coords)
        res_w = SCREEN_WIDTH - tw + 1
        res_h = SCREEN_HEIGHT - th + 1
        if not (0 <= TARGET_X <= res_w - 1 and 0 <= TARGET_Y <= res_h - 1):
            print(f"ERROR: TARGET ({TARGET_X},{TARGET_Y}) is outside match result bounds (0..{res_w-1},0..{res_h-1}).", file=sys.stderr)
            return

        print("\n--- STARTING HORIZONTAL ALIGNMENT PHASE ---", flush=True)
        # press & hold at HOLD_X,HOLD_Y to start horizontal dragging operations
        press_and_hold_at(HOLD_X, HOLD_Y)
        print(f"Held left mouse at ({HOLD_X},{HOLD_Y}) to begin horizontal alignment", flush=True)

        success = horizontal_alignment_loop(tpl_gray)

        # release left button at the end
        release_left()
        if success:
            print("Horizontal alignment succeeded.", flush=True)
        else:
            if stop_event.is_set():
                print("Horizontal alignment aborted by user (F10).", flush=True)
            else:
                print("Horizontal alignment did not succeed within limits.", flush=True)

    finally:
        # ensure listener stopped and left button released
        try:
            listener.stop()
        except Exception:
            pass
        try:
            release_left()
        except Exception:
            pass

if __name__ == "__main__":
    main()
