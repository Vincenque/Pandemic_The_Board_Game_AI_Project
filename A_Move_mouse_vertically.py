"""
Hold left mouse at (1000, 740), drag cursor to top of screen, check pixel change.

Requirements:
 - python3
 - pynput
 - mss (preferred) or Pillow as fallback

Behavior:
 - Repeat pressing-and-holding at (1000, 740) and dragging to (1000, 0)
   until the RGB value of pixel (1600, 730) remains the same after a drag.
 - Prints before/after RGB and coordinates each iteration.
 - Gracefully releases the mouse if interrupted.
"""

import time
import sys
from pynput.mouse import Controller as MouseController, Button
try:
    import mss
    import numpy as np
    _HAS_MSS = True
except Exception:
    _HAS_MSS = False
    from PIL import ImageGrab

# CONFIG (fullHD assumptions)
HOLD_X = 1000
HOLD_Y = 740

CHECK_X = 1600
CHECK_Y = 730

# Drag target (same X, top of screen)
DRAG_TO_X = HOLD_X
DRAG_TO_Y = 0

# movement parameters for smooth dragging
DRAG_STEP_PIXELS = 10    # pixel step per sub-move
STEP_DELAY = 0.01        # seconds between sub-moves

# optional safety: maximum number of loops (None for unlimited)
MAX_LOOPS = None  # set to an int to limit repeats, e.g. 100

mouse = MouseController()

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
                img = sct.grab(bbox)  # returns BGRA
                arr = np.array(img)   # shape (1,1,4) or (1,1,3)
                # handle BGRA or BGR
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

def smooth_drag_to(target_x, target_y):
    """
    Smoothly move cursor from current position to (target_x, target_y)
    using small relative moves. Does not press/release mouse buttons.
    """
    cur_x, cur_y = mouse.position
    dx = target_x - cur_x
    dy = target_y - cur_y
    dist = max(1, int(max(abs(dx), abs(dy))))
    steps = max(1, dist // DRAG_STEP_PIXELS)
    step_dx = dx / steps
    step_dy = dy / steps
    for i in range(steps):
        mouse.move(int(round(step_dx)), int(round(step_dy)))
        time.sleep(STEP_DELAY)
    # final adjust to exact coordinates
    mouse.position = (target_x, target_y)
    # tiny pause to let the system settle
    time.sleep(0.02)

def press_and_hold_at(x, y):
    """Move cursor to (x,y) and press & hold left mouse button."""
    mouse.position = (x, y)
    time.sleep(0.02)
    mouse.press(Button.left)
    # slight pause to ensure OS registers press
    time.sleep(0.02)

def release_left():
    """Release left mouse button if currently pressed (best-effort)."""
    try:
        mouse.release(Button.left)
    except Exception:
        # pynput may raise if not pressed; ignore
        pass

def main():
    loop_count = 0
    try:
        while True:
            if MAX_LOOPS is not None and loop_count >= MAX_LOOPS:
                print(f"Reached MAX_LOOPS ({MAX_LOOPS}). Exiting.")
                break
            loop_count += 1
            print(f"\n=== ITERATION {loop_count} ===")

            # sample pixel before pressing
            before_rgb = grab_pixel_rgb(CHECK_X, CHECK_Y)
            if before_rgb is None:
                print(f"Failed to read BEFORE pixel at ({CHECK_X},{CHECK_Y}). Exiting.", file=sys.stderr)
                break
            print(f"BEFORE: pixel ({CHECK_X},{CHECK_Y}) RGB = {before_rgb}")

            # press & hold at HOLD_X,HOLD_Y
            press_and_hold_at(HOLD_X, HOLD_Y)
            print(f"Held left mouse at ({HOLD_X},{HOLD_Y})")

            # drag to top (DRAG_TO_X, DRAG_TO_Y) while holding
            smooth_drag_to(DRAG_TO_X, DRAG_TO_Y)
            print(f"Dragged cursor to ({DRAG_TO_X},{DRAG_TO_Y}) while holding left")

            # sample pixel after dragging (still holding)
            after_rgb = grab_pixel_rgb(CHECK_X, CHECK_Y)
            if after_rgb is None:
                print(f"Failed to read AFTER pixel at ({CHECK_X},{CHECK_Y}). Releasing and exiting.", file=sys.stderr)
                release_left()
                break
            print(f"AFTER:  pixel ({CHECK_X},{CHECK_Y}) RGB = {after_rgb}")

            # release the left mouse button (end of this attempt)
            release_left()
            print("Released left mouse button")

            # compare
            if after_rgb != before_rgb:
                print("Pixel changed -> repeating the whole procedure.")
                # small pause before next iteration
                time.sleep(0.15)
                continue
            else:
                print("Pixel did not change -> finishing.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user (KeyboardInterrupt). Releasing mouse and exiting.")
        release_left()
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        release_left()

if __name__ == "__main__":
    main()
