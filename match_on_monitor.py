#!/usr/bin/env python3
"""
Continuously capture the first monitor area 1920x1080 (top-left corner),
search for template ./Pictures/JOHANNESBURG.png using normalized template matching,
and print a single-line result per iteration:

    <best_val (6dp)> (x,y)

Press F10 to quit.

Requires: opencv-python, numpy, mss, pillow (fallback), pynput
"""

import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

# try mss for fast screen capture; fallback to PIL ImageGrab if mss missing
try:
    import mss
    _HAS_MSS = True
except Exception:
    _HAS_MSS = False
    from PIL import ImageGrab  # fallback


from pynput import keyboard
from pynput.keyboard import Key

# --- config ---
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PICTURES = SCRIPT_DIR / "Pictures"
TEMPLATE_NAME = "JOHANNESBURG.png"
TEMPLATE_PATH = PICTURES / TEMPLATE_NAME

# area to capture: first monitor 1920x1080 at top-left corner
CAP_LEFT = 0
CAP_TOP = 0
CAP_WIDTH = 1920
CAP_HEIGHT = 1080

SLEEP_SECONDS = 0.6  # delay between iterations

# stop event set by F10
stop_event = threading.Event()

def on_press(key):
    if key == Key.f10:
        # signal main loop to stop
        stop_event.set()
        # stop the listener
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
    """
    Capture CAP_LEFT,TOP,WIDTH,HEIGHT and return grayscale BGR image (uint8).
    Uses mss if available, otherwise PIL ImageGrab.
    """
    if _HAS_MSS:
        with mss.mss() as sct:
            bbox = {"left": CAP_LEFT, "top": CAP_TOP, "width": CAP_WIDTH, "height": CAP_HEIGHT}
            sct_img = sct.grab(bbox)
            arr = np.array(sct_img)  # shape (H, W, 4) BGRA
            if arr.shape[2] == 4:
                bgr = arr[..., :3]  # keep BGR
            else:
                bgr = arr
    else:
        # PIL returns RGB, convert to BGR for OpenCV
        pil_img = ImageGrab.grab(bbox=(CAP_LEFT, CAP_TOP, CAP_LEFT + CAP_WIDTH, CAP_TOP + CAP_HEIGHT))
        rgb = np.array(pil_img)
        bgr = rgb[..., ::-1]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray

def main_loop():
    tpl_bgr, tpl_gray = load_template_gray()
    th, tw = tpl_gray.shape[:2]

    # quick check: template must fit inside capture area
    if th > CAP_HEIGHT or tw > CAP_WIDTH:
        print("ERROR: template size is larger than capture area (1920x1080).", file=sys.stderr)
        sys.exit(1)

    # keyboard listener in separate thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while not stop_event.is_set():
            bgr, gray = grab_screen_gray()

            # compute normalized cross-correlation
            res = cv2.matchTemplate(gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            best_x, best_y = max_loc  # top-left of match in capture coordinates

            # print single line: value (6dp) and coords
            print(f"{max_val:.6f} ({best_x},{best_y})", flush=True)

            # wait a bit, but exit promptly if stop_event set
            for _ in range(int(SLEEP_SECONDS / 0.05)):
                if stop_event.is_set():
                    break
                time.sleep(0.05)
    except KeyboardInterrupt:
        # also allow Ctrl+C to stop
        pass
    finally:
        stop_event.set()
        listener.stop()

if __name__ == "__main__":
    main_loop()
