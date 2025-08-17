#!/usr/bin/env python3
"""
When you press CTRL (left or right) this script prints:
  - the pixel coordinates under the mouse cursor (x, y)
  - the RGB color of that pixel

Press F10 to exit the program.

Requires: pillow, pynput
Optional fallbacks: pyautogui, pyscreenshot
"""

from datetime import datetime
import sys

from pynput import keyboard
from pynput.keyboard import Key
from pynput.mouse import Controller as MouseController

mouse_controller = MouseController()

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_pixel_color(x: int, y: int):
    """
    Try several methods to grab a pixel color:
      1) PIL.ImageGrab (works on Windows and macOS)
      2) pyautogui.screenshot (works on many platforms if installed)
      3) pyscreenshot.grab (Linux fallback)
    Returns an (R, G, B) or (R, G, B, A) tuple, or None on failure.
    """
    # 1) Pillow ImageGrab
    try:
        from PIL import ImageGrab
        im = ImageGrab.grab(bbox=(x, y, x + 1, y + 1))
        return im.getpixel((0, 0))
    except Exception:
        pass

    # 2) pyautogui fallback
    try:
        import pyautogui
        return pyautogui.screenshot().getpixel((x, y))
    except Exception:
        pass

    # 3) pyscreenshot fallback
    try:
        import pyscreenshot as ImageGrab2
        im = ImageGrab2.grab(bbox=(x, y, x + 1, y + 1))
        return im.getpixel((0, 0))
    except Exception:
        pass

    return None

def on_press(key):
    """
    - If CTRL pressed (left or right or generic), read mouse pos and pixel color.
    - If F10 pressed, stop the listener and exit.
    """
    try:
        # Quit on F10
        if key == Key.f10:
            print(f"[{now_str()}] F10 detected — exiting.")
            return False  # stops keyboard listener

        # CTRL keys (generic and left/right)
        if key in {Key.ctrl, Key.ctrl_l, Key.ctrl_r}:
            x, y = mouse_controller.position
            cx, cy = int(x), int(y)
            color = get_pixel_color(cx, cy)
            if color is None:
                print(f"[{now_str()}] CTRL pressed at ({cx}, {cy}) — failed to read color.")
            else:
                rgb = tuple(color[:3])  # discard alpha if present
                print(f"[{now_str()}] CTRL pressed at ({cx}, {cy}) — RGB: {rgb}")
            sys.stdout.flush()
    except Exception as e:
        # avoid crashing the listener on unexpected exceptions
        print(f"[{now_str()}] Error in on_press: {e}", file=sys.stderr)
        sys.stderr.flush()

def main():
    print("Running. Press CTRL to print cursor pixel + RGB. Press F10 to quit.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    print("Finished.")

if __name__ == "__main__":
    main()
