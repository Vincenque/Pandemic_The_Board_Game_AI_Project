#!/usr/bin/env python3
"""
Check availability of UI buttons by reading pixels.

Rules:
 - RGB == (255,255,255)  -> available (True)
 - RGB == (46,46,46)     -> unavailable (False)
 - any other RGB value   -> error -> print details and exit with code 1

Requires:
    pip install mss pillow numpy pynput    # mss preferred; pillow fallback
"""

from typing import Optional, Tuple, Dict
import sys
import numpy as np

# prefer mss for fast capture
try:
    import mss
    _HAS_MSS = True
except Exception:
    _HAS_MSS = False
    try:
        from PIL import ImageGrab
    except Exception:
        ImageGrab = None

def get_pixel_rgb(x: int, y: int) -> Optional[Tuple[int, int, int]]:
    """
    Return (R, G, B) for absolute screen coordinates (x,y).
    Returns None on failure.
    """
    try:
        if _HAS_MSS:
            with mss.mss() as sct:
                bbox = {"left": x, "top": y, "width": 1, "height": 1}
                img = sct.grab(bbox)  # returns raw BGRA/BGR buffer convertible to array
                arr = np.array(img)   # shape (1,1,4) or (1,1,3)
                if arr.ndim >= 3 and arr.shape[0] >= 1 and arr.shape[1] >= 1:
                    b = int(arr[0, 0, 0])
                    g = int(arr[0, 0, 1])
                    r = int(arr[0, 0, 2])
                    return (r, g, b)
                else:
                    return None
        else:
            if ImageGrab is None:
                raise RuntimeError("Neither mss nor PIL.ImageGrab is available.")
            im = ImageGrab.grab(bbox=(x, y, x + 1, y + 1))
            px = im.getpixel((0, 0))
            if isinstance(px, tuple) and len(px) >= 3:
                return (int(px[0]), int(px[1]), int(px[2]))
            else:
                return None
    except Exception as e:
        print(f"Error reading pixel at ({x},{y}): {e}", file=sys.stderr)
        return None

def main():
    # mapping button name -> (x,y)
    buttons = {
        "Move":  (820, 1020),
        "Treat": (975, 1020),
        "Cure":  (1120, 1020),
        "Build": (1260, 1018),
        "Share": (1415, 1020),
        "Event": (1550, 1020),
    }

    # store results here
    results: Dict[str, bool] = {}

    for name, (x, y) in buttons.items():
        rgb = get_pixel_rgb(x, y)
        if rgb is None:
            print(f"[ERROR] Failed to read pixel for '{name}' at ({x},{y}). Exiting.", file=sys.stderr)
            sys.exit(1)

        # print coordinates, RGB and button name
        print(f"Checking '{name}' at ({x},{y}): RGB = {rgb}")

        if rgb == (255, 255, 255):
            results[name] = True
            print(f" -> '{name}' is AVAILABLE (True)")
        elif rgb == (46, 46, 46):
            results[name] = False
            print(f" -> '{name}' is UNAVAILABLE (False)")
        else:
            print(f"[ERROR] Unexpected color for '{name}' at ({x},{y}): RGB={rgb}. Expected (255,255,255) or (46,46,46).", file=sys.stderr)
            sys.exit(1)

    # create individual variables as requested
    Move_available  = results["Move"]
    Treat_available = results["Treat"]
    Cure_available  = results["Cure"]
    Build_available = results["Build"]
    Share_available = results["Share"]
    Event_available = results["Event"]

    # summary print
    print("\nSummary:")
    print(f" Move_available  = {Move_available}")
    print(f" Treat_available = {Treat_available}")
    print(f" Cure_available  = {Cure_available}")
    print(f" Build_available = {Build_available}")
    print(f" Share_available = {Share_available}")
    print(f" Event_available = {Event_available}")

    # If you need to use these variables later in code, return them or import this module.
    return {
        "Move_available": Move_available,
        "Treat_available": Treat_available,
        "Cure_available": Cure_available,
        "Build_available": Build_available,
        "Share_available": Share_available,
        "Event_available": Event_available,
    }

if __name__ == "__main__":
    main()
