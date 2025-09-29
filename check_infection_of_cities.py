from typing import Tuple
from PIL import Image
import numpy as np
import pyautogui
import sys

def filter_exact_rgb(img: Image.Image, target_rgb: Tuple[int, int, int]) -> Image.Image:
    """Pixels equal to target -> white, else -> black."""
    arr = np.array(img.convert("RGB"))
    r, g, b = target_rgb
    mask = (arr[..., 0] == r) & (arr[..., 1] == g) & (arr[..., 2] == b)
    out = np.zeros_like(arr, dtype=np.uint8)
    out[mask] = 255
    return Image.fromarray(out).convert("RGB")

if __name__ == "__main__":
    LEFT, TOP = 375, 0
    WIDTH, HEIGHT = 30, 1080
    TARGET_RGB = (10, 90, 97)

    shot = pyautogui.screenshot(region=(LEFT, TOP, WIDTH, HEIGHT))
    result = filter_exact_rgb(shot, TARGET_RGB)

    # If any white pixel exists, click at (400, 1040). Otherwise exit.
    arr_result = np.array(result.convert("RGB"))
    white_mask = (arr_result[..., 0] == 255) & (arr_result[..., 1] == 255) & (arr_result[..., 2] == 255)
    if white_mask.any():
        pyautogui.click(400, 1040)
    print("Left bar is hidden")
