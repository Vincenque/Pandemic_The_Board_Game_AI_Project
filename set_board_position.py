import os
import time
from PIL import Image, ImageGrab
import numpy as np
import matplotlib.pyplot as plt
import pyautogui

# Move the mouse cursor to absolute screen coordinates (x, y).
def move_cursor(x, y):
    pyautogui.FAILSAFE = False
    pyautogui.moveTo(int(x), int(y))


# Perform a mouse drag by dx, dy pixels while holding the left mouse button.
# Assumes the cursor is already at the desired start position.
def drag_by(dx, dy=0, duration=0.2):
    pyautogui.FAILSAFE = False
    pyautogui.mouseDown(button='left')
    pyautogui.moveRel(int(dx), int(dy), duration=duration)
    pyautogui.mouseUp(button='left')


# Return directory where the script is located, or current working dir as fallback.
def get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
    except NameError:
        # __file__ may be undefined in some interactive environments; fall back to cwd
        return os.getcwd()


# Load image from disk and convert to RGB mode.
def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


# Given a PIL RGB image, set pixels to white only if R>240 and G>240 and B>240.
# All other pixels become black. Return a new PIL RGB image.
def filter_image(img):
    arr = np.array(img, dtype=np.uint8)
    mask = (arr[..., 0] > 240) & (arr[..., 1] > 240) & (arr[..., 2] > 240)
    out = np.zeros_like(arr, dtype=np.uint8)
    out[mask] = [255, 255, 255]
    return Image.fromarray(out).convert("RGB")


# Capture a horizontal strip of the primary display between y_top (inclusive) and y_bottom (exclusive),
# spanning the full width (default 1920). Coordinates use origin (0,0) at top-left of the screen.
def capture_horizontal_strip(y_top=775, y_bottom=795, width=1920):
    left = 0
    top = int(y_top)
    right = int(width)
    bottom = int(y_bottom)
    if top < 0 or bottom <= top or right <= 0:
        raise ValueError("Invalid capture strip coordinates.")
    try:
        img = ImageGrab.grab(bbox=(left, top, right, bottom))
        return img.convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Unable to capture screen strip with PIL.ImageGrab: {e}")


# Compute 2D convolution using FFT and return the real part.
def _fft_convolve2d(a, b):
    s0 = a.shape[0] + b.shape[0] - 1
    s1 = a.shape[1] + b.shape[1] - 1
    fa = np.fft.fft2(a, (s0, s1))
    fb = np.fft.fft2(b, (s0, s1))
    conv = np.fft.ifft2(fa * fb).real
    return conv


# Compute normalized cross-correlation between a template image and an image.
# Returns: correlation array, template shape (h,w), image shape (h,w).
def compute_normalized_cross_correlation(template_img, image_img, eps=1e-10):
    tpl = np.array(template_img.convert("L"), dtype=np.float64) / 255.0
    img = np.array(image_img.convert("L"), dtype=np.float64) / 255.0

    if tpl.shape[0] > img.shape[0] or tpl.shape[1] > img.shape[1]:
        tpl_pil = Image.fromarray((tpl * 255).astype(np.uint8))
        tpl_pil = tpl_pil.resize((img.shape[1], img.shape[0]), Image.LANCZOS)
        tpl = np.array(tpl_pil, dtype=np.float64) / 255.0

    th, tw = tpl.shape
    ih, iw = img.shape

    out_h = ih - th + 1
    out_w = iw - tw + 1
    if out_h <= 0 or out_w <= 0:
        return np.zeros((0, 0), dtype=np.float64), (th, tw), (ih, iw)

    N = th * tw
    tpl_mean = tpl.mean()
    tpl_zero = tpl - tpl_mean
    tpl_ssd = np.sum(tpl_zero * tpl_zero)
    if tpl_ssd < eps:
        return np.zeros((out_h, out_w), dtype=np.float64), (th, tw), (ih, iw)

    tpl_flip = tpl_zero[::-1, ::-1]
    num_full = _fft_convolve2d(img, tpl_flip)
    start_r = th - 1
    start_c = tw - 1
    num = num_full[start_r:start_r + out_h, start_c:start_c + out_w]

    ones = np.ones((th, tw), dtype=np.float64)
    sum_I_full = _fft_convolve2d(img, ones)
    sum_I2_full = _fft_convolve2d(img * img, ones)
    sum_I = sum_I_full[start_r:start_r + out_h, start_c:start_c + out_w]
    sum_I2 = sum_I2_full[start_r:start_r + out_h, start_c:start_c + out_w]

    var_I = sum_I2 - (sum_I * sum_I) / N
    var_I = np.maximum(var_I, 0.0)
    denom = np.sqrt(var_I * tpl_ssd)
    corr = num / (denom + eps)
    return corr, (th, tw), (ih, iw)


# Right-click the center of a screen (screen_w x screen_h), scroll the mouse wheel down for approximately scroll_time seconds,
# then hold left mouse button at the center and drag the cursor to the top (y=0).
def center_rightclick_scroll_and_drag_up(screen_w=1920, screen_h=1080, scroll_time=0.5, scroll_step=-1000, step_delay=0.01, drag_duration=0.25):
    import time
    import pyautogui

    pyautogui.FAILSAFE = False

    cx = int(screen_w // 2)
    cy = int(screen_h // 2)

    pyautogui.moveTo(cx, cy)
    pyautogui.click(button='right')

    end = time.perf_counter() + float(scroll_time)
    while time.perf_counter() < end:
        pyautogui.scroll(scroll_step)
        time.sleep(step_delay)

    pyautogui.moveTo(cx, cy)
    pyautogui.mouseDown(button='left')
    pyautogui.moveTo(cx, 0, duration=drag_duration)
    pyautogui.mouseUp(button='left')

def main():
    script_dir = get_script_dir()
    pictures_dir = os.path.join(script_dir, "Pictures")
    os.makedirs(pictures_dir, exist_ok=True)

    # Load the already filtered reference image instead of the unfiltered original.
    input_name = "JOHANNESBURG_filtered.png"
    input_path = os.path.join(pictures_dir, input_name)
    reference_img = load_image(input_path)

    # Set screen ready
    center_rightclick_scroll_and_drag_up()

    # Capture only the horizontal strip spanning full width=1920.
    screenshot_strip = capture_horizontal_strip(y_top=775, y_bottom=795, width=1920)

    # Filter strip
    filtered_strip = filter_image(screenshot_strip)

    # Compute initial correlation and best-match metrics before any drag.
    corr, tpl_shape, img_shape = compute_normalized_cross_correlation(reference_img, filtered_strip)

    # Compute numeric best-match info
    best_idx = np.unravel_index(np.argmax(corr), corr.shape)
    best_row, best_col = int(best_idx[0]), int(best_idx[1])
    best_value = float(corr[best_row, best_col])

    th, tw = tpl_shape
    ih, iw = img_shape

    match_center_x = best_col + tw / 2.0
    match_center_y = best_row + th / 2.0
    image_center_x = iw / 2.0
    image_center_y = ih / 2.0

    dx = match_center_x - image_center_x
    dy = match_center_y - image_center_y

    # Print initial correlation and offsets.
    print(f"Best correlation value: {best_value:.6f}")
    print(f"Best match top-left position in strip (x, y): ({best_col}, {best_row})")
    print(f"Match center (x, y) in strip coords: ({match_center_x:.1f}, {match_center_y:.1f})")
    print(f"Strip center (x, y): ({image_center_x:.1f}, {image_center_y:.1f})")
    print(f"Center offset (dx, dy) in pixels: ({dx:.1f}, {dy:.1f})")

    # Example start cursor position (x, y) where user wants to place cursor before holding LMB.
    start_cursor_x = 1000
    start_cursor_y = 100

    move_cursor(start_cursor_x, start_cursor_y)

    # Desired absolute horizontal pixel where the TEMPLATE CENTER should end up.
    desired_center_x = 1030

    # Current match center in strip coordinates (pixels from left of strip)
    current_center_x = match_center_x

    # Compute needed horizontal move (positive -> move right)
    move_needed = desired_center_x - current_center_x
    dx_pixels = int(round(move_needed))

    if dx_pixels != 0:
        print(f"Performing mouse drag by dx = {dx_pixels} pixels (horizontal) to place template center at x={desired_center_x} from start ({start_cursor_x},{start_cursor_y}).")
        drag_by(dx_pixels, 0, duration=0.25)
    else:
        print(f"Template already at desired horizontal center x={desired_center_x}; no drag performed.")

if __name__ == "__main__":
    main()
