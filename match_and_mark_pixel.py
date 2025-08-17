#!/usr/bin/env python3
"""
Load two images from ./Pictures:
  - Whole_board.png          (large image)
  - JOHANNESBURG.png   (template, smaller)

Compute normalized template matching (TM_CCOEFF_NORMED),
show correlation heatmap with the best-correlation PIXEL marked in red,
and show the large image with a yellow rectangle for the matched template region.

This version DOES NOT SAVE PNG files — it only displays the figures.

Requires: opencv-python, numpy, matplotlib
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- paths: Pictures folder is next to this script ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PICTURES = SCRIPT_DIR / "Pictures"
LARGE_NAME = "Whole_board.png"
SMALL_NAME = "JOHANNESBURG.png"

LARGE_PATH = PICTURES / LARGE_NAME
SMALL_PATH = PICTURES / SMALL_NAME

def exit_with(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

def load_images():
    if not PICTURES.exists() or not PICTURES.is_dir():
        exit_with(f"Pictures folder not found at: {PICTURES}")
    if not LARGE_PATH.exists():
        exit_with(f"Large image not found: {LARGE_PATH}")
    if not SMALL_PATH.exists():
        exit_with(f"Small image (template) not found: {SMALL_PATH}")

    large_bgr = cv2.imread(str(LARGE_PATH), cv2.IMREAD_COLOR)
    small_bgr = cv2.imread(str(SMALL_PATH), cv2.IMREAD_COLOR)

    if large_bgr is None:
        exit_with(f"Failed to read large image: {LARGE_PATH}")
    if small_bgr is None:
        exit_with(f"Failed to read small image: {SMALL_PATH}")

    return large_bgr, small_bgr

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def compute_correlation(large_gray, small_gray):
    """
    Use OpenCV matchTemplate with TM_CCOEFF_NORMED
    Returns result (2D float32 map), best_val (float), best_loc (x,y)
    """
    result = cv2.matchTemplate(large_gray, small_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return result, max_val, max_loc

def plot_and_show(large_bgr, small_bgr, corr_map, best_val, best_loc):
    # convert BGR->RGB for matplotlib
    large_rgb = cv2.cvtColor(large_bgr, cv2.COLOR_BGR2RGB)
    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

    h_t, w_t = small_bgr.shape[:2]
    map_h, map_w = corr_map.shape  # rows (y), cols (x)

    # --- Plot correlation heatmap and mark best pixel in RED ---
    plt.figure(figsize=(9, 6))
    extent = [0, map_w, 0, map_h]
    im = plt.imshow(corr_map, origin='lower', aspect='auto', extent=extent)
    plt.colorbar(im, label='Normalized correlation (TM_CCOEFF_NORMED)')
    plt.title(f"Correlation map (best score = {best_val:.4f})")
    plt.xlabel("x (template top-left)")
    plt.ylabel("y (template top-left)")

    best_x, best_y = best_loc

    # place marker at center of that heatmap "pixel" for visibility
    plt.scatter([best_x + 0.5], [best_y + 0.5],
                s=120, marker='s', facecolors='red', edgecolors='black', linewidths=0.8,
                label='best correlation (top-left)')
    plt.legend(loc='lower right')

    # annotate value
    plt.annotate(f"{best_val:.4f}\n({best_x}, {best_y})",
                 xy=(best_x + 0.5, best_y + 0.5),
                 xytext=(10, 10), textcoords='offset points',
                 color='white', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))

    plt.tight_layout()

    # --- Show matched location on the large image (yellow rectangle only) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(large_rgb)
    rect = plt.Rectangle((best_x, best_y), w_t, h_t,
                         edgecolor='yellow', linewidth=2, facecolor='none')
    ax.add_patch(rect)
    ax.text(best_x, max(best_y - 10, 0),
            f"{best_val:.4f} @ top-left {best_loc}",
            color='yellow', fontsize=10, backgroundcolor='black', alpha=0.6)
    ax.set_title("Best match (yellow rectangle = template region)")
    ax.axis('off')
    plt.tight_layout()

    # show template also
    plt.figure(figsize=(4,4))
    plt.imshow(small_rgb)
    plt.title("Template (JOHANNESBURG.png)")
    plt.axis('off')

    # finally display all figures (blocking)
    plt.show()

def main():
    print("Script directory:", SCRIPT_DIR)
    print("Loading images from:", PICTURES)
    large_bgr, small_bgr = load_images()

    if small_bgr.shape[0] > large_bgr.shape[0] or small_bgr.shape[1] > large_bgr.shape[1]:
        exit_with("Template is larger than the large image. Cannot do matching.")

    large_gray = to_gray(large_bgr)
    small_gray = to_gray(small_bgr)

    corr_map, best_val, best_loc = compute_correlation(large_gray, small_gray)
    print(f"Best normalized correlation value: {best_val:.4f}")
    print(f"Best match top-left coordinates (x, y): {best_loc}")

    plot_and_show(large_bgr, small_bgr, corr_map, best_val, best_loc)
    print("Done.")

if __name__ == "__main__":
    main()
