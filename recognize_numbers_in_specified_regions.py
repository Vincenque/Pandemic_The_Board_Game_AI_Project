import os
import time

import pytesseract
import pyautogui
from PIL import Image, ImageOps, ImageFilter

# Index of rows where to look for infection cubes values
TOP_INFECTION_CUBES = 8
BOTTOM_INFECTION_CUBES = 33

# Regions are (left, top, width, height).
REGIONS = [
    (1275, TOP_INFECTION_CUBES, 1307 - 1275, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),  # region 1: Yellow cubes
    (1350, TOP_INFECTION_CUBES, 1378 - 1350, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),  # region 2: Black cubes
    (1421, TOP_INFECTION_CUBES, 1453 - 1421, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),  # region 3: Red cubes
    (1491, TOP_INFECTION_CUBES, 1522 - 1491, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),  # region 4: Blue cube
    (750, 20, 776 - 750, 58 - 20),  # region 5: Cards of infected cities (special filtering)
]

# index of the special region (last one)
INFECTED_CITIES_CARDS_INDEX = len(REGIONS) - 1

# Preprocessing parameters
UPSCALE = 3
THRESHOLD = 160

def preprocess_image(img):
    img = img.convert("L")
    w, h = img.size
    img = img.resize((max(1, w * UPSCALE), max(1, h * UPSCALE)), resample=Image.NEAREST)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda p: 255 if p > THRESHOLD else 0)
    return img

def try_ocr_best(pil_img):
    """
    Try several preprocessing variants and tesseract configs.
    Returns digits string ('' if none found).
    """
    base = pil_img.convert("L")
    variants = {}
    try:
        variants["orig"] = base
        variants["inverted"] = ImageOps.invert(base)
        variants["res_x4"] = base.resize((base.width*4, base.height*4), resample=Image.NEAREST)
        variants["res_x4_inverted"] = ImageOps.invert(variants["res_x4"])
        variants["res_x4_dilated"] = variants["res_x4"].filter(ImageFilter.MaxFilter(3))
        variants["res_x4_eroded"] = variants["res_x4"].filter(ImageFilter.MinFilter(3))
        variants["autocontrast"] = ImageOps.autocontrast(base)
        variants["autocontrast_res_x4"] = ImageOps.autocontrast(variants["res_x4"])
    except Exception:
        variants = {"orig": base}

    configs = [
        r'-c tessedit_char_whitelist=0123456789 --psm 10',  # single char
        r'-c tessedit_char_whitelist=0123456789 --psm 7',   # single line
        r'-c tessedit_char_whitelist=0123456789 --psm 8',
        r'-c tessedit_char_whitelist=0123456789 --psm 6',
    ]

    for vimg in variants.values():
        for cfg in configs:
            try:
                raw = pytesseract.image_to_string(vimg, config=cfg)
            except Exception:
                raw = ""
            digits = ''.join(ch for ch in raw if ch.isdigit())
            if digits:
                return digits
    return ""

def main():
    results = []
    for i, region in enumerate(REGIONS):
        try:
            shot = pyautogui.screenshot(region=region)
        except Exception:
            results.append(None)
            continue

        # If this is infected cities cards region, then convert all non-white pixels to black.
        if i == INFECTED_CITIES_CARDS_INDEX:
            try:
                shot = shot.convert("RGB")
                w, h = shot.size
                px = shot.load()
                for x in range(w):
                    for y in range(h):
                        if px[x, y] != (255, 255, 255):
                            px[x, y] = (0, 0, 0)
            except Exception:
                # if special filtering fails, continue with the original shot
                pass

        try:
            proc = preprocess_image(shot)
            digits = try_ocr_best(proc)
            if digits:
                # keep as integer-like string (allow "9", "12", etc.)
                results.append(digits)
            else:
                results.append(None)
        except Exception:
            results.append(None)

    # prepare output: empty string for None
    printable = [str(x) if x is not None else "" for x in results]
    # Print only a single line with values separated by commas and a space (no extra text)
    print(", ".join(printable))

if __name__ == "__main__":
    main()
