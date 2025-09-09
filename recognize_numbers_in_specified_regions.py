import os
import time

import pytesseract
import pyautogui
from PIL import Image, ImageOps, ImageFilter

# Top and bottom values (y measured from top of screen)
TOP = 100
BOTTOM = 123

# Regions are (left, top, width, height).
REGIONS = [
    (1275, TOP, 1307 - 1275, BOTTOM - TOP),  # region 1: x 1275..1307, y TOP..BOTTOM
    (1350, TOP, 1378 - 1350, BOTTOM - TOP),  # region 2: x 1350..1378
    (1421, TOP, 1453 - 1421, BOTTOM - TOP),  # region 3: x 1421..1453
    (1491, TOP, 1522 - 1491, BOTTOM - TOP),  # region 4: x 1491..1522
]

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
    for region in REGIONS:
        try:
            shot = pyautogui.screenshot(region=region)
        except Exception:
            results.append(None)
            continue

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
