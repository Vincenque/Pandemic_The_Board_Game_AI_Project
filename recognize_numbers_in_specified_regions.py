import os
import time
import pytesseract
import pyautogui
from PIL import Image, ImageOps, ImageFilter

# --- CONFIGURATION ---
SAVE_DEBUG_IMAGES = True  # Set to False to disable saving debug images to disk
DEBUG_FOLDER = "ocr_debug_steps"

# Create debug folder if enabled
if SAVE_DEBUG_IMAGES and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

# Index of rows where to look for infection cubes values
TOP_INFECTION_CUBES = 12
BOTTOM_INFECTION_CUBES = 33

# Regions are (left, top, width, height).
REGIONS = [
    (484, 20, 535 - 484, 58 - 20),  # 0: Number of cities cards left
    (743, 20, 784 - 743, 58 - 20),  # 1: Number of infected cities cards
    (907, TOP_INFECTION_CUBES, 936 - 907, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),  # 2: Infection rate
    (1118, TOP_INFECTION_CUBES, 1144 - 1118, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),  # 3: Number of outbreaks
    (1275, TOP_INFECTION_CUBES, 1307 - 1275, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),  # 4: Yellow cubes
    (1349, TOP_INFECTION_CUBES, 1378 - 1350, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),  # 5: Black cubes
    (1421, TOP_INFECTION_CUBES, 1453 - 1421, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),  # 6: Red cubes
    (1491, TOP_INFECTION_CUBES, 1522 - 1491, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),  # 7: Blue cube
]

# Preprocessing parameters
UPSCALE = 3
THRESHOLD = 160

def preprocess_image(img):
    """
    Standard preprocessing: Grayscale -> Upscale -> Sharpen -> Binary Threshold.
    """
    img = img.convert("L")
    w, h = img.size
    img = img.resize((max(1, w * UPSCALE), max(1, h * UPSCALE)), resample=Image.NEAREST)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda p: 255 if p > THRESHOLD else 0)
    return img

def try_ocr_best(pil_img, region_idx):
    """
    Try several preprocessing variants and tesseract configs.
    Returns digits string ('' if none found).
    """
    base = pil_img.convert("L")
    variants = {}
    
    # Generate image variants to improve OCR chances
    try:
        variants["orig"] = base
        # Add slight horizontal stretch to separate digits if they are too close
        variants["stretched"] = base.resize((int(base.width * 1.5), base.height), resample=Image.NEAREST)
        
        variants["inverted"] = ImageOps.invert(base)
        
        res_x4 = base.resize((base.width*4, base.height*4), resample=Image.NEAREST)
        variants["res_x4"] = res_x4
        
        # Also stretch the upscaled version
        variants["res_x4_stretched"] = res_x4.resize((int(res_x4.width * 1.2), res_x4.height), resample=Image.NEAREST)
        
        variants["res_x4_inverted"] = ImageOps.invert(res_x4)
        variants["res_x4_dilated"] = res_x4.filter(ImageFilter.MaxFilter(3))
        
        # Autocontrast usually helps
        variants["autocontrast"] = ImageOps.autocontrast(base)
        variants["autocontrast_res_x4"] = ImageOps.autocontrast(res_x4)
    except Exception:
        variants = {"orig": base}

    # Tesseract configurations to test
    configs = [
        r'-c tessedit_char_whitelist=0123456789 --psm 13', # Raw line (New!)
        r'-c tessedit_char_whitelist=0123456789 --psm 7',  # Single line
        r'-c tessedit_char_whitelist=0123456789 --psm 8',  # Single word
        r'-c tessedit_char_whitelist=0123456789 --psm 6',  # Block of text
        r'-c tessedit_char_whitelist=0123456789 --psm 10', # Single char (fallback)
    ]

    for name, vimg in variants.items():
        # Save debug images if enabled
        if SAVE_DEBUG_IMAGES:
            filename = f"region_{region_idx}_04_ocr_variant_{name}.png"
            vimg.save(os.path.join(DEBUG_FOLDER, filename))

        for cfg in configs:
            try:
                raw = pytesseract.image_to_string(vimg, config=cfg)
            except Exception:
                raw = ""
            
            # Extract only digits
            digits = ''.join(ch for ch in raw if ch.isdigit())
            if digits:
                # Basic validation: Infection cubes shouldn't be > 24 usually
                # But let's trust OCR if it sees reasonable length
                return digits
    return ""

def main():
    if SAVE_DEBUG_IMAGES:
        print(f"DEBUG MODE ON. Images saved to: {os.path.abspath(DEBUG_FOLDER)}")
    
    results = []
    
    for i, region in enumerate(REGIONS):
        try:
            shot = pyautogui.screenshot(region=region)
            if SAVE_DEBUG_IMAGES:
                shot.save(os.path.join(DEBUG_FOLDER, f"region_{i}_01_raw.png"))
        except Exception:
            results.append(None)
            continue

        shot = shot.convert("RGB")
        w, h = shot.size
        px = shot.load()
        
        # --- MANUAL THRESHOLDING LOOP ---
        # Iterates over every pixel to separate text from background
        # based on specific color rules (White, Gray, Red).
        for x in range(w):
            for y in range(h):
                r, g, b = px[x, y]
                
                # Rule 1: White text (Standard)
                is_white = (r > 200 and g > 200 and b > 200)
                
                # Rule 2: Specific Gray (Background artifact removal)
                is_specific_gray = (px[x, y] == (162, 163, 163))
                
                # Rule 3: Red text (Critical for low supply numbers)
                # High Red, Low Green, Low Blue
                is_red = (r > 130 and g < 100 and b < 100)

                if is_white or is_specific_gray or is_red:
                    px[x, y] = (255, 255, 255) # Turn text to white
                else:
                    px[x, y] = (0, 0, 0)       # Turn background to black

        if SAVE_DEBUG_IMAGES:
            shot.save(os.path.join(DEBUG_FOLDER, f"region_{i}_02_manual_threshold.png"))

        try:
            # Apply further preprocessing (Upscaling/Sharpening)
            proc = preprocess_image(shot)
            
            if SAVE_DEBUG_IMAGES:
                proc.save(os.path.join(DEBUG_FOLDER, f"region_{i}_03_preprocessed.png"))
            
            # Attempt OCR
            digits = try_ocr_best(proc, region_idx=i)
            
            if digits:
                results.append(digits)
            else:
                results.append(None)
        except Exception:
            results.append(None)

    # Prepare output: empty string for None values
    printable = [str(x) if x is not None else "" for x in results]
    
    # Print result as comma-separated string
    print(", ".join(printable))

if __name__ == "__main__":
    main()
