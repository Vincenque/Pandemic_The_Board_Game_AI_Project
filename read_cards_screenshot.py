import pyautogui

# -------------------------
# Configuration
# -------------------------
region = (0, 0, 400, 1080)        # (x, y, width, height) for screenshot
target_color = (10, 90, 97)       # color to find the first row
dark_threshold = 30               # if R,G,B <= this -> consider dark -> set to (0,0,0)

# Horizontal crop to apply before saving filtered.png (columns inclusive)
crop_col_start = 87
crop_col_end = 375  # inclusive

# -------------------------
# 1) Take screenshot -> find first row with target color -> vertical crop
# -------------------------
screenshot = pyautogui.screenshot(region=region)
screenshot = screenshot.convert("RGB")

found_row = None
s_pixels = screenshot.load()
w_s, h_s = screenshot.size

for y in range(h_s):
    for x in range(w_s):
        if s_pixels[x, y] == target_color:
            found_row = y
            break
    if found_row is not None:
        break

if found_row is None:
    print(f"No pixel with value {target_color} found in the screenshot. Aborting crop/filter.")
else:
    print(f"First row with pixel {target_color}: {found_row}")

    # Crop vertically from found_row to the bottom
    cropped = screenshot.crop((0, found_row, w_s, h_s))
    cropped.save("cropped.png")
    print("Cropped image saved as cropped.png")

    # -------------------------
    # 2) Replace dark pixels (R,G,B <= dark_threshold) with pure black
    # -------------------------
    cropped = cropped.convert("RGB")
    px = cropped.load()
    w, h = cropped.size

    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if 0 <= r <= dark_threshold and 0 <= g <= dark_threshold and 0 <= b <= dark_threshold:
                px[x, y] = (0, 0, 0)

    # -------------------------
    # 3) Horizontal crop (columns crop_col_start..crop_col_end inclusive) and save filtered.png
    # -------------------------
    if crop_col_start >= w:
        # Requested start column is outside image width: save full filtered image and warn
        cropped.save("filtered.png")
        print(f"Warning: crop_col_start ({crop_col_start}) is outside image width ({w}). Saved full filtered image as filtered.png instead.")
    else:
        # Adjust end column if it exceeds width
        if crop_col_end >= w:
            print(f"Warning: crop_col_end ({crop_col_end}) exceeds image width ({w}). Adjusting to {w-1}.")
            crop_col_end_adj = w - 1
        else:
            crop_col_end_adj = crop_col_end

        # PIL crop box uses (left, upper, right, lower) with right exclusive -> use crop_col_end_adj + 1
        right = crop_col_end_adj + 1
        filtered = cropped.crop((crop_col_start, 0, right, h))
        filtered.save("filtered.png")
        print(f"Filtered image cropped horizontally to columns {crop_col_start}..{crop_col_end_adj} and saved as filtered.png (size: {filtered.size}).")
