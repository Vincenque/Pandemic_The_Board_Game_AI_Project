import pyautogui

# Define the capture area: (x, y, width, height)
region = (0, 0, 400, 1080)

# Take a screenshot of the defined region
screenshot = pyautogui.screenshot(region=region)

# Ensure image is in RGB mode
screenshot = screenshot.convert("RGB")

# Target color to find the first row
target_color = (10, 90, 97)

# Search for the first row containing the target color
found_row = None
pixels = screenshot.load()

for y in range(screenshot.height):
    for x in range(screenshot.width):
        if pixels[x, y] == target_color:
            found_row = y
            break
    if found_row is not None:
        break

if found_row is not None:
    print(f"First row with pixel {target_color}: {found_row}")
    # Crop the image from the found row down to the bottom
    cropped = screenshot.crop((0, found_row, screenshot.width, screenshot.height))
    cropped.save("cropped.png")
    print("Cropped image saved as cropped.png")

    # Load pixels of the cropped image and replace dark pixels with black
    cropped = cropped.convert("RGB")
    px = cropped.load()
    w, h = cropped.width, cropped.height

    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            # jeśli wszystkie składowe mieszczą się w przedziale 0..30 (włącznie)
            if 0 <= r <= 30 and 0 <= g <= 30 and 0 <= b <= 30:
                px[x, y] = (0, 0, 0)

    cropped.save("filtered.png")
    print("Filtered image saved as filtered.png")
else:
    print(f"No pixel with value {target_color} found.")
