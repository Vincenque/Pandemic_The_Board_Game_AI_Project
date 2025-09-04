import pyautogui

# Define the capture area: (x, y, width, height)
region = (0, 0, 400, 1080)

# Take a screenshot of the defined region
screenshot = pyautogui.screenshot(region=region)

# Convert to RGB mode
screenshot = screenshot.convert("RGB")

# Target color
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
else:
    print(f"No pixel with value {target_color} found.")
