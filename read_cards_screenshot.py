import pyautogui

# Define the capture area: (x, y, width, height)
region = (0, 0, 400, 1080)

# Take a screenshot of the defined region
screenshot = pyautogui.screenshot(region=region)

# Save the screenshot as color.png
screenshot.save("color.png")
print("Screenshot saved as color.png")

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
else:
    print(f"No pixel with value {target_color} found.")
