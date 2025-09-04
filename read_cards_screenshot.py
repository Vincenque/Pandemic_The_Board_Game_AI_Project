import pyautogui

# Define the capture area: (x, y, width, height)
region = (0, 0, 400, 1080)

# Take a screenshot of the defined region
screenshot = pyautogui.screenshot(region=region)

# Save the screenshot as color.png
screenshot.save("color.png")

print("Screenshot saved as color.png")
