import pyautogui
import time

print("Starting in 3 seconds... take your hands off the mouse!")

# This gives you 3 seconds to get ready
time.sleep(3)

print("Moving mouse!")

# Tell the mouse to move to coordinate (100, 100)
pyautogui.moveTo(100, 100, duration=1)

# Tell the mouse to click
pyautogui.click()

print("Done!")