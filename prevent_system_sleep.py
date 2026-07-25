import time
import pyautogui
 
# Prevent PyAutoGUI from throwing an exception if the mouse is moved
# to the top-left corner.
pyautogui.FAILSAFE = False
 
INTERVAL = 5 * 60  # 5 minutes
 
print("Keep-awake script started.")
print("Press Ctrl+C to stop.")
 
try:
    while True:
        time.sleep(INTERVAL)
 
        # Get current mouse position
        x, y = pyautogui.position()
 
        # Move 1 pixel right and back
        pyautogui.moveTo(x + 1, y, duration=0)
        pyautogui.moveTo(x, y, duration=0)
 
        print(f"Mouse nudged at {time.strftime('%H:%M:%S')}")
 
except KeyboardInterrupt:
    print("\nStopped.")
 