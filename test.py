import pyautogui

print("Move your mouse to the desired top-left and bottom-right points...")
while True:
    x, y = pyautogui.position()
    print(f"X: {x}, Y: {y}", end='\r')
