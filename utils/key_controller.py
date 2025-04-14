import pyautogui
import time
import config # Import config to potentially use settings if needed

# Configuration
pyautogui.PAUSE = 0.05  # Small delay between actions
pyautogui.FAILSAFE = True # Move mouse to corner to stop

# Action Mapping (Matches the environment's action space, indices defined in config)
ACTION_MAP = {
    0: 'left',
    1: 'right',
    2: 'up',    # Jump
    3: 'down',  # Roll
    4: None,    # Do nothing (No-Op)
    # Add more if NUM_ACTIONS in config increases (e.g., 'space')
}
if len(ACTION_MAP) != config.NUM_ACTIONS:
     print(f"Warning: Mismatch between ACTION_MAP size ({len(ACTION_MAP)}) and config.NUM_ACTIONS ({config.NUM_ACTIONS})")

def perform_action(action_index):
    """Sends the corresponding keystroke for the action index."""
    key = ACTION_MAP.get(action_index)
    if key:
        # print(f"Action: {key}") # Debug
        pyautogui.press(key)
        time.sleep(0.05) # Optional: Small delay after action

def press_start_key():
    """Presses the key typically used to start/restart (e.g., Space)."""
    print("Pressing 'space' to attempt start/restart...")
    pyautogui.press('space')
    time.sleep(1.0) # Give game time to react

def click_location(x, y):
    """Clicks at a specific screen coordinate."""
    print(f"Clicking at ({x}, {y})")
    pyautogui.click(x, y)
    time.sleep(0.5)