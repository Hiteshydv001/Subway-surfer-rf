import mss
import cv2
import numpy as np
import config # Import config file

# Use the region defined in the config file
GAME_REGION = config.GAME_REGION

def capture_screen(grayscale=True):
    """
    Captures the defined GAME_REGION of the screen.

    Args:
        grayscale (bool): If True, converts the image to grayscale.

    Returns:
        numpy.ndarray: The captured screen image (Grayscale or BGR),
                       or None if capture fails.
    """
    if GAME_REGION is None:
        # This check is now mainly redundant due to the check in config.py, but good practice
        print("Error: GAME_REGION not set.")
        return None

    with mss.mss() as sct:
        try:
            # Grab the screen region directly using the dictionary
            screen = np.array(sct.grab(GAME_REGION))
            # Convert from BGRA (mss default) to BGR
            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

            if grayscale:
                # Convert to Grayscale
                screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
                return screen_gray
            else:
                return screen_bgr
        except mss.ScreenShotError as e:
            print(f"Error capturing screen: {e}")
            print(f"Check if the GAME_REGION ({GAME_REGION}) is valid and visible.")
            return None
        except Exception as e:
             print(f"An unexpected error occurred during screen capture: {e}")
             return None

# Example Usage (optional, for testing this module directly)
if __name__ == '__main__':
    print(f"Testing screen capture for region: {GAME_REGION}")
    print("Press 'q' in the display window to quit.")
    if GAME_REGION is None:
        print("Cannot run test because GAME_REGION is not set in config.py")
    else:
        while True:
            img_gray = capture_screen(grayscale=True)
            img_bgr = capture_screen(grayscale=False)

            if img_gray is not None:
                cv2.imshow("Grayscale Capture Test", img_gray)
            else:
                print("Failed to capture grayscale image.")

            if img_bgr is not None:
                 cv2.imshow("Color Capture Test", img_bgr)
            else:
                print("Failed to capture color image.")


            # Break loop if 'q' is pressed or capture failed severely
            if cv2.waitKey(100) & 0xFF == ord('q') or (img_gray is None and img_bgr is None):
                break

        cv2.destroyAllWindows()