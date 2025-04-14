#!/usr/bin/env python
# screen.py - Test script for screen capture, template loading, and matching.

import cv2
import time
import numpy as np # cv2 often uses numpy implicitly

# --- IMPORTANT ---
# This script assumes it's located in the root 'subway_ai' directory
# and that the necessary subdirectories (game_capture, detection, assets)
# and the config.py file exist relative to it.
# ---

try:
    from game_capture.screen_capture import capture_screen
    from detection.template_matcher import load_templates, match_template
    import config # Import global config
except ImportError as e:
    print("\nERROR: Could not import necessary modules.")
    print(f" Details: {e}")
    print(" Please ensure:")
    print("  1. You are running this script from the main 'subway_ai' directory.")
    print("  2. The directories 'game_capture', 'detection', 'assets' exist.")
    print("  3. The file 'config.py' exists.")
    print("  4. All requirements from 'requirements.txt' are installed.")
    exit()
except Exception as e:
    print(f"\nAn unexpected error occurred during imports: {e}")
    exit()


# --- Configuration for this Test ---
# V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V
# >> EDIT THIS LINE to choose which template from your assets/ folder to test <<
TEST_TEMPLATE_FILENAME = "train.png" # Example: test matching for train.png
# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

SHOW_TEMPLATE_WINDOW = True # Set to False if you don't want the small template preview

def run_test():
    """Runs the screen capture and template matching test."""
    print("\n--- Starting Screen & Template Test ---")

    # 1. Check GAME_REGION Configuration
    if config.GAME_REGION is None:
        print("\nERROR: GAME_REGION is not set in config.py. Please define it first.")
        return # Exit the function
    else:
        print(f"Using GAME_REGION from config.py: {config.GAME_REGION}")

    # 2. Load All Templates
    try:
        print(f"\nLoading templates from: {config.TEMPLATE_DIR}")
        templates = load_templates()
        if not templates:
            print("ERROR: No valid templates were loaded. Check assets folder and config.py.")
            return
        print(f" Successfully loaded {len(templates)} templates: {list(templates.keys())}")
    except FileNotFoundError:
        print(f"ERROR: Template directory '{config.TEMPLATE_DIR}' not found.")
        return
    except Exception as e:
        print(f"ERROR loading templates: {e}")
        return

    # 3. Validate and Select the Test Template
    test_template_name = TEST_TEMPLATE_FILENAME.replace(".png", "") # Get name without extension
    if test_template_name not in templates:
        print(f"\nERROR: The specified test template '{TEST_TEMPLATE_FILENAME}' "
              f"(name: '{test_template_name}') was not found among the loaded templates.")
        print("Please check the filename in the 'assets' folder and the 'TEST_TEMPLATE_FILENAME' variable in this script.")
        return
    else:
        test_template_img = templates[test_template_name]
        print(f"\nSelected template for testing: '{test_template_name}'")
        if SHOW_TEMPLATE_WINDOW:
            try:
                cv2.imshow(f"Template: {test_template_name}", test_template_img)
            except cv2.error as e:
                 print(f"Warning: Could not display template preview window: {e}")


    # 4. Start Live Test Loop
    print("\nStarting live capture and matching...")
    print(f" -> Look for GREEN boxes around '{test_template_name}' in the 'Live Test' window.")
    print(" -> Ensure the game object appears within the defined GAME_REGION.")
    print(" -> Press 'q' in the 'Live Test' window to quit.")
    print("\n--- Make sure the game window is visible and positioned correctly! ---")
    time.sleep(3) # Give user time to prepare

    main_window_name = "Live Test - Screen Capture & Matching"
    cv2.namedWindow(main_window_name, cv2.WINDOW_NORMAL) # Make it resizable

    while True:
        # Capture screen (grayscale needed for matching)
        screen_gray = capture_screen(grayscale=True)
        if screen_gray is None:
            print("Capture Failed (Grayscale). Check region/visibility.")
            time.sleep(0.5)
            continue # Skip this frame

        # --- Perform Matching ---
        start_time = time.time()
        matches = match_template(screen_gray, test_template_img, threshold=config.TEMPLATE_MATCH_THRESHOLD)
        match_time = time.time() - start_time

        # --- Visualization ---
        # Capture color screen for display
        screen_bgr = capture_screen(grayscale=False)
        if screen_bgr is None:
            print("Capture Failed (Color). Displaying grayscale.")
            # If color fails, convert the grayscale we have
            screen_bgr = cv2.cvtColor(screen_gray, cv2.COLOR_GRAY2BGR)

        # Draw results
        match_count = 0
        if matches:
            match_count = len(matches)
            for (x, y), w, h, confidence in matches:
                 # Draw rectangle (Green) around match
                 cv2.rectangle(screen_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                 # Put confidence score near the box
                 text = f"{confidence:.2f}"
                 cv2.putText(screen_bgr, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Display info text
        info_text_matches = f"Matches ('{test_template_name}'): {match_count}"
        info_text_time = f"Match Time: {match_time:.4f}s"
        info_text_quit = "Press 'q' to quit"
        cv2.putText(screen_bgr, info_text_matches, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(screen_bgr, info_text_time, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(screen_bgr, info_text_quit, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        # Show the frame
        try:
            cv2.imshow(main_window_name, screen_bgr)
        except cv2.error as e:
             print(f"Error displaying frame: {e}. Quitting test.")
             break # Exit loop on display error

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n'q' pressed. Exiting test loop.")
            break

    # Cleanup
    print("\n--- Test Finished ---")
    cv2.destroyAllWindows()
    print("Closed OpenCV windows.")

# --- Main execution block ---
if __name__ == "__main__":
    run_test()