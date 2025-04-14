import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2 # For rendering



from subway_ai.game_capture.screen_capture import capture_screen # Absolute import
from subway_ai.detection.template_matcher import load_templates, match_template
from subway_ai.detection.state_extractor import extract_state
from subway_ai.utils.key_controller import perform_action, press_start_key
# Also change the import config if it was relative
import subway_ai.config as config # Explicit absolute import

class SubwayEnv(gym.Env):
    """Custom Gym environment for Subway Surfers using template matching."""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, render_mode=None):
        super().__init__()

        # Validate game region is set (already done in config, but belt-and-suspenders)
        if config.GAME_REGION is None:
             raise ValueError("GAME_REGION must be set in config.py before creating the environment.")

        self.render_mode = render_mode

        # --- Load Templates ---
        self.all_templates = load_templates(config.TEMPLATE_DIR)
        self.object_templates = {k: v for k, v in self.all_templates.items()
                                 if k in config.OBSTACLE_TYPES and k != 'clear'}
        self.game_over_template = self.all_templates.get('game_over')
        self.start_game_template = self.all_templates.get('start_game')
        if self.game_over_template is None: print("Warning: 'game_over.png' not found.")
        if self.start_game_template is None: print("Warning: 'start_game.png' not found.")

        # --- Define action and observation space (using config) ---
        self.action_space = spaces.Discrete(config.NUM_ACTIONS)
        # Observation: Type of closest obstacle in each lane [L, M, R]
        self.observation_space = spaces.MultiDiscrete([config.NUM_OBSTACLE_TYPES] * 3)

        # --- Internal State ---
        self.current_lane = 1 # Start in middle lane (0=Left, 1=Middle, 2=Right)
        self.last_screen_raw_gray = None # Store last grayscale screen for checks/state
        self.last_screen_raw_bgr = None # Store last color screen for rendering

        # --- Rendering ---
        self.window_name = "Subway Surfers AI View"
        if self.render_mode == "human":
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # Adjust size as needed
            cv2.resizeWindow(self.window_name, config.GAME_REGION['width']//2, config.GAME_REGION['height']//2)

    def _get_observation_and_raw(self):
        """Captures screen, extracts state, and stores raw images."""
        self.last_screen_raw_gray = capture_screen(grayscale=True)
        if self.render_mode == "human": # Only capture color if needed
             self.last_screen_raw_bgr = capture_screen(grayscale=False)

        if self.last_screen_raw_gray is None:
            print("Error: Failed screen capture for observation.")
            # Return a dummy state (e.g., all clear) to avoid crashing SB3
            # Note: This might lead to incorrect behavior if capture fails repeatedly.
            return np.array([config.OBSTACLE_TYPES["clear"]] * 3, dtype=np.int32)

        return extract_state(self.last_screen_raw_gray, self.object_templates)

    def _check_template(self, template_name, threshold=config.CRITICAL_MATCH_THRESHOLD):
         """Helper to check if a specific utility template is visible."""
         template = self.all_templates.get(template_name)
         if template is None or self.last_screen_raw_gray is None:
             return False # Template not loaded or screen not captured

         matches = match_template(self.last_screen_raw_gray, template, threshold=threshold)
         return len(matches) > 0

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)

        print("\n----- Resetting Environment -----")
        print("Ensure the Poki game window has focus! Waiting 5 seconds...")
        time.sleep(5.0)

        restarted_successfully = False
        for attempt in range(5): # Retry loop
            print(f"Attempting restart (Attempt {attempt + 1}/5)...")
            # Capture fresh screen for checks
            self.last_screen_raw_gray = capture_screen(grayscale=True)
            if self.last_screen_raw_gray is None:
                print("  Screen capture failed during reset check. Retrying...")
                time.sleep(1)
                continue

            is_over = self._check_template('game_over')
            is_start = self._check_template('start_game')

            if is_over or is_start:
                print(f"  {'Game Over' if is_over else 'Start Screen'} detected. Sending start key...")
                press_start_key() # Simulate key press
                time.sleep(1.5) # Wait for game reaction

                # Check if restart worked
                self.last_screen_raw_gray = capture_screen(grayscale=True)
                if self.last_screen_raw_gray is not None and \
                   not self._check_template('game_over') and \
                   not self._check_template('start_game'):
                    print("  Restart successful!")
                    restarted_successfully = True
                    break # Exit retry loop
                else:
                    print("  Restart possibly failed (still seeing start/over screen).")
            else:
                # Neither screen detected, assume game is running or in playable state
                print("  Game seems ready or running.")
                restarted_successfully = True
                break

            time.sleep(1.0) # Wait before next retry

        if not restarted_successfully:
            print("WARNING: Could not confirm game restart after 5 attempts. Proceeding anyway.")

        # Reset internal state
        self.current_lane = 1 # Assume starting in middle lane

        observation = self._get_observation_and_raw()
        info = {"reset_successful": restarted_successfully}

        if self.render_mode == "human": self.render()

        print(f"Reset complete. Initial Observation: {observation}")
        return observation, info


    def step(self, action):
        """Performs an action, gets results."""
        # 1. Perform Action
        perform_action(action)

        # 2. Update Internal Lane State
        action_key = config.ACTION_MAP.get(action)
        if action_key == 'left':
            self.current_lane = max(0, self.current_lane - 1)
        elif action_key == 'right':
            self.current_lane = min(2, self.current_lane + 1)

        # 3. Wait for game state to update visually
        time.sleep(0.1) # Adjust if needed

        # 4. Get New Observation
        observation = self._get_observation_and_raw()
        if observation is None: # Handle case where capture failed during step
             print("Error: Failed screen capture during step. Ending episode.")
             return (np.array([config.OBSTACLE_TYPES["clear"]] * 3, dtype=np.int32),
                     config.REWARD_CRASH, True, False, {"error": "capture_fail"})

        # 5. Check for Termination (Crash)
        terminated = False
        # Check 5a: Official Game Over Screen
        if self._check_template('game_over'):
            print("  Terminated: Game Over screen detected.")
            terminated = True
        # Check 5b: Proactive State Check (Lethal obstacle in current lane)
        # Note: This simple check doesn't account for jump/roll timing.
        # A more complex state or logic would be needed for perfect proactive checks.
        elif not terminated: # Only check if not already terminated
            obstacle_in_current_lane = observation[self.current_lane]
            if obstacle_in_current_lane in config.LETHAL_OBSTACLES:
                 # Check if the action taken was potentially corrective
                 # This is a heuristic - needs refinement!
                 corrective_action_taken = False
                 if obstacle_in_current_lane == config.OBSTACLE_TYPES['barrier_low'] and action_key == 'up':
                     corrective_action_taken = True
                 elif obstacle_in_current_lane == config.OBSTACLE_TYPES['barrier_high'] and action_key == 'down':
                     corrective_action_taken = True
                 # Add train check if needed (e.g., if action was left/right)

                 if not corrective_action_taken:
                     print(f"  Terminated: Lethal obstacle ({obstacle_in_current_lane}) "
                           f"detected in current lane ({self.current_lane}) without apparent correct action.")
                     terminated = True


        # 6. Calculate Reward
        if terminated:
            reward = config.REWARD_CRASH
        else:
            reward = config.REWARD_SURVIVE # Base survival reward
            # Bonus for potential coin collection
            obstacle_in_current_lane = observation[self.current_lane]
            if obstacle_in_current_lane == config.OBSTACLE_TYPES["coin"]:
                reward += config.REWARD_COIN
            # Optional Penalty (can sometimes hinder learning)
            # if obstacle_in_current_lane != config.OBSTACLE_TYPES["clear"] and \
            #    obstacle_in_current_lane != config.OBSTACLE_TYPES["coin"]:
            #     reward += config.PENALTY_OBSTACLE_LANE


        # 7. Truncated (Usually False for this type of game)
        truncated = False

        # 8. Info Dictionary
        info = {'current_lane': self.current_lane, 'action_performed': action_key}

        # 9. Render (Optional)
        if self.render_mode == "human": self.render()

        # Debug print
        # print(f"Step: Act={action}({action_key}), Obs={observation}, Rew={reward:.2f}, Term={terminated}, Lane={self.current_lane}")

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renders the current game view using OpenCV."""
        if self.render_mode == "human":
             if self.last_screen_raw_bgr is not None:
                 display_img = self.last_screen_raw_bgr.copy() # Work on a copy
                 # Optional: Draw debug info (e.g., detected state, lane)
                 font = cv2.FONT_HERSHEY_SIMPLEX
                 obs_text = f"Obs: {self._get_observation_and_raw()}" # Re-get state for display consistency
                 lane_text = f"Lane: {self.current_lane}"
                 cv2.putText(display_img, obs_text, (10, 30), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                 cv2.putText(display_img, lane_text, (10, 60), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

                 try:
                    cv2.imshow(self.window_name, display_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'): # Allow quitting via window
                         self.close()
                         raise SystemExit("Exited via render window 'q' press")
                 except cv2.error as e:
                     print(f"Warning: OpenCV window error during render: {e}. Closing window.")
                     self.close() # Attempt cleanup

        elif self.render_mode == "rgb_array":
            # Return the last captured BGR frame (ensure it's captured even if not rendering human)
            if self.last_screen_raw_bgr is None:
                 self.last_screen_raw_bgr = capture_screen(grayscale=False) # Capture if needed
            return self.last_screen_raw_bgr

    def close(self):
        """Cleans up resources (OpenCV window)."""
        if self.render_mode == "human":
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass # Ignore error if window already closed
        print("Environment closed.")