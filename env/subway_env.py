# env/subway_env.py
import gymnasium as gym
import numpy as np
import time
from subway_ai.game_capture.screen_capture import capture_screen
from subway_ai.detection.state_extractor import extract_state
from subway_ai.detection.template_matcher import load_templates, match_template
from subway_ai.utils.key_controller import perform_action, press_start_key
import subway_ai.config as config

class SubwayEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(config.NUM_ACTIONS)
        self.observation_space = gym.spaces.MultiDiscrete([config.NUM_OBSTACLE_TYPES] * 3)
        self.render_mode = render_mode
        self.templates = load_templates()
        self.last_screen_raw_gray = None
        self.episode_count = 0

    def _check_template(self, template_name, threshold=None):
        if template_name not in self.templates:
            return False
        template = self.templates[template_name]
        threshold = threshold or config.CRITICAL_MATCH_THRESHOLD
        matches = match_template(self.last_screen_raw_gray, template, threshold=threshold)
        return len(matches) > 0

    def _get_state(self):
        self.last_screen_raw_gray = capture_screen(grayscale=True)
        if self.last_screen_raw_gray is None:
            return None
        return extract_state(self.last_screen_raw_gray, self.templates)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("\n----- Resetting Environment -----")
        self.episode_count += 1
        print(f"Ensure the Poki game window has focus! Waiting 5 seconds...")
        time.sleep(5)

        max_attempts = 5
        for attempt in range(max_attempts):
            print(f"Attempting restart (Attempt {attempt + 1}/{max_attempts})...")
            press_start_key()
            time.sleep(2)
            self.last_screen_raw_gray = capture_screen(grayscale=True)
            if self.last_screen_raw_gray is None:
                continue
            is_over = self._check_template('game_over')
            is_start = self._check_template('start_game')
            if not is_over and not is_start:
                break
            time.sleep(1)
        else:
            print("Warning: Could not confirm game start after max attempts.")

        state = self._get_state()
        if state is None:
            state = np.zeros(3, dtype=np.int32)
        return state, {}

    def step(self, action):
        perform_action(action)
        time.sleep(0.1)
        state = self._get_state()
        if state is None:
            state = np.zeros(3, dtype=np.int32)
            reward = config.REWARD_CRASH
            done = True
            info = {"reason": "capture_failed"}
            return state, reward, done, False, info

        is_over = self._check_template('game_over', threshold=config.CRITICAL_MATCH_THRESHOLD)
        reward = config.REWARD_SURVIVE
        done = is_over
        truncated = False
        info = {}

        if is_over:
            reward = config.REWARD_CRASH
            info["reason"] = "game_over"
        else:
            for i, obstacle_type in enumerate(state):
                if obstacle_type == config.OBSTACLE_TYPES["coin"]:
                    reward += config.REWARD_COIN
                elif obstacle_type in config.LETHAL_OBSTACLES:
                    reward = config.REWARD_CRASH
                    done = True
                    info["reason"] = "lethal_obstacle"
                    break

        return state, reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        print("Environment closed.")