import os

# --- Screen Capture ---
# IMPORTANT: DEFINE THIS BASED ON YOUR SCREEN SETUP!
# Use a screen measurement tool. Find the top-left (x, y) and width/height.
# Example: GAME_REGION = {'left': 100, 'top': 200, 'width': 400, 'height': 600}
GAME_REGION = {
    "left": 296,
    "top": 173,
    "width": 1045,
    "height": 587
}


# --- Detection ---
TEMPLATE_DIR = "assets"
# Threshold for template matching (0.0 to 1.0). Adjust based on template quality.
TEMPLATE_MATCH_THRESHOLD = 0.75
# Threshold for critical detections like game over/start (can be slightly higher)
CRITICAL_MATCH_THRESHOLD = 0.80
# Define the vertical "danger zone" (relative height 0.0=top, 1.0=bottom)
# Only obstacles within this vertical slice affect the state significantly.
DANGER_ZONE_Y_START = 0.40
DANGER_ZONE_Y_END = 0.95

# Map template filenames (without extension) to integer types
# IMPORTANT: Ensure these names match your files in assets/
# 'clear' is reserved for no obstacle detected.
OBSTACLE_TYPES = {
    "clear": 0,
    "barrier_low": 1,   # Obstacle needing jump
    "barrier_high": 2,  # Obstacle needing roll (if distinct)
    "train": 3,         # Obstacle needing lane change
    "coin": 4,          # Collectible (optional for state/reward)
    # Add other types if you have templates for them
}
NUM_OBSTACLE_TYPES = len(OBSTACLE_TYPES)
# Obstacles considered "lethal" if agent is in their lane without correct action
# This helps terminate episodes faster than waiting for the game over screen.
LETHAL_OBSTACLES = [
    OBSTACLE_TYPES["train"],
    OBSTACLE_TYPES["barrier_low"], # Agent needs to learn to jump/roll based on state
    OBSTACLE_TYPES["barrier_high"],
]


# --- Environment ---
# Actions: 0: Left, 1: Right, 2: Up (Jump), 3: Down (Roll), 4: No-Op
NUM_ACTIONS = 5
# Rewards
REWARD_SURVIVE = 0.1         # Base reward per step survived
REWARD_COIN = 0.5           # Bonus for being in a lane with a coin
REWARD_CRASH = -10.0        # Penalty for crashing/game over
# PENALTY_OBSTACLE_LANE = -0.1 # Optional: Small penalty for being in lane with any obstacle

# --- Agent Training (PPO Example) ---
MODEL_DIR = "models"
LOG_DIR = "logs"
MODEL_FILENAME = "ppo_subway_template" # Base name for saving models/logs

TOTAL_TIMESTEPS = 250000     # Adjust based on required training time
LEARNING_RATE = 3e-4         # 0.0003
N_STEPS = 2048               # Steps per env per update
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99                 # Discount factor
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01              # Slightly encourage exploration
LEARNING_STARTS = 1000       # Timesteps before learning starts (not applicable to PPO directly, but good concept)
SAVE_FREQ = 20000            # Save checkpoint every N steps

# --- Agent Evaluation ---
EVAL_MODEL_NAME = "ppo_subway_template_final.zip" # Or specify a checkpoint zip file
NUM_EVAL_EPISODES = 10

# --- Ensure directories exist ---
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Validation (Basic) ---
if GAME_REGION is None:
    raise ValueError("CRITICAL: `GAME_REGION` is not set in config.py. Please define the screen coordinates.")