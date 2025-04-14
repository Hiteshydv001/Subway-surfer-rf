# config.py
import os

# --- Screen Capture ---
GAME_REGION = {
    "left": 296,
    "top": 173,
    "width": 1045,
    "height": 587
}

# --- Detection ---
TEMPLATE_DIR = "assets"
TEMPLATE_MATCH_THRESHOLD = 0.75
CRITICAL_MATCH_THRESHOLD = 0.80
DANGER_ZONE_Y_START = 0.40
DANGER_ZONE_Y_END = 0.95

OBSTACLE_TYPES = {
    "clear": 0,
    "barrier_low": 1,
    "barrier_high": 2,
    "train": 3,
    "coin": 4,
}
NUM_OBSTACLE_TYPES = len(OBSTACLE_TYPES)
LETHAL_OBSTACLES = [
    OBSTACLE_TYPES["train"],
    OBSTACLE_TYPES["barrier_low"],
    OBSTACLE_TYPES["barrier_high"],
]

# --- Environment ---
NUM_ACTIONS = 5
REWARD_SURVIVE = 0.1
REWARD_COIN = 0.5
REWARD_CRASH = -10.0

# --- Agent Training (PPO Example) ---
MODEL_DIR = "models"
LOG_DIR = "logs"
MODEL_FILENAME = "ppo_subway_template"
N_ENVS = 1  # Added for single environment
TOTAL_TIMESTEPS = 250000
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
LEARNING_STARTS = 1000
SAVE_FREQ = 20000

# --- Agent Evaluation ---
EVAL_MODEL_NAME = "ppo_subway_template_final.zip"
NUM_EVAL_EPISODES = 10

# --- Ensure directories exist ---
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Validation (Basic) ---
if GAME_REGION is None:
    raise ValueError("CRITICAL: `GAME_REGION` is not set in config.py. Please define the screen coordinates.")