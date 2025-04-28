
*(Note: The `env/subway_env.py` file is inferred based on imports like `from subway_ai.env.subway_env import SubwayEnv`. Ensure this file exists and contains the Gymnasium environment definition.)*

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd hiteshydv001-subway-surfer-rf
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Depending on your OS and setup, you might need additional libraries for `mss` (screen capture) or specific versions of PyTorch compatible with `stable-baselines3`.*

## Configuration (`config.py`)

Before running any scripts, **you MUST configure `config.py`**:

1.  **`GAME_REGION` (CRITICAL):** This dictionary defines the exact pixel coordinates of your Subway Surfers game window on the screen.
    *   **How to find coordinates:**
        *   Run the `test.py` script: `python test.py`.
        *   Move your mouse cursor to the top-left corner of the game area and note the X, Y coordinates.
        *   Move your mouse cursor to the bottom-right corner of the game area and note the X, Y coordinates.
        *   Calculate `width = bottom_right_X - top_left_X`
        *   Calculate `height = bottom_right_Y - top_left_Y`
        *   Update the `GAME_REGION` dictionary in `config.py` with your `left`, `top`, `width`, and `height` values.
    *   **Example:**
        ```python
        GAME_REGION = {
            "left": 100,  # X coordinate of the top-left corner
            "top": 150,   # Y coordinate of the top-left corner
            "width": 800, # Width of the game area
            "height": 600 # Height of the game area
        }
        ```

2.  **`TEMPLATE_DIR` and `assets/`:**
    *   Ensure the `assets/` directory exists and contains `.png` images of the game elements you want to detect (e.g., `train.png`, `barrier_low.png`, `coin.png`, `game_over.png`).
    *   The filenames (without `.png`) **must** match the keys in `config.OBSTACLE_TYPES` or be `game_over` / `start_game` for the detection to work correctly.

3.  **Other Parameters:** Review other parameters like detection thresholds (`TEMPLATE_MATCH_THRESHOLD`), reward values, and PPO agent hyperparameters (`TOTAL_TIMESTEPS`, `LEARNING_RATE`, etc.) and adjust if needed.

## Usage

**Important:** For training and evaluation, the Subway Surfers game window **must be visible, unobstructed, and have focus** so that keyboard inputs are registered correctly.

1.  **Test Screen Capture & Template Matching (`screen.py`):**
    *   This script helps verify that `GAME_REGION` is set correctly and that template matching works for a specific template.
    *   Edit the `TEST_TEMPLATE_FILENAME` variable inside `screen.py` to specify which `.png` file from `assets/` you want to test.
    *   Run the script:
        ```bash
        python screen.py
        ```
    *   A window titled "Live Test" should appear, showing the captured game region. Detected instances of your chosen template should be highlighted with green boxes. Press 'q' in the window to quit.
    *   Use this to fine-tune `GAME_REGION` and check if your templates are effective.

2.  **Train the Agent (`main_train.py`):**
    *   Ensure `config.py` is correctly set up.
    *   Start Subway Surfers and bring it to the main game screen (ready to play).
    *   Run the training script:
        ```bash
        python main_train.py
        ```
    *   The script will start interacting with the game. Training progress and statistics will be printed to the console.
    *   Models will be saved periodically to the `models/` directory.
    *   Logs for TensorBoard will be saved in the `logs/` directory. You can monitor training by running:
        ```bash
        tensorboard --logdir logs/
        ```
        Then navigate to `http://localhost:6006/` in your web browser.
    *   Training can take a significant amount of time depending on `TOTAL_TIMESTEPS` and your hardware. Press `Ctrl+C` to interrupt training (it will attempt to save the current model).

3.  **Evaluate the Agent (`main_evaluate.py`):**
    *   Ensure you have a trained model saved in the `models/` directory (e.g., `ppo_subway_template_final.zip`).
    *   Update `EVAL_MODEL_NAME` in `config.py` to match the filename of the model you want to evaluate.
    *   Start Subway Surfers and bring it to the main game screen.
    *   Run the evaluation script:
        ```bash
        python main_evaluate.py
        ```
    *   The script will load the specified model and run it for `NUM_EVAL_EPISODES` (defined in `config.py`). The game will be played automatically, and the average reward over the episodes will be reported.

## How It Works (Simplified Flow)

1.  **Capture:** `screen_capture.py` grabs the pixels from the `GAME_REGION`.
2.  **Detect:** `template_matcher.py` searches the captured image for all known templates (from `assets/`).
3.  **Extract State:** `state_extractor.py` processes the detected objects, determines the closest relevant object in each lane within a "danger zone", and creates a state vector (e.g., `[clear, train, barrier_low]`).
4.  **Agent Action:** The PPO agent (loaded/trained using `agent/`) receives the state vector and chooses an action (left, right, jump, roll, nothing) based on its learned policy.
5.  **Control:** `key_controller.py` translates the agent's chosen action into a keyboard press sent to the game window.
6.  **Environment Step:** The (inferred) `SubwayEnv` advances the game by one step, captures the new screen, calculates the reward (based on survival time, coins collected, or crashing), determines if the episode is done, and extracts the next state.
7.  **Learn (During Training):** The agent uses the experience (state, action, reward, next state) to update its policy via the PPO algorithm to perform better over time.
8.  **Repeat:** The loop continues until the game ends or training completes.

## Dependencies

See `requirements.txt`. Key libraries include:

*   `numpy`: Numerical operations.
*   `opencv-python`: Image processing and template matching.
*   `mss`: Fast screen capture.
*   `Pillow`: Image library (often a dependency).
*   `gymnasium`: RL environment standard interface.
*   `stable-baselines3`: PPO implementation and RL utilities.
*   `pygame`: (Potentially used by Gymnasium rendering or environment internals).
*   `pynput`: (Potentially used as an alternative for keyboard/mouse control, though `pyautogui` seems to be used in `key_controller.py`).
*   `pyautogui`: Used for simulating keyboard presses.

## Troubleshooting / Notes

*   **Game Focus:** The game window *must* have focus for `pyautogui` keyboard presses to work.
*   **`GAME_REGION` Accuracy:** Incorrect `GAME_REGION` is the most common cause of failure. Use `screen.py` to verify.
*   **Template Quality:** The quality and distinctiveness of your `.png` templates in `assets/` significantly impact detection performance. Ensure they are cropped accurately and represent the objects well.
*   **Performance:** Template matching can be CPU-intensive. Performance depends on the size of the `GAME_REGION`, the number of templates, and your hardware.
*   **Game Updates/Resolution:** Changes in the game's graphics, UI, or running the game at a different resolution may break template matching and require new templates or `GAME_REGION` adjustments.
*   **Permissions (macOS):** On macOS, you might need to grant accessibility permissions to your terminal or IDE for `pyautogui` and `mss` to control the keyboard and capture the screen.
*   **Dataset Directory:** The `dataset/` directory seems structured for supervised learning (image classification). It might be from a previous iteration or a separate part of the project not directly used by the current RL agent implementation shown.

## Disclaimer

This project is for educational purposes to demonstrate AI and RL concepts. Using bots or automation in games may violate the terms of service. Use responsibly.
