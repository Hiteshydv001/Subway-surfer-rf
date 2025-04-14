import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Import environment and config
# from env.subway_env import SubwayEnv
# import config

from subway_ai.env.subway_env import SubwayEnv  # Absolute import
import subway_ai.config as config

def evaluate_agent():
    """Loads and evaluates a trained agent."""
    print("----- Starting Evaluation -----")

    model_path = os.path.join(config.MODEL_DIR, config.EVAL_MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Ensure you have trained a model or specified the correct checkpoint file in config.py (EVAL_MODEL_NAME).")
        return

    # Create environment for evaluation (with rendering)
    env_lambda = lambda: SubwayEnv(render_mode="human") # Render the view
    vec_env = make_vec_env(env_lambda, n_envs=1, vec_env_cls=DummyVecEnv) # Only 1 env for eval

    # Load the trained model
    print(f"Loading model from: {model_path}")
    try:
        model = PPO.load(model_path, env=vec_env, device='auto')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the model file is compatible with the current Stable Baselines3/Torch versions.")
        vec_env.close()
        return


    total_rewards = []
    print(f"\nStarting evaluation for {config.NUM_EVAL_EPISODES} episodes...")
    print("IMPORTANT: Ensure the game window has focus!")

    for episode in range(config.NUM_EVAL_EPISODES):
        obs, _ = vec_env.reset() # Get initial observation
        done = False
        episode_reward = 0
        step = 0
        print(f"\n--- Episode {episode + 1}/{config.NUM_EVAL_EPISODES} ---")

        # Give a moment for the user to focus the window after reset
        if episode == 0: time.sleep(2)

        while not done:
            # Use deterministic=True for evaluation (agent uses best action)
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = vec_env.step(action)
            done = terminated or truncated # Episode ends if terminated or truncated

            episode_reward += reward[0] # Reward is scalar here since n_envs=1
            step += 1

            # Optional: Small delay if needed, but env step/rendering usually has some delay
            # time.sleep(0.01)

            # Check if done is an array (shouldn't be with n_envs=1, but safe check)
            if isinstance(done, (list, np.ndarray)): done = done[0]

        print(f"--- Episode Finished ---")
        print(f"  Steps: {step}")
        print(f"  Total Reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)
        time.sleep(1) # Pause briefly between episodes

    vec_env.close() # Close the environment

    print("\n----- Evaluation Finished -----")
    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average Reward over {len(total_rewards)} episodes: {avg_reward:.2f}")
    else:
        print("No episodes were completed.")

# Note: main_evaluate.py will call this function