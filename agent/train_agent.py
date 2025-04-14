import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv # Use Dummy for GUI interaction
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch # Check if GPU is available

# Import environment and config (adjust path if needed)
# from env.subway_env import SubwayEnv
# import config

from subway_ai.env.subway_env import SubwayEnv  # Absolute import
import subway_ai.config as config

def train_agent():
    """Configures and trains the PPO agent."""
    print("----- Starting Training -----")
    print(f"PyTorch using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Create the vectorized environment
    # Lambda function ensures each environment gets the config correctly
    env_lambda = lambda: Monitor(SubwayEnv(render_mode=None)) # No rendering during training
    vec_env = make_vec_env(env_lambda, n_envs=config.N_ENVS, vec_env_cls=DummyVecEnv)

    # Callback for saving models periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.SAVE_FREQ // config.N_ENVS, 1), # Adjust freq based on n_envs
        save_path=config.MODEL_DIR,
        name_prefix=config.MODEL_FILENAME,
        save_replay_buffer=False, # Not needed for PPO
        save_vecnormalize=False   # Not using VecNormalize here
    )

    # Define the PPO model
    # Policy needs to match the observation space (MultiDiscrete -> MlpPolicy)
    model = PPO("MlpPolicy",
                vec_env,
                verbose=1,
                learning_rate=config.LEARNING_RATE,
                n_steps=config.N_STEPS,
                batch_size=config.BATCH_SIZE,
                n_epochs=config.N_EPOCHS,
                gamma=config.GAMMA,
                gae_lambda=config.GAE_LAMBDA,
                clip_range=config.CLIP_RANGE,
                ent_coef=config.ENT_COEF,
                tensorboard_log=config.LOG_DIR,
                device='auto' # Automatically use GPU if available
               )

    print("\n--- Model Architecture ---")
    print(model.policy)
    print("--------------------------\n")

    print(f"Starting training for {config.TOTAL_TIMESTEPS} timesteps...")
    print("IMPORTANT: Ensure the game window has focus during training!")

    try:
        model.learn(total_timesteps=config.TOTAL_TIMESTEPS,
                    log_interval=1, # Log training stats frequently
                    callback=checkpoint_callback,
                    tb_log_name=config.MODEL_FILENAME # Group logs under model name
                   )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Save the final model
        final_model_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_FILENAME}_final")
        model.save(final_model_path)
        print(f"\nFinal model saved to {final_model_path}.zip")
        vec_env.close() # Close the environment

    print("\n----- Training Finished -----")
    print(f"Models saved in: {config.MODEL_DIR}")
    print(f"Logs saved in: {config.LOG_DIR}")
    print("To view logs: tensorboard --logdir logs/")

# Note: main_train.py will call this function