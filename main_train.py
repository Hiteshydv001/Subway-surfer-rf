#!/usr/bin/env python
# Entry point for training the agent

# Ensure the project root is potentially discoverable if run from elsewhere
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Optional if needed

from subway_ai.agent.train_agent import train_agent  # Changed to absolute import (optional)
import subway_ai.config as config  # Changed to absolute import (optional)

if __name__ == "__main__":
    print("Executing Training Script...")
    # Basic check for GAME_REGION in config before starting
    try:
        import config
        if config.GAME_REGION is None:
             raise ValueError("GAME_REGION is not set in config.py")
        train_agent()
    except ImportError as e:
         print(f"Import Error: {e}")
         print("Please ensure all modules are correctly placed and requirements installed.")
    except ValueError as e:
         print(f"Configuration Error: {e}")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()