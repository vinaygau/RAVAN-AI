"""Validate the Ravan driving environment with Gymnasium's checker."""

import sys
from pathlib import Path

from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ravan_game.rl.environment import RavanDrivingEnv


def main() -> None:
    environment = RavanDrivingEnv(max_episode_steps=120)
    try:
        check_env(environment, skip_render_check=True)
    finally:
        environment.close()
    print("RavanDrivingEnv passed Gymnasium's environment checker.")


if __name__ == "__main__":
    main()
