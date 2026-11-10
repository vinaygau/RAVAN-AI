"""Train PPO on the Ravan simulation with validation checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src.ravan_game.rl.environment import RavanDrivingEnv
from src.ravan_game.rl.reward import RewardConfig
from src.ravan_game.rl.scenarios import TRAIN_SCENARIOS, VALIDATION_SCENARIOS


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume PPO weights from an existing .zip model.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.timesteps < 1 or args.envs < 1 or args.eval_every < 1 or args.eval_episodes < 1:
        raise SystemExit("timesteps, envs, eval-every, and eval-episodes must be positive.")
    run_dir = args.run_dir or ROOT / "data" / "rl_runs" / datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    reward = RewardConfig.from_json()

    def make_env(rank: int, families: tuple[str, ...]):
        def _factory():
            env = RavanDrivingEnv(
                max_episode_steps=800,
                scenario_families=families,
                reward_config=reward,
            )
            return env
        return _factory

    train_env = VecMonitor(DummyVecEnv([
        make_env(rank, TRAIN_SCENARIOS) for rank in range(args.envs)
    ]))
    eval_env = VecMonitor(DummyVecEnv([make_env(1000, VALIDATION_SCENARIOS)]))
    checkpoint = CheckpointCallback(
        save_freq=max(args.eval_every // args.envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_ravan",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    evaluate = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "validation"),
        eval_freq=max(args.eval_every // args.envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    metadata = {
        "seed": args.seed,
        "requested_timesteps": args.timesteps,
        "resumed_from": str(args.resume.resolve()) if args.resume else None,
        "parallel_envs": args.envs,
        "train_scenarios": TRAIN_SCENARIOS,
        "validation_scenarios": VALIDATION_SCENARIOS,
        "held_out_test_scenarios": ("cut_in", "blind_spot", "dense_traffic"),
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "device": args.device,
        "reward": reward.__dict__,
    }
    (run_dir / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.resume:
        model = PPO.load(args.resume, env=train_env, device=args.device)
        model.set_random_seed(args.seed)
    else:
        model = PPO(
            "MultiInputPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=128,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            seed=args.seed,
            device=args.device,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint, evaluate],
            progress_bar=False,
            reset_num_timesteps=not bool(args.resume),
            tb_log_name="RavanPPO",
        )
        model.save(run_dir / "final_model")
    finally:
        train_env.close()
        eval_env.close()
    print(f"Training complete. Model: {run_dir / 'final_model.zip'}")
    print(f"Validation best model: {run_dir / 'best' / 'best_model.zip'}")
    print(f"Experiment metadata: {run_dir / 'experiment.json'}")


if __name__ == "__main__":
    main()
