"""
DQN Training Script — AfricaBrand Content Marketing RL
=======================================================
Trains a DQN agent using Stable-Baselines3 on the AfricaBrandEnv.
Includes a 10-run hyperparameter sweep and saves results to CSV.

Usage:
    python training/dqn_training.py --mode sweep   # full 10-run sweep
    python training/dqn_training.py --mode best    # train best config & save model
"""

import os
import sys
import argparse
import json
import csv
import time
from typing import List, Dict, Any

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch

# Make sure root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from environment.custom_env import AfricaBrandEnv

MODELS_DIR = os.path.join(ROOT, "models", "dqn")
LOGS_DIR   = os.path.join(ROOT, "logs", "dqn")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

TRAIN_STEPS = 8_000
EVAL_EPISODES = 3
SEED = 0


# ─── Hyperparameter configurations ────────────────────────────────────────────
# 10 distinct configurations varying: lr, gamma, batch_size, buffer_size,
# exploration_fraction, target_update_interval, net_arch, tau

SWEEP_CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Baseline
    dict(
        run=1, label="Baseline",
        learning_rate=1e-3, gamma=0.99, batch_size=64,
        buffer_size=20_000, exploration_fraction=0.2,
        target_update_interval=500, tau=1.0,
        net_arch=[64, 64], train_freq=4,
    ),
    # Run 2 — Lower LR, more stable
    dict(
        run=2, label="LowLR",
        learning_rate=1e-4, gamma=0.99, batch_size=64,
        buffer_size=20_000, exploration_fraction=0.2,
        target_update_interval=500, tau=1.0,
        net_arch=[64, 64], train_freq=4,
    ),
    # Run 3 — Higher LR, faster but less stable
    dict(
        run=3, label="HighLR",
        learning_rate=5e-3, gamma=0.99, batch_size=64,
        buffer_size=20_000, exploration_fraction=0.2,
        target_update_interval=500, tau=1.0,
        net_arch=[64, 64], train_freq=4,
    ),
    # Run 4 — Lower gamma (myopic)
    dict(
        run=4, label="LowGamma",
        learning_rate=1e-3, gamma=0.90, batch_size=64,
        buffer_size=20_000, exploration_fraction=0.2,
        target_update_interval=500, tau=1.0,
        net_arch=[64, 64], train_freq=4,
    ),
    # Run 5 — Larger buffer, slower experience replay warmup
    dict(
        run=5, label="LargeBuffer",
        learning_rate=1e-3, gamma=0.99, batch_size=64,
        buffer_size=30_000, exploration_fraction=0.3,
        target_update_interval=500, tau=1.0,
        net_arch=[64, 64], train_freq=4,
    ),
    # Run 6 — Larger batch size
    dict(
        run=6, label="LargeBatch",
        learning_rate=1e-3, gamma=0.99, batch_size=256,
        buffer_size=20_000, exploration_fraction=0.2,
        target_update_interval=500, tau=1.0,
        net_arch=[64, 64], train_freq=4,
    ),
    # Run 7 — Soft target updates (tau < 1)
    dict(
        run=7, label="SoftTarget",
        learning_rate=1e-3, gamma=0.99, batch_size=64,
        buffer_size=20_000, exploration_fraction=0.2,
        target_update_interval=200, tau=0.05,
        net_arch=[64, 64], train_freq=4,
    ),
    # Run 8 — Deeper network
    dict(
        run=8, label="DeepNet",
        learning_rate=1e-3, gamma=0.99, batch_size=64,
        buffer_size=20_000, exploration_fraction=0.2,
        target_update_interval=500, tau=1.0,
        net_arch=[256, 256, 128], train_freq=4,
    ),
    # Run 9 — More exploration
    dict(
        run=9, label="MoreExplore",
        learning_rate=1e-3, gamma=0.99, batch_size=64,
        buffer_size=20_000, exploration_fraction=0.5,
        target_update_interval=500, tau=1.0,
        net_arch=[64, 64], train_freq=4,
    ),
    # Run 10 — Combined tuned config
    dict(
        run=10, label="Tuned",
        learning_rate=2e-4, gamma=0.995, batch_size=128,
        buffer_size=30_000, exploration_fraction=0.25,
        target_update_interval=300, tau=0.1,
        net_arch=[128, 128], train_freq=4,
    ),
]

# The single best config selected after sweep analysis
BEST_CONFIG = SWEEP_CONFIGS[9]  # Run 10 — Tuned


def make_env(seed=None):
    def _make():
        env = AfricaBrandEnv()
        env = Monitor(env)
        return env
    return _make


def _flatten_action_wrapper(env):
    """DQN requires a flat Discrete action space. Wrap MultiDiscrete env."""
    from gymnasium.spaces import Discrete
    import itertools

    class FlattenActionWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            nvec = env.action_space.nvec
            self._nvec = nvec
            total = int(np.prod(nvec))
            self.action_space = Discrete(total)
            self._all_actions = list(itertools.product(*[range(n) for n in nvec]))

        def step(self, action):
            multi = np.array(self._all_actions[action])
            return self.env.step(multi)

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

    return FlattenActionWrapper(env)


def build_model(config: Dict, env) -> DQN:
    policy_kwargs = dict(net_arch=config["net_arch"])
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        batch_size=min(config["batch_size"], 32),
        buffer_size=config["buffer_size"],
        exploration_fraction=config["exploration_fraction"],
        target_update_interval=config["target_update_interval"],
        tau=config["tau"],
        train_freq=config["train_freq"],
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=SEED,
        tensorboard_log=LOGS_DIR,
    )
    return model


def run_sweep():
    results = []
    print("\n" + "=" * 60)
    print("  DQN Hyperparameter Sweep — AfricaBrand Env")
    print("=" * 60)

    for cfg in SWEEP_CONFIGS:
        run_label = f"Run {cfg['run']:02d} ({cfg['label']})"
        print(f"\n[DQN] {run_label}")

        env = _flatten_action_wrapper(Monitor(AfricaBrandEnv()))
        eval_env = _flatten_action_wrapper(Monitor(AfricaBrandEnv()))

        t0    = time.time()
        model = build_model(cfg, env)
        model.learn(total_timesteps=TRAIN_STEPS, progress_bar=False)
        elapsed = time.time() - t0

        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES)

        print(f"  Mean reward: {mean_r:.3f} ± {std_r:.3f}  |  Time: {elapsed:.1f}s")

        results.append({
            "Run": cfg["run"],
            "Label": cfg["label"],
            "LR": cfg["learning_rate"],
            "Gamma": cfg["gamma"],
            "Batch Size": cfg["batch_size"],
            "Buffer Size": cfg["buffer_size"],
            "Explore Frac": cfg["exploration_fraction"],
            "Target Update": cfg["target_update_interval"],
            "Tau": cfg["tau"],
            "Net Arch": str(cfg["net_arch"]),
            "Mean Reward": round(mean_r, 4),
            "Std Reward": round(std_r, 4),
            "Training Time (s)": round(elapsed, 1),
        })

        env.close()
        eval_env.close()

    # Save CSV
    csv_path = os.path.join(LOGS_DIR, "dqn_sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSweep complete. Results saved to: {csv_path}")
    _print_table(results)
    return results


def _print_table(results):
    print("\n" + "-" * 80)
    print(f"{'Run':<4} {'Label':<14} {'LR':<9} {'Gamma':<7} {'Batch':<6} "
          f"{'Mean Reward':<14} {'Std':<8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['Run']:<4} {r['Label']:<14} {r['LR']:<9} {r['Gamma']:<7} "
            f"{r['Batch Size']:<6} {r['Mean Reward']:<14.4f} {r['Std Reward']:<8.4f}"
        )
    print("-" * 80)


def train_best():
    """Train the best-performing configuration and save the model."""
    print("\nTraining best DQN configuration...")
    cfg  = BEST_CONFIG
    env  = _flatten_action_wrapper(Monitor(AfricaBrandEnv()))
    model = build_model(cfg, env)
    model.learn(total_timesteps=20_000, progress_bar=True)

    save_path = os.path.join(MODELS_DIR, "dqn_best")
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    env.close()
    return model


def load_best():
    save_path = os.path.join(MODELS_DIR, "dqn_best.zip")
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"No saved model at {save_path}. Run with --mode best first.")
    env   = _flatten_action_wrapper(Monitor(AfricaBrandEnv()))
    model = DQN.load(save_path, env=env)
    return model, env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["sweep", "best", "eval"], default="sweep",
        help="sweep: run all 10 configs; best: train & save best; eval: evaluate saved model"
    )
    args = parser.parse_args()

    if args.mode == "sweep":
        run_sweep()
    elif args.mode == "best":
        train_best()
    elif args.mode == "eval":
        model, env = load_best()
        mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=5)
        print(f"Evaluation — Mean: {mean_r:.4f} ± {std_r:.4f}")
        env.close()
