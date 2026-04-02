"""
Policy Gradient Training Script — AfricaBrand Content Marketing RL
===================================================================
Trains REINFORCE, PPO, and A2C agents using Stable-Baselines3 on AfricaBrandEnv.
Includes 10-run hyperparameter sweeps for each algorithm and saves results to CSV.

Usage:
    python training/pg_training.py --algo all --mode sweep
    python training/pg_training.py --algo ppo --mode best
    python training/pg_training.py --algo a2c --mode eval
"""

import os
import sys
import argparse
import csv
import time
from typing import List, Dict, Any

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from environment.custom_env import AfricaBrandEnv

MODELS_DIR = os.path.join(ROOT, "models", "pg")
LOGS_DIR   = os.path.join(ROOT, "logs", "pg")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

TRAIN_STEPS   = 150_000
EVAL_EPISODES = 20
SEED          = 0


# ─── PPO Hyperparameter Sweep ─────────────────────────────────────────────────

PPO_CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Baseline PPO
    dict(run=1, label="Baseline",
         learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95,
         net_arch=[64, 64]),
    # Run 2 — Lower LR
    dict(run=2, label="LowLR",
         learning_rate=1e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95,
         net_arch=[64, 64]),
    # Run 3 — Higher LR
    dict(run=3, label="HighLR",
         learning_rate=1e-3, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95,
         net_arch=[64, 64]),
    # Run 4 — Lower gamma
    dict(run=4, label="LowGamma",
         learning_rate=3e-4, gamma=0.90, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95,
         net_arch=[64, 64]),
    # Run 5 — High entropy (more exploration)
    dict(run=5, label="HighEntropy",
         learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.05, clip_range=0.2, gae_lambda=0.95,
         net_arch=[64, 64]),
    # Run 6 — Tight clip range
    dict(run=6, label="TightClip",
         learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.1, gae_lambda=0.95,
         net_arch=[64, 64]),
    # Run 7 — Wide clip range
    dict(run=7, label="WideClip",
         learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.3, gae_lambda=0.95,
         net_arch=[64, 64]),
    # Run 8 — More epochs per update
    dict(run=8, label="MoreEpochs",
         learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=20, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95,
         net_arch=[64, 64]),
    # Run 9 — Larger network
    dict(run=9, label="DeepNet",
         learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95,
         net_arch=[256, 256, 128]),
    # Run 10 — Combined tuned
    dict(run=10, label="Tuned",
         learning_rate=2e-4, gamma=0.995, n_steps=1024, batch_size=128,
         n_epochs=15, ent_coef=0.02, clip_range=0.2, gae_lambda=0.98,
         net_arch=[128, 128]),
]


# ─── A2C Hyperparameter Sweep ─────────────────────────────────────────────────

A2C_CONFIGS: List[Dict[str, Any]] = [
    dict(run=1,  label="Baseline",     learning_rate=7e-4, gamma=0.99,  n_steps=5,   ent_coef=0.00, vf_coef=0.5, max_grad_norm=0.5,  net_arch=[64, 64]),
    dict(run=2,  label="LowLR",        learning_rate=1e-4, gamma=0.99,  n_steps=5,   ent_coef=0.00, vf_coef=0.5, max_grad_norm=0.5,  net_arch=[64, 64]),
    dict(run=3,  label="HighLR",       learning_rate=3e-3, gamma=0.99,  n_steps=5,   ent_coef=0.00, vf_coef=0.5, max_grad_norm=0.5,  net_arch=[64, 64]),
    dict(run=4,  label="LowGamma",     learning_rate=7e-4, gamma=0.90,  n_steps=5,   ent_coef=0.00, vf_coef=0.5, max_grad_norm=0.5,  net_arch=[64, 64]),
    dict(run=5,  label="HighEntropy",  learning_rate=7e-4, gamma=0.99,  n_steps=5,   ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,  net_arch=[64, 64]),
    dict(run=6,  label="LongerRoll",   learning_rate=7e-4, gamma=0.99,  n_steps=20,  ent_coef=0.00, vf_coef=0.5, max_grad_norm=0.5,  net_arch=[64, 64]),
    dict(run=7,  label="HighVF",       learning_rate=7e-4, gamma=0.99,  n_steps=5,   ent_coef=0.00, vf_coef=1.0, max_grad_norm=0.5,  net_arch=[64, 64]),
    dict(run=8,  label="LowVF",        learning_rate=7e-4, gamma=0.99,  n_steps=5,   ent_coef=0.00, vf_coef=0.25,max_grad_norm=0.5,  net_arch=[64, 64]),
    dict(run=9,  label="DeepNet",      learning_rate=7e-4, gamma=0.99,  n_steps=5,   ent_coef=0.00, vf_coef=0.5, max_grad_norm=0.5,  net_arch=[256, 256, 128]),
    dict(run=10, label="Tuned",        learning_rate=3e-4, gamma=0.995, n_steps=15,  ent_coef=0.02, vf_coef=0.5, max_grad_norm=0.5,  net_arch=[128, 128]),
]


# ─── REINFORCE Sweep ──────────────────────────────────────────────────────────
# REINFORCE is not natively in SB3; we use PPO with n_epochs=1, no clipping,
# and very high n_steps (Monte Carlo rollouts) to approximate it.

REINFORCE_CONFIGS: List[Dict[str, Any]] = [
    dict(run=1,  label="Baseline",   learning_rate=1e-3, gamma=0.99,  n_steps=512, ent_coef=0.00, normalize_advantage=True),
    dict(run=2,  label="LowLR",      learning_rate=1e-4, gamma=0.99,  n_steps=512, ent_coef=0.00, normalize_advantage=True),
    dict(run=3,  label="HighLR",     learning_rate=5e-3, gamma=0.99,  n_steps=512, ent_coef=0.00, normalize_advantage=True),
    dict(run=4,  label="LowGamma",   learning_rate=1e-3, gamma=0.90,  n_steps=512, ent_coef=0.00, normalize_advantage=True),
    dict(run=5,  label="Entropy",    learning_rate=1e-3, gamma=0.99,  n_steps=512, ent_coef=0.05, normalize_advantage=True),
    dict(run=6,  label="LongRoll",   learning_rate=1e-3, gamma=0.99,  n_steps=1024,ent_coef=0.00, normalize_advantage=True),
    dict(run=7,  label="ShortRoll",  learning_rate=1e-3, gamma=0.99,  n_steps=256, ent_coef=0.00, normalize_advantage=True),
    dict(run=8,  label="NoNorm",     learning_rate=1e-3, gamma=0.99,  n_steps=512, ent_coef=0.00, normalize_advantage=False),
    dict(run=9,  label="HighGamma",  learning_rate=1e-3, gamma=0.999, n_steps=512, ent_coef=0.00, normalize_advantage=True),
    dict(run=10, label="Tuned",      learning_rate=5e-4, gamma=0.995, n_steps=800, ent_coef=0.02, normalize_advantage=True),
]

BEST_PPO_CONFIG        = PPO_CONFIGS[9]
BEST_A2C_CONFIG        = A2C_CONFIGS[9]
BEST_REINFORCE_CONFIG  = REINFORCE_CONFIGS[9]


# ─── Builder helpers ──────────────────────────────────────────────────────────

def make_env():
    return Monitor(AfricaBrandEnv())


def build_ppo(cfg: Dict, env) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        ent_coef=cfg["ent_coef"],
        clip_range=cfg["clip_range"],
        gae_lambda=cfg["gae_lambda"],
        policy_kwargs=dict(net_arch=cfg["net_arch"]),
        verbose=0,
        seed=SEED,
        tensorboard_log=LOGS_DIR,
    )


def build_a2c(cfg: Dict, env) -> A2C:
    return A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        n_steps=cfg["n_steps"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        policy_kwargs=dict(net_arch=cfg["net_arch"]),
        verbose=0,
        seed=SEED,
        tensorboard_log=LOGS_DIR,
    )


def build_reinforce(cfg: Dict, env) -> PPO:
    """
    REINFORCE approximation via PPO with:
    - n_epochs=1 (single gradient step per rollout)
    - clip_range=1.0 (no clipping — pure policy gradient)
    - vf_coef=0 (no value function bootstrapping in loss)
    - Full Monte Carlo rollout (n_steps large)
    """
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["n_steps"],   # one batch = full rollout
        n_epochs=1,
        clip_range=1.0,
        ent_coef=cfg["ent_coef"],
        vf_coef=0.0,
        normalize_advantage=cfg["normalize_advantage"],
        gae_lambda=1.0,              # no GAE bias — pure MC returns
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=0,
        seed=SEED,
        tensorboard_log=LOGS_DIR,
    )


# ─── Sweep runner ─────────────────────────────────────────────────────────────

def run_sweep(algo: str):
    if algo == "ppo":
        configs = PPO_CONFIGS
        build_fn = build_ppo
        label = "PPO"
    elif algo == "a2c":
        configs = A2C_CONFIGS
        build_fn = build_a2c
        label = "A2C"
    elif algo == "reinforce":
        configs = REINFORCE_CONFIGS
        build_fn = build_reinforce
        label = "REINFORCE"
    else:
        raise ValueError(f"Unknown algo: {algo}")

    print(f"\n{'='*60}")
    print(f"  {label} Hyperparameter Sweep — AfricaBrand Env")
    print(f"{'='*60}")

    results = []
    for cfg in configs:
        run_label = f"Run {cfg['run']:02d} ({cfg['label']})"
        print(f"\n[{label}] {run_label}")

        env      = make_env()
        eval_env = make_env()

        t0    = time.time()
        model = build_fn(cfg, env)
        model.learn(total_timesteps=TRAIN_STEPS, progress_bar=False)
        elapsed = time.time() - t0

        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES)
        print(f"  Mean reward: {mean_r:.3f} ± {std_r:.3f}  |  Time: {elapsed:.1f}s")

        row = {
            "Run": cfg["run"],
            "Label": cfg["label"],
            "LR": cfg["learning_rate"],
            "Gamma": cfg["gamma"],
            "Mean Reward": round(mean_r, 4),
            "Std Reward": round(std_r, 4),
            "Training Time (s)": round(elapsed, 1),
        }
        # Algo-specific columns
        if algo == "ppo":
            row.update({
                "N Steps": cfg["n_steps"], "Batch Size": cfg["batch_size"],
                "N Epochs": cfg["n_epochs"], "Ent Coef": cfg["ent_coef"],
                "Clip Range": cfg["clip_range"], "GAE Lambda": cfg["gae_lambda"],
                "Net Arch": str(cfg["net_arch"]),
            })
        elif algo == "a2c":
            row.update({
                "N Steps": cfg["n_steps"], "Ent Coef": cfg["ent_coef"],
                "VF Coef": cfg["vf_coef"], "Max Grad Norm": cfg["max_grad_norm"],
                "Net Arch": str(cfg["net_arch"]),
            })
        elif algo == "reinforce":
            row.update({
                "N Steps": cfg["n_steps"], "Ent Coef": cfg["ent_coef"],
                "Normalize Adv": cfg["normalize_advantage"],
            })

        results.append(row)
        env.close()
        eval_env.close()

    csv_path = os.path.join(LOGS_DIR, f"{algo}_sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSweep complete. Results saved to: {csv_path}")
    _print_table(results, label)
    return results


def _print_table(results, algo):
    print(f"\n{algo} Results:")
    print("-" * 70)
    print(f"{'Run':<4} {'Label':<14} {'LR':<10} {'Gamma':<7} {'Mean Reward':<14} {'Std'}")
    print("-" * 70)
    for r in results:
        print(f"{r['Run']:<4} {r['Label']:<14} {r['LR']:<10} {r['Gamma']:<7} "
              f"{r['Mean Reward']:<14.4f} {r['Std Reward']:.4f}")
    print("-" * 70)


# ─── Best model training ──────────────────────────────────────────────────────

def train_best(algo: str):
    print(f"\nTraining best {algo.upper()} configuration...")
    env = make_env()

    if algo == "ppo":
        model = build_ppo(BEST_PPO_CONFIG, env)
    elif algo == "a2c":
        model = build_a2c(BEST_A2C_CONFIG, env)
    elif algo == "reinforce":
        model = build_reinforce(BEST_REINFORCE_CONFIG, env)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    model.learn(total_timesteps=300_000, progress_bar=True)
    save_path = os.path.join(MODELS_DIR, f"{algo}_best")
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    env.close()
    return model


def load_best(algo: str):
    save_path = os.path.join(MODELS_DIR, f"{algo}_best.zip")
    env = make_env()
    if algo in ("ppo", "reinforce"):
        model = PPO.load(save_path, env=env)
    elif algo == "a2c":
        model = A2C.load(save_path, env=env)
    else:
        raise ValueError(f"Unknown algo: {algo}")
    return model, env


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", choices=["ppo", "a2c", "reinforce", "all"], default="all",
        help="Which algorithm to run"
    )
    parser.add_argument(
        "--mode", choices=["sweep", "best", "eval"], default="sweep",
        help="sweep: hyperparameter sweep; best: train & save best; eval: evaluate saved"
    )
    args = parser.parse_args()

    algos = ["reinforce", "ppo", "a2c"] if args.algo == "all" else [args.algo]

    for algo in algos:
        if args.mode == "sweep":
            run_sweep(algo)
        elif args.mode == "best":
            train_best(algo)
        elif args.mode == "eval":
            model, env = load_best(algo)
            mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=30)
            print(f"[{algo.upper()}] Evaluation — Mean: {mean_r:.4f} ± {std_r:.4f}")
            env.close()
