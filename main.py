"""
main.py — AfricaBrand RL Agent Simulation
==========================================
Entry point for running the best-performing trained agent.
Loads the best model, runs a full campaign episode, renders the
Pygame GUI, prints verbose terminal output, and serializes each
step to JSON (simulating a REST API response for frontend integration).

Usage:
    python main.py                        # run best model (auto-selects)
    python main.py --algo ppo             # specify algorithm
    python main.py --algo dqn --steps 30  # run DQN for 30 steps
    python main.py --random               # random agent baseline
    python main.py --api                  # output JSON to stdout (API mode)
"""

import os
import sys
import argparse
import json
import time
import itertools

import numpy as np
import pygame

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from environment.custom_env import AfricaBrandEnv, PLATFORMS, CONTENT_TYPES, TIME_SLOTS, AUDIENCE_SEGMENTS
from environment.rendering import AfricaBrandRenderer

MODELS_DIR_DQN = os.path.join(ROOT, "models", "dqn")
MODELS_DIR_PG  = os.path.join(ROOT, "models", "pg")

SEPARATOR = "─" * 70


def _load_model(algo: str):
    """Load a trained SB3 model."""
    if algo == "dqn":
        from stable_baselines3 import DQN
        from stable_baselines3.common.monitor import Monitor
        import gymnasium as gym
        from gymnasium.spaces import Discrete

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

        path = os.path.join(MODELS_DIR_DQN, "dqn_best.zip")
        raw_env = AfricaBrandEnv()
        env = FlattenActionWrapper(Monitor(raw_env))
        model = DQN.load(path, env=env)
        return model, raw_env, env

    elif algo in ("ppo", "reinforce"):
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        path = os.path.join(MODELS_DIR_PG, f"{algo}_best.zip")
        env = Monitor(AfricaBrandEnv())
        model = PPO.load(path, env=env)
        return model, env.unwrapped, env

    elif algo == "a2c":
        from stable_baselines3 import A2C
        from stable_baselines3.common.monitor import Monitor
        path = os.path.join(MODELS_DIR_PG, "a2c_best.zip")
        env = Monitor(AfricaBrandEnv())
        model = A2C.load(path, env=env)
        return model, env.unwrapped, env

    else:
        raise ValueError(f"Unknown algo: {algo}")


def _print_step_verbose(step: int, action: np.ndarray, reward: float, info: dict, cumulative: dict):
    """Print rich terminal output for each step."""
    print(f"\n{SEPARATOR}")
    print(f"  STEP {step:>3}  |  Day {info.get('day', '?'):>2}  |  Reward: {reward:+.4f}")
    print(SEPARATOR)
    print(f"  Decision Made:")
    print(f"    Platform      : {info.get('platform', '?')}")
    print(f"    Content Type  : {info.get('content_type', '?')}")
    print(f"    Time Slot     : {info.get('time_slot', '?')}")
    print(f"    Audience      : {info.get('segment', '?')}")
    print(f"    Budget Spent  : ${info.get('spend', 0):.2f}")
    print(f"  Outcomes:")
    print(f"    Engagement    : {info.get('engagement', 0):>10,.0f}")
    print(f"    Reach         : {info.get('reach', 0):>10,.0f}")
    print(f"    Revenue       : ${info.get('revenue', 0):>9.2f}")
    print(f"  Campaign State:")
    print(f"    Budget Left   : ${info.get('budget_remaining', 0):.2f}")
    print(f"    Sentiment     : {info.get('brand_sentiment', 0):+.3f}")
    print(f"  Cumulative:")
    print(f"    Total Engage  : {cumulative['engagement']:>10,.0f}")
    print(f"    Total Revenue : ${cumulative['revenue']:>9.2f}")


def _step_to_json(step: int, action: np.ndarray, reward: float, obs: np.ndarray,
                  info: dict, env_state: dict) -> dict:
    """Serialize a step into a JSON-serializable dict — API response format."""
    return {
        "step": step,
        "action": {
            "platform": info.get("platform"),
            "content_type": info.get("content_type"),
            "time_slot": info.get("time_slot"),
            "audience_segment": info.get("segment"),
            "budget_spent_usd": round(info.get("spend", 0), 2),
        },
        "outcomes": {
            "reward": round(reward, 6),
            "engagement": round(info.get("engagement", 0), 0),
            "reach": round(info.get("reach", 0), 0),
            "revenue_usd": round(info.get("revenue", 0), 2),
        },
        "environment_state": env_state,
    }


def run_episode(
    algo: str = "ppo",
    render: bool = True,
    verbose: bool = True,
    api_mode: bool = False,
    max_steps: int = 200,
    random_agent: bool = False,
):
    """Run one full campaign episode."""

    if random_agent:
        raw_env = AfricaBrandEnv()
        base_env = raw_env
        model = None
    else:
        model, raw_env, base_env = _load_model(algo)

    renderer = None
    if render:
        renderer = AfricaBrandRenderer(raw_env)

    obs, _ = base_env.reset(seed=42)
    cumulative = {"engagement": 0.0, "revenue": 0.0}
    episode_data = []
    step = 0

    if verbose and not api_mode:
        print(f"\n{'='*70}")
        print(f"  AfricaBrand RL Agent — {'Random Baseline' if random_agent else algo.upper()}")
        print(f"  Campaign Window: {raw_env.max_days} days  |  Budget: ${raw_env.max_budget:.0f}")
        print(f"{'='*70}")

    while step < max_steps:
        if random_agent:
            action = raw_env.sample_random_action()
            obs_env = obs
            reward, info, terminated, truncated = None, {}, False, False
            obs_env, reward, terminated, truncated, info = raw_env.step(action)
            obs = obs_env
        else:
            action, _ = model.predict(obs, deterministic=True)
            # For DQN, action is flat int — unwrap to get info from raw_env
            if algo == "dqn":
                # Step through wrapper env
                obs, reward, terminated, truncated, info = base_env.step(action)
            else:
                obs, reward, terminated, truncated, info = base_env.step(action)

        cumulative["engagement"] += info.get("engagement", 0)
        cumulative["revenue"]    += info.get("revenue", 0)

        env_state = raw_env.to_json_state() if hasattr(raw_env, "to_json_state") else {}
        action_arr = action if isinstance(action, np.ndarray) else np.array([action])

        if api_mode:
            step_json = _step_to_json(step, action_arr, reward, obs, info, env_state)
            episode_data.append(step_json)
        elif verbose:
            _print_step_verbose(step, action_arr, reward, info, cumulative)

        if render and renderer:
            renderer.update_info(info)
            renderer.render()

        step += 1
        if terminated or truncated:
            break

    if api_mode:
        summary = {
            "algorithm": "random" if random_agent else algo,
            "total_steps": step,
            "total_engagement": round(cumulative["engagement"], 0),
            "total_revenue_usd": round(cumulative["revenue"], 2),
            "final_budget_remaining": round(raw_env.budget_remaining, 2),
            "final_sentiment": round(raw_env.brand_sentiment, 3),
            "episode": episode_data,
        }
        print(json.dumps(summary, indent=2))
    elif verbose:
        print(f"\n{SEPARATOR}")
        print(f"  EPISODE COMPLETE")
        print(f"  Steps            : {step}")
        print(f"  Total Engagement : {cumulative['engagement']:,.0f}")
        print(f"  Total Revenue    : ${cumulative['revenue']:,.2f}")
        print(f"  Budget Remaining : ${raw_env.budget_remaining:.2f}")
        print(f"  Brand Sentiment  : {raw_env.brand_sentiment:+.3f}")
        print(SEPARATOR)

    if render and renderer:
        print("\nSimulation complete. Close the window to exit.")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        renderer.close()

    if hasattr(base_env, "close"):
        base_env.close()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AfricaBrand RL Agent — main runner")
    parser.add_argument(
        "--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default="ppo",
        help="Algorithm to use (default: ppo)"
    )
    parser.add_argument("--steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--no-render", action="store_true", help="Disable GUI")
    parser.add_argument("--no-verbose", action="store_true", help="Suppress terminal output")
    parser.add_argument("--random", action="store_true", help="Use random agent instead of trained model")
    parser.add_argument("--api", action="store_true", help="Output JSON to stdout (API mode)")
    args = parser.parse_args()

    run_episode(
        algo=args.algo,
        render=not args.no_render and not args.api,
        verbose=not args.no_verbose,
        api_mode=args.api,
        max_steps=args.steps,
        random_agent=args.random,
    )
