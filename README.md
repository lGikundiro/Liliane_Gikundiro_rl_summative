# AfricaBrand Content Marketing RL

A reinforcement learning system that trains an intelligent content marketing agent for African local brands. The agent learns to optimize social media decisions — platform selection, content type, posting time, audience targeting, and budget allocation — to maximize engagement, reach, and brand revenue across a simulated 30-day campaign.

---

## Mission

To build systems where storytelling creates real economic opportunity — for brands, for youth, and for Africa. This project frames digital marketing content management as an RL problem, enabling brands to discover optimal content strategies through learned policies rather than trial-and-error or intuition alone.

---

## Project Structure

```text
project_root/
├── environment/
│   ├── custom_env.py        # Custom Gymnasium environment
│   ├── rendering.py         # Pygame visualization + static demo
│   └── __init__.py
├── training/
│   ├── dqn_training.py      # DQN training + 10-run hyperparameter sweep
│   ├── pg_training.py       # REINFORCE, PPO, A2C training + sweeps
│   └── __init__.py
├── models/
│   ├── dqn/                 # Saved DQN models
│   └── pg/                  # Saved policy gradient models
├── evaluation/
│   ├── analysis.py          # Generates all report visualizations
│   └── plots/               # Output directory for figures
├── main.py                  # Entry point — run best agent
├── requirements.txt
└── README.md
```

---

## Environment Overview

**AfricaBrandEnv** is a custom Gymnasium environment simulating a 30-day social media content campaign for a local African brand.

### Observation Space (20-dimensional, continuous)

| Index | Feature | Range |
|-------|---------|-------|
| 0 | Day in campaign (normalized) | [0, 1] |
| 1 | Remaining budget (normalized) | [0, 1] |
| 2 | Cumulative engagement (normalized) | [0, 1] |
| 3 | Cumulative reach (normalized) | [0, 1] |
| 4 | Brand sentiment score | [−1, 1] |
| 5–9 | Platform fatigue (per platform) | [0, 1] |
| 10–14 | Content performance history (rolling avg) | [0, 1] |
| 15 | Audience saturation index | [0, 1] |
| 16 | Trend momentum signal | [0, 1] |
| 17 | Competitor activity level | [0, 1] |
| 18 | Revenue-per-engagement estimate | [0, 1] |
| 19 | Campaign health index (composite) | [0, 1] |

### Action Space (MultiDiscrete)

| Dimension | Choices |
|-----------|---------|
| Platform | Instagram, TikTok, Twitter/X, Facebook, YouTube |
| Content type | Reel/Short, Static Image, Story, Long Video, Thread/Post, Live |
| Time slot | Morning, Midday, Evening, Late Night |
| Audience segment | Youth 16–24, Young Adults 25–34, Adults 35–49 |
| Budget tier | 5%, 10%, 20%, 30%, 40% of remaining budget |

Total action combinations: 5 × 6 × 4 × 3 × 5 = **1,800**

### Reward Structure

The shaped reward at each step combines:

- **Engagement reward** — primary signal, scaled from interaction count
- **Reach bonus** — weighted at 0.4× engagement reward
- **Revenue signal** — conversion × average order value, weighted 0.6×
- **Fatigue penalty** — discourages overposting on one platform
- **Consistency bonus** — rewards regular posting cadence
- **Budget penalty** — penalizes exhausting the budget prematurely

### Terminal Conditions

An episode ends when:

1. The 30-day campaign window is complete, or
2. The remaining budget drops below the minimum platform cost

---

## Setup

```bash
# Clone the repository
git clone https://github.com/<your_username>/student_name_rl_summative.git
cd student_name_rl_summative

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Code

### Static Environment Demo (no model, random actions)

```bash
python environment/rendering.py
```

This launches the Pygame GUI and shows the agent taking random actions — demonstrating the environment visualization without any trained policy.

### Train All Models

```bash
# DQN hyperparameter sweep (10 configurations)
python training/dqn_training.py --mode sweep

# Policy gradient sweeps (REINFORCE, PPO, A2C)
python training/pg_training.py --algo all --mode sweep

# Train and save best configurations
python training/dqn_training.py --mode best
python training/pg_training.py --algo ppo --mode best
python training/pg_training.py --algo a2c --mode best
python training/pg_training.py --algo reinforce --mode best
```

### Run Best Agent

```bash
# Default: PPO with GUI and terminal output
python main.py

# Specific algorithm
python main.py --algo dqn
python main.py --algo a2c

# Random baseline for comparison
python main.py --random

# JSON API mode (no GUI, outputs structured JSON)
python main.py --api --algo ppo
```

### Generate Report Visualizations

```bash
python evaluation/analysis.py --mode all
```

Outputs are saved to `evaluation/plots/`.

---

## Algorithms Implemented

| Algorithm | Type | Library |
|-----------|------|---------|
| DQN | Value-based | Stable-Baselines3 |
| REINFORCE | Policy Gradient (Monte Carlo) | SB3 (PPO approximation) |
| PPO | Policy Gradient (clipped surrogate) | Stable-Baselines3 |
| A2C | Actor-Critic | Stable-Baselines3 |

Each algorithm undergoes a 10-run hyperparameter sweep. Results are saved as CSV files in `logs/dqn/` and `logs/pg/`.

---

## API / Frontend Integration

Running `python main.py --api --algo ppo` outputs a JSON response suitable for consumption by a frontend or REST API. Each step contains the agent's decision, outcomes, and full environment state:

```json
{
  "algorithm": "ppo",
  "total_steps": 30,
  "total_engagement": 42500,
  "total_revenue_usd": 1284.50,
  "episode": [
    {
      "step": 0,
      "action": {
        "platform": "TikTok",
        "content_type": "Reel/Short",
        "time_slot": "Evening (18-21)",
        "audience_segment": "Youth 16-24",
        "budget_spent_usd": 50.0
      },
      "outcomes": {
        "reward": 1.842,
        "engagement": 1240,
        "reach": 9800,
        "revenue_usd": 62.50
      },
      "environment_state": { ... }
    }
  ]
}
```

---

## Hyperparameter Tables

Sweep result CSV files are generated at:

- `logs/dqn/dqn_sweep_results.csv`
- `logs/pg/ppo_sweep_results.csv`
- `logs/pg/a2c_sweep_results.csv`
- `logs/pg/reinforce_sweep_results.csv`

---

## Key Design Decisions

**Why MultiDiscrete actions?** A single discrete action cannot represent the combinatorial nature of content decisions — platform, type, timing, audience, and spend are orthogonal axes that should be learned independently.

**Why platform fatigue?** Overposting on one platform yields diminishing returns in real campaigns. This dynamic forces the agent to distribute attention across channels rather than collapsing to a single-platform policy.

**Why shaped rewards?** Pure engagement reward ignores budget efficiency and long-term brand health. Revenue signals, sentiment drift, and consistency bonuses push the agent toward strategies that are sustainable across the full 30-day window.

---

## Dependencies

See `requirements.txt`. Key packages:

- `gymnasium` — environment API
- `stable-baselines3` — RL algorithm implementations
- `pygame` — environment visualization
- `torch` — neural network backend
- `matplotlib` — analysis plots
- `tensorboard` — training metrics

---

## Author

Summative RL Project — Advanced Machine Learning
