"""
AfricaBrand Content Marketing RL Environment
=============================================
A custom Gymnasium environment simulating a social media content manager
optimizing digital marketing campaigns for African local brands.

The agent decides what content type to post, on which platform, at what time,
and with what budget allocation — learning to maximize engagement, reach,
and brand revenue over a campaign window.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


# ─── Constants ────────────────────────────────────────────────────────────────

PLATFORMS        = ["Instagram", "TikTok", "Twitter/X", "Facebook", "YouTube"]
CONTENT_TYPES    = ["Reel/Short", "Static Image", "Story", "Long Video", "Thread/Post", "Live"]
TIME_SLOTS       = ["Morning (6-9)", "Midday (12-14)", "Evening (18-21)", "Late Night (21-24)"]
AUDIENCE_SEGMENTS = ["Youth 16-24", "Young Adults 25-34", "Adults 35-49"]

N_PLATFORMS   = len(PLATFORMS)
N_CONTENT     = len(CONTENT_TYPES)
N_TIMESLOTS   = len(TIME_SLOTS)
N_SEGMENTS    = len(AUDIENCE_SEGMENTS)

MAX_DAYS      = 30        # campaign window
MAX_BUDGET    = 1000.0    # USD per campaign
MIN_BUDGET    = 0.0


# ─── Platform × Content affinity matrix ───────────────────────────────────────
# Rows = platforms, Cols = content types
# Values represent base engagement multiplier (domain knowledge encoded)

AFFINITY = np.array([
    # Reel  Static  Story  LongVid  Thread  Live
    [0.95,  0.70,   0.85,  0.40,    0.30,   0.75],   # Instagram
    [0.98,  0.50,   0.60,  0.55,    0.25,   0.80],   # TikTok
    [0.40,  0.55,   0.35,  0.30,    0.90,   0.45],   # Twitter/X
    [0.65,  0.75,   0.70,  0.60,    0.55,   0.70],   # Facebook
    [0.50,  0.20,   0.15,  0.95,    0.10,   0.85],   # YouTube
], dtype=np.float32)

# Cost per post per platform (USD)
PLATFORM_COST = np.array([15.0, 10.0, 5.0, 12.0, 20.0], dtype=np.float32)

# Time-slot audience activity multipliers (per platform)
TIMESLOT_MULTIPLIER = np.array([
    [0.80, 0.75, 1.00, 0.60],   # Instagram
    [0.70, 0.85, 1.00, 0.90],   # TikTok
    [0.75, 0.90, 0.95, 0.70],   # Twitter/X
    [0.85, 0.70, 0.90, 0.50],   # Facebook
    [0.65, 0.60, 0.95, 0.45],   # YouTube
], dtype=np.float32)

# Segment receptivity per platform
SEGMENT_RECEPTIVITY = np.array([
    [0.90, 0.85, 0.60],   # Instagram
    [0.98, 0.70, 0.40],   # TikTok
    [0.75, 0.88, 0.65],   # Twitter/X
    [0.55, 0.80, 0.90],   # Facebook
    [0.65, 0.80, 0.75],   # YouTube
], dtype=np.float32)


# ─── Environment ──────────────────────────────────────────────────────────────

class AfricaBrandEnv(gym.Env):
    """
    AfricaBrand Content Marketing Environment.

    Observation Space (continuous, 20-dimensional):
        [0]   Day in campaign (normalized 0–1)
        [1]   Remaining budget (normalized 0–1)
        [2]   Cumulative engagement score (normalized)
        [3]   Cumulative reach score (normalized)
        [4]   Brand sentiment score (–1 to 1)
        [5-9] Platform fatigue levels (5 platforms, 0–1 each)
        [10-14] Content type recent performance (5 platforms' top content affinity rolling avg)
        [15]  Audience saturation index (0–1)
        [16]  Trend momentum score (external signal, 0–1)
        [17]  Competitor activity level (0–1)
        [18]  Revenue-per-engagement rolling estimate (normalized)
        [19]  Campaign health index (composite, 0–1)

    Action Space (MultiDiscrete):
        [0] Platform choice       (0–4)
        [1] Content type choice   (0–5)
        [2] Time slot choice      (0–3)
        [3] Audience segment      (0–2)
        [4] Budget allocation     (0–4 → maps to 5%, 10%, 20%, 30%, 40% of remaining budget)

    Reward:
        Shaped reward combining:
        - Engagement reward (primary)
        - Reach bonus
        - Revenue signal
        - Penalty for over-budget actions
        - Penalty for platform fatigue
        - Bonus for consistent posting
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()

        self.render_mode = render_mode
        self.max_days    = MAX_DAYS
        self.max_budget  = MAX_BUDGET

        # ── Action space ──────────────────────────────────────────────────────
        self.action_space = spaces.MultiDiscrete([
            N_PLATFORMS,   # platform
            N_CONTENT,     # content type
            N_TIMESLOTS,   # time slot
            N_SEGMENTS,    # audience segment
            5,             # budget allocation tier (0–4)
        ])

        # ── Observation space ─────────────────────────────────────────────────
        low  = np.full(20, -1.0, dtype=np.float32)
        high = np.full(20,  1.0, dtype=np.float32)
        # Most observations are in [0, 1]; sentiment is in [-1, 1]
        low[4]  = -1.0
        high[4] =  1.0
        low[[i for i in range(20) if i != 4]] = 0.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ── Internal state ────────────────────────────────────────────────────
        self._rng = np.random.default_rng(seed)
        self._reset_state()

        # Rendering
        self._renderer = None

    # ── State helpers ──────────────────────────────────────────────────────────

    def _reset_state(self):
        self.day               = 0
        self.budget_remaining  = self.max_budget
        self.cumulative_engage = 0.0
        self.cumulative_reach  = 0.0
        self.brand_sentiment   = 0.0          # –1 (negative) to +1 (positive)
        self.platform_fatigue  = np.zeros(N_PLATFORMS, dtype=np.float32)
        self.content_perf_hist = np.full(N_PLATFORMS, 0.5, dtype=np.float32)
        self.audience_sat      = 0.0
        self.trend_momentum    = float(self._rng.uniform(0.3, 0.8))
        self.competitor_level  = float(self._rng.uniform(0.1, 0.6))
        self.rpe               = 0.0          # revenue per engagement
        self.posts_this_week   = 0
        self.total_revenue     = 0.0
        self.action_history    = []
        self.reward_history    = []
        self.last_post_day     = -1

    def _get_obs(self) -> np.ndarray:
        obs = np.array([
            self.day / self.max_days,
            self.budget_remaining / self.max_budget,
            min(self.cumulative_engage / 50000.0, 1.0),
            min(self.cumulative_reach  / 200000.0, 1.0),
            np.clip(self.brand_sentiment, -1.0, 1.0),
            *np.clip(self.platform_fatigue, 0.0, 1.0),
            *np.clip(self.content_perf_hist, 0.0, 1.0),
            np.clip(self.audience_sat, 0.0, 1.0),
            np.clip(self.trend_momentum, 0.0, 1.0),
            np.clip(self.competitor_level, 0.0, 1.0),
            np.clip(self.rpe / 5.0, 0.0, 1.0),
            self._campaign_health(),
        ], dtype=np.float32)
        return obs

    def _campaign_health(self) -> float:
        """Composite index: budget pacing + engagement trajectory + sentiment."""
        budget_pace = 1.0 - (self.budget_remaining / self.max_budget)
        expected_pace = self.day / max(self.max_days, 1)
        pacing_ok = 1.0 - abs(budget_pace - expected_pace)
        sentiment_contrib = (self.brand_sentiment + 1.0) / 2.0
        engage_contrib = min(self.cumulative_engage / (self.day * 500 + 1), 1.0)
        return float(np.clip((pacing_ok + sentiment_contrib + engage_contrib) / 3.0, 0.0, 1.0))

    def _get_info(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "budget_remaining": self.budget_remaining,
            "total_engagement": self.cumulative_engage,
            "total_reach": self.cumulative_reach,
            "total_revenue": self.total_revenue,
            "brand_sentiment": self.brand_sentiment,
            "platform_fatigue": self.platform_fatigue.tolist(),
        }

    # ── Core Gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        # Randomize initial market conditions slightly
        self.trend_momentum   = float(self._rng.uniform(0.2, 0.9))
        self.competitor_level = float(self._rng.uniform(0.05, 0.75))
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        platform_idx  = int(action[0])
        content_idx   = int(action[1])
        timeslot_idx  = int(action[2])
        segment_idx   = int(action[3])
        budget_tier   = int(action[4])

        budget_fractions = [0.05, 0.10, 0.20, 0.30, 0.40]
        spend = budget_fractions[budget_tier] * self.budget_remaining
        spend = min(spend, self.budget_remaining)

        # ── Engagement computation ─────────────────────────────────────────────
        base_affinity     = AFFINITY[platform_idx, content_idx]
        time_mult         = TIMESLOT_MULTIPLIER[platform_idx, timeslot_idx]
        segment_mult      = SEGMENT_RECEPTIVITY[platform_idx, segment_idx]
        fatigue_penalty   = 1.0 - self.platform_fatigue[platform_idx] * 0.6
        trend_boost       = 1.0 + self.trend_momentum * 0.3
        competitor_debuff = 1.0 - self.competitor_level * 0.2
        budget_boost      = 1.0 + np.log1p(spend / PLATFORM_COST[platform_idx]) * 0.15

        noise = float(self._rng.normal(1.0, 0.12))

        engagement_rate = (
            base_affinity
            * time_mult
            * segment_mult
            * fatigue_penalty
            * trend_boost
            * competitor_debuff
            * budget_boost
            * noise
        )
        engagement_rate = max(0.0, engagement_rate)

        # Scale to realistic counts
        max_reach_platform = {0: 8000, 1: 12000, 2: 5000, 3: 6000, 4: 4000}
        reach      = engagement_rate * max_reach_platform[platform_idx]
        engagement = reach * float(self._rng.uniform(0.03, 0.12))

        # Revenue: engagement × conversion rate × avg order value
        conversion_rate = float(self._rng.uniform(0.005, 0.025))
        avg_order_value = float(self._rng.uniform(20.0, 80.0))
        revenue = engagement * conversion_rate * avg_order_value

        # ── Update state ──────────────────────────────────────────────────────
        self.budget_remaining  = max(0.0, self.budget_remaining - spend)
        self.cumulative_engage += engagement
        self.cumulative_reach  += reach
        self.total_revenue     += revenue
        self.rpe               = revenue / max(engagement, 1.0)

        # Fatigue: posting too often on the same platform increases fatigue
        self.platform_fatigue[platform_idx] = min(
            1.0, self.platform_fatigue[platform_idx] + 0.08
        )
        # All platforms recover slightly each day
        self.platform_fatigue = np.clip(self.platform_fatigue - 0.02, 0.0, 1.0)

        # Audience saturation
        self.audience_sat = min(1.0, self.audience_sat + 0.03)
        self.audience_sat = max(0.0, self.audience_sat - 0.01)

        # Sentiment: good engagement improves it, poor spend without results hurts
        sentiment_delta = (engagement_rate - 0.5) * 0.05
        self.brand_sentiment = np.clip(self.brand_sentiment + sentiment_delta, -1.0, 1.0)

        # Market dynamics: trend momentum drifts
        self.trend_momentum += float(self._rng.normal(0.0, 0.05))
        self.trend_momentum  = np.clip(self.trend_momentum, 0.0, 1.0)
        self.competitor_level += float(self._rng.normal(0.0, 0.03))
        self.competitor_level = np.clip(self.competitor_level, 0.0, 1.0)

        # Content performance history (exponential moving average)
        alpha = 0.3
        self.content_perf_hist[platform_idx] = (
            alpha * engagement_rate + (1 - alpha) * self.content_perf_hist[platform_idx]
        )

        # Consistency bonus: posting within 2 days of last post
        consistency_bonus = 0.0
        if self.last_post_day >= 0 and (self.day - self.last_post_day) <= 2:
            consistency_bonus = 0.05
        self.last_post_day = self.day

        # Advance day
        self.day += 1
        self.posts_this_week = (self.posts_this_week + 1) % 7

        # ── Reward shaping ────────────────────────────────────────────────────
        r_engage   = engagement / 500.0                        # normalized
        r_reach    = reach / 5000.0                            # normalized
        r_revenue  = revenue / 50.0                            # normalized
        r_fatigue  = -self.platform_fatigue[platform_idx] * 0.3
        r_consist  = consistency_bonus
        r_budget   = -0.1 if self.budget_remaining <= 0 else 0.0

        reward = r_engage + r_reach * 0.4 + r_revenue * 0.6 + r_fatigue + r_consist + r_budget

        # ── Terminal conditions ───────────────────────────────────────────────
        terminated = (
            self.day >= self.max_days
            or self.budget_remaining < PLATFORM_COST.min()
        )
        truncated = False

        self.action_history.append(action.tolist())
        self.reward_history.append(reward)

        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "engagement": engagement,
            "reach": reach,
            "revenue": revenue,
            "spend": spend,
            "engagement_rate": engagement_rate,
            "platform": PLATFORMS[platform_idx],
            "content_type": CONTENT_TYPES[content_idx],
            "time_slot": TIME_SLOTS[timeslot_idx],
            "segment": AUDIENCE_SEGMENTS[segment_idx],
        })

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            from environment.rendering import AfricaBrandRenderer
            if self._renderer is None:
                self._renderer = AfricaBrandRenderer(self)
            self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── Utility ───────────────────────────────────────────────────────────────

    def action_meanings(self) -> Dict:
        return {
            "platform": PLATFORMS,
            "content_type": CONTENT_TYPES,
            "time_slot": TIME_SLOTS,
            "audience_segment": AUDIENCE_SEGMENTS,
            "budget_tier": ["5%", "10%", "20%", "30%", "40%"],
        }

    def sample_random_action(self) -> np.ndarray:
        return self.action_space.sample()

    def to_json_state(self) -> Dict:
        """Serialize current state for API / frontend consumption."""
        return {
            "day": self.day,
            "budget_remaining": round(self.budget_remaining, 2),
            "cumulative_engagement": round(self.cumulative_engage, 0),
            "cumulative_reach": round(self.cumulative_reach, 0),
            "total_revenue": round(self.total_revenue, 2),
            "brand_sentiment": round(self.brand_sentiment, 3),
            "platform_fatigue": {
                PLATFORMS[i]: round(float(self.platform_fatigue[i]), 3)
                for i in range(N_PLATFORMS)
            },
            "trend_momentum": round(self.trend_momentum, 3),
            "competitor_activity": round(self.competitor_level, 3),
            "campaign_health": round(self._campaign_health(), 3),
            "observation": self._get_obs().tolist(),
        }
