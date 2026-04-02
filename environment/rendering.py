"""
AfricaBrand Environment Renderer
=================================
Pygame-based visualization of the social media content marketing simulation.
Displays platform allocation, engagement metrics, budget burn, fatigue bars,
and a live scrolling reward graph.
"""

import pygame
import numpy as np
import sys
import os

# ─── Palette ──────────────────────────────────────────────────────────────────

BG_DARK      = (12,  17,  26)
BG_PANEL     = (20,  28,  42)
ACCENT_GOLD  = (255, 196,  0)
ACCENT_GREEN = (52,  211, 153)
ACCENT_RED   = (248,  91,  80)
ACCENT_BLUE  = (96,  165, 250)
ACCENT_PURP  = (167, 139, 250)
TEXT_MAIN    = (230, 230, 240)
TEXT_MUTED   = (120, 130, 150)
TEXT_HEADER  = (255, 255, 255)

PLATFORM_COLORS = [
    (225,  48, 108),    # Instagram – pink
    (0,   0,    0),     # TikTok – will draw with two tones
    (29,  161, 242),    # Twitter/X – blue
    (24,  119, 242),    # Facebook – blue
    (255,   0,   0),    # YouTube – red
]

PLATFORM_COLORS_SAFE = [
    (225,  48, 108),
    (105, 201, 208),
    ( 29, 161, 242),
    ( 24, 119, 242),
    (255,  80,  80),
]

CONTENT_COLORS = [
    ACCENT_GOLD,
    ACCENT_GREEN,
    ACCENT_BLUE,
    ACCENT_PURP,
    (251, 146,  60),
    (52,  211, 153),
]

WIDTH, HEIGHT = 1280, 720


class AfricaBrandRenderer:
    """Manages the Pygame window and renders all environment state."""

    def __init__(self, env):
        self.env = env
        pygame.init()
        pygame.display.set_caption("AfricaBrand – Content Marketing RL Environment")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock  = pygame.time.Clock()

        # Fonts
        font_path = None  # use system default
        self.font_h1   = pygame.font.SysFont("dejavusans", 22, bold=True)
        self.font_h2   = pygame.font.SysFont("dejavusans", 16, bold=True)
        self.font_body = pygame.font.SysFont("dejavusans", 13)
        self.font_sm   = pygame.font.SysFont("dejavusans", 11)

        self.reward_history_render: list = []
        self.engage_history_render: list = []

        self._last_info: dict = {}

    def update_info(self, info: dict):
        self._last_info = info

    def _draw_rect(self, surface, color, rect, radius=6, alpha=255):
        if alpha < 255:
            s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=radius)
            surface.blit(s, (rect[0], rect[1]))
        else:
            pygame.draw.rect(surface, color, rect, border_radius=radius)

    def _draw_panel(self, x, y, w, h, title=None):
        self._draw_rect(self.screen, BG_PANEL, (x, y, w, h), radius=8)
        pygame.draw.rect(self.screen, (40, 55, 80), (x, y, w, h), width=1, border_radius=8)
        if title:
            label = self.font_h2.render(title, True, TEXT_MUTED)
            self.screen.blit(label, (x + 12, y + 8))

    def _draw_bar(self, x, y, w, h, value, color, bg=(35, 45, 65), label=None):
        pygame.draw.rect(self.screen, bg, (x, y, w, h), border_radius=4)
        fill_w = int(w * np.clip(value, 0.0, 1.0))
        if fill_w > 0:
            pygame.draw.rect(self.screen, color, (x, y, fill_w, h), border_radius=4)
        if label:
            lbl = self.font_sm.render(label, True, TEXT_MUTED)
            self.screen.blit(lbl, (x, y - 14))

    def _draw_sparkline(self, x, y, w, h, data, color):
        if len(data) < 2:
            return
        pts = []
        mn, mx = min(data), max(data)
        span = mx - mn if mx != mn else 1.0
        for i, v in enumerate(data[-w:]):
            px = x + int(i * w / min(len(data), w))
            py = y + h - int((v - mn) / span * h)
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, color, False, pts, 2)

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        env = self.env
        self.screen.fill(BG_DARK)

        # ── Header ────────────────────────────────────────────────────────────
        title = self.font_h1.render("AfricaBrand Content Marketing — RL Simulation", True, TEXT_HEADER)
        self.screen.blit(title, (20, 14))
        day_label = self.font_h2.render(
            f"Day {env.day} / {env.max_days}   |   Budget: ${env.budget_remaining:.0f}  remaining",
            True, ACCENT_GOLD
        )
        self.screen.blit(day_label, (20, 40))

        # ── Left column: KPI cards ─────────────────────────────────────────────
        lx, ly = 20, 72
        kpis = [
            ("Engagement",  f"{env.cumulative_engage:,.0f}",   ACCENT_GREEN),
            ("Reach",       f"{env.cumulative_reach:,.0f}",    ACCENT_BLUE),
            ("Revenue",     f"${env.total_revenue:,.2f}",      ACCENT_GOLD),
            ("Sentiment",   f"{env.brand_sentiment:+.2f}",     ACCENT_PURP),
        ]
        for i, (name, val, col) in enumerate(kpis):
            cx = lx + i * 160
            self._draw_panel(cx, ly, 150, 60)
            n_lbl = self.font_sm.render(name, True, TEXT_MUTED)
            v_lbl = self.font_h2.render(val, True, col)
            self.screen.blit(n_lbl, (cx + 10, ly + 8))
            self.screen.blit(v_lbl, (cx + 10, ly + 28))

        # ── Platform fatigue panel ─────────────────────────────────────────────
        fx, fy = 20, 150
        self._draw_panel(fx, fy, 320, 160, title="Platform Fatigue")
        from environment.custom_env import PLATFORMS
        n_platforms = len(PLATFORMS)
        for i, pname in enumerate(PLATFORMS):
            bx, by = fx + 12, fy + 30 + i * 24
            self._draw_bar(bx, by + 10, 220, 12, env.platform_fatigue[i],
                           PLATFORM_COLORS_SAFE[i], label=pname)

        # ── Last action info ───────────────────────────────────────────────────
        ax, ay = 350, 72
        self._draw_panel(ax, ay, 340, 240, title="Last Content Decision")
        if self._last_info:
            fields = [
                ("Platform",     self._last_info.get("platform", "—")),
                ("Content Type", self._last_info.get("content_type", "—")),
                ("Time Slot",    self._last_info.get("time_slot", "—")),
                ("Segment",      self._last_info.get("segment", "—")),
                ("Spend",        f"${self._last_info.get('spend', 0):.2f}"),
                ("Engagement",   f"{self._last_info.get('engagement', 0):,.0f}"),
                ("Reach",        f"{self._last_info.get('reach', 0):,.0f}"),
                ("Revenue",      f"${self._last_info.get('revenue', 0):.2f}"),
            ]
            for j, (k, v) in enumerate(fields):
                k_lbl = self.font_body.render(k + ":", True, TEXT_MUTED)
                v_lbl = self.font_body.render(str(v), True, TEXT_MAIN)
                self.screen.blit(k_lbl, (ax + 12, ay + 28 + j * 24))
                self.screen.blit(v_lbl, (ax + 150, ay + 28 + j * 24))

        # ── Budget burn bar ────────────────────────────────────────────────────
        bx, by = 700, 72
        self._draw_panel(bx, by, 560, 60, title="Budget Consumed")
        spent_ratio = 1.0 - env.budget_remaining / env.max_budget
        self._draw_bar(bx + 12, by + 32, 530, 16, spent_ratio, ACCENT_GOLD)
        pct_lbl = self.font_sm.render(f"{spent_ratio*100:.1f}% spent", True, TEXT_MUTED)
        self.screen.blit(pct_lbl, (bx + 12, by + 52))

        # ── Campaign health gauge ──────────────────────────────────────────────
        hx, hy = 700, 145
        health = env._campaign_health()
        color = ACCENT_GREEN if health > 0.65 else (ACCENT_GOLD if health > 0.35 else ACCENT_RED)
        self._draw_panel(hx, hy, 270, 60, title="Campaign Health")
        self._draw_bar(hx + 12, hy + 32, 240, 16, health, color)
        h_lbl = self.font_sm.render(f"{health*100:.0f}%", True, TEXT_MUTED)
        self.screen.blit(h_lbl, (hx + 12, hy + 52))

        # ── Trend / competitor ─────────────────────────────────────────────────
        tx, ty = 980, 145
        self._draw_panel(tx, ty, 280, 60, title="Market Signals")
        self._draw_bar(tx + 12, ty + 22, 110, 10, env.trend_momentum, ACCENT_GREEN, label="Trend")
        self._draw_bar(tx + 155, ty + 22, 110, 10, env.competitor_level, ACCENT_RED, label="Competitor")

        # ── Reward sparkline ──────────────────────────────────────────────────
        if env.reward_history:
            self.reward_history_render = env.reward_history[-120:]
        rx, ry = 20, 320
        self._draw_panel(rx, ry, 600, 180, title="Reward History")
        if len(self.reward_history_render) > 1:
            self._draw_sparkline(rx + 12, ry + 28, 570, 140,
                                 self.reward_history_render, ACCENT_GREEN)
            mn = min(self.reward_history_render)
            mx = max(self.reward_history_render)
            mn_lbl = self.font_sm.render(f"{mn:.2f}", True, TEXT_MUTED)
            mx_lbl = self.font_sm.render(f"{mx:.2f}", True, TEXT_MUTED)
            self.screen.blit(mn_lbl, (rx + 12, ry + 158))
            self.screen.blit(mx_lbl, (rx + 12, ry + 28))

        # ── Engagement history ─────────────────────────────────────────────────
        ex, ey = 640, 320
        self._draw_panel(ex, ey, 620, 180, title="Engagement per Step")
        if len(env.reward_history) > 1:
            # reconstruct from cumulative via diff
            rews = env.reward_history[-120:]
            self._draw_sparkline(ex + 12, ey + 28, 590, 140, rews, ACCENT_BLUE)

        # ── Content type distribution (current run) ───────────────────────────
        dx, dy = 20, 515
        self._draw_panel(dx, dy, 300, 185, title="Action Distribution")
        if env.action_history:
            actions = np.array(env.action_history)
            from environment.custom_env import CONTENT_TYPES
            counts = np.bincount(actions[:, 1].astype(int), minlength=len(CONTENT_TYPES))
            total  = counts.sum() or 1
            bar_h  = 22
            for i, (ct, cnt) in enumerate(zip(CONTENT_TYPES, counts)):
                bx2 = dx + 12
                by2 = dy + 32 + i * (bar_h + 4)
                self._draw_bar(bx2, by2, 200, bar_h - 4, cnt / total, CONTENT_COLORS[i])
                lbl = self.font_sm.render(f"{ct[:14]}: {cnt}", True, TEXT_MUTED)
                self.screen.blit(lbl, (bx2, by2 - 12))

        # ── Platform distribution pie-style bars ───────────────────────────────
        ppx, ppy = 330, 515
        self._draw_panel(ppx, ppy, 300, 185, title="Platform Distribution")
        if env.action_history:
            actions   = np.array(env.action_history)
            p_counts  = np.bincount(actions[:, 0].astype(int), minlength=n_platforms)
            p_total   = p_counts.sum() or 1
            for i, (pn, pc) in enumerate(zip(PLATFORMS, p_counts)):
                bx3 = ppx + 12
                by3 = ppy + 32 + i * 28
                self._draw_bar(bx3, by3, 200, 14, pc / p_total, PLATFORM_COLORS_SAFE[i])
                pl = self.font_sm.render(f"{pn}: {pc}", True, TEXT_MUTED)
                self.screen.blit(pl, (bx3, by3 - 12))

        # ── Footer ────────────────────────────────────────────────────────────
        foot = self.font_sm.render(
            "AfricaBrand RL Env  |  obs_dim=20  |  action_space=MultiDiscrete([5,6,4,3,5])",
            True, TEXT_MUTED
        )
        self.screen.blit(foot, (20, HEIGHT - 18))

        pygame.display.flip()
        self.clock.tick(self.env.metadata["render_fps"])

    def close(self):
        pygame.quit()


# ─── Standalone static demo ───────────────────────────────────────────────────

def run_static_demo(steps: int = 200):
    """
    Run the environment with random actions and render the visualization.
    No model is involved — this demonstrates the environment GUI only.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from environment.custom_env import AfricaBrandEnv

    env      = AfricaBrandEnv(render_mode="human")
    obs, _   = env.reset(seed=42)
    renderer = AfricaBrandRenderer(env)

    for step in range(steps):
        action          = env.sample_random_action()
        obs, reward, terminated, truncated, info = env.step(action)
        renderer.update_info(info)
        renderer.render()

        if terminated or truncated:
            obs, _ = env.reset()

    renderer.close()


if __name__ == "__main__":
    run_static_demo()
