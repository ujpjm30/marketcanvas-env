"""Gymnasium wrapper around the canvas, actions, observation, and reward.

The reward is terminal, the design quality is a property of the finished
artifact, and scoring partial layouts every step would reward churn. Each
step returns 0.0 until the episode ends.
"""

from typing import Any

import gymnasium as gym

from marketcanvas.actions import ActionHandler
from marketcanvas.canvas import Canvas
from marketcanvas.observation import build_observation
from marketcanvas.render import render_array
from marketcanvas.reward import compute_reward
from marketcanvas.tasks import Task, get_task

DEFAULT_MAX_STEPS = 20


class MarketCanvasEnv(gym.Env):
    """A design task as an episodic MDP.
    Spaces are intentionally loose: actions are dicts and observations are
    JSON, because the policy here is a language model consuming tool calls,
    not a network over fixed-width tensors.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_id: str = "summer_sale_banner",
        max_steps: int = DEFAULT_MAX_STEPS,
        allow_low_level: bool = True,
    ):
        super().__init__()
        self.task: Task = get_task(task_id)
        self.max_steps = max_steps
        self.allow_low_level = allow_low_level

        self.canvas = Canvas(
            width=self.task.canvas_width,
            height=self.task.canvas_height,
            max_elements=self.task.max_elements,
        )
        self.handler = ActionHandler(self.canvas, allow_low_level=allow_low_level)
        self.step_count = 0
        self.last_result = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        """Clear the canvas and return the opening observation."""
        super().reset(seed=seed)
        self.canvas.clear()
        self.handler = ActionHandler(
            self.canvas, allow_low_level=self.allow_low_level
        )
        self.step_count = 0
        self.last_result = None
        return self._observation(), {"task_id": self.task.task_id}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        """Apply one action and report the result.
        A rejected action still consumes a step. Free retries would let a
        policy brute-force the action schema instead of learning it.
        """
        result = self.handler.apply(action)
        self.last_result = result
        self.step_count += 1

        truncated = self.step_count >= self.max_steps
        terminated = bool(action.get("type") == "submit")
        done = terminated or truncated

        reward = 0.0
        info: dict[str, Any] = {
            "action_ok": result.ok,
            "action_message": result.message,
        }

        if done:
            breakdown = compute_reward(self.canvas, self.task)
            reward = breakdown.total
            info["reward_breakdown"] = breakdown.as_dict()

        return self._observation(), reward, terminated, truncated, info

    def render(self):
        """Pixels, on request only."""
        return render_array(self.canvas)

    def _observation(self) -> dict:
        return build_observation(
            self.canvas, self.task, self.step_count, self.max_steps
        )