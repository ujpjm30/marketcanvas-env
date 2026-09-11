"""Policy tests.

These check the claim the write-up rests on: that the task is reachable
without a learned model, and that a policy which decides nothing scores far
worse than one that reads the observation. If those two collapse into each
other, the reward is not measuring design.
"""

import pytest

from marketcanvas.env import MarketCanvasEnv
from marketcanvas.policy import HeuristicPolicy, Policy, RandomPolicy, run_episode


def episode(policy: Policy, task: str = "summer_sale_banner") -> dict:
    return run_episode(MarketCanvasEnv(task_id=task), policy)


# The baseline gap

def test_heuristic_solves_the_task():
    """The reward has to be reachable, or PPO has nothing to climb toward."""
    assert episode(HeuristicPolicy())["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_scores_poorly(seed):
    """And it has to be unreachable by accident, or it measures nothing."""
    assert episode(RandomPolicy(seed=seed))["reward"] < 0.5


def test_the_gap_is_wide():
    assert episode(HeuristicPolicy())["reward"] - episode(RandomPolicy())["reward"] > 1.0


# Policy contract

def test_policies_only_read_the_observation():
    """A policy handed a plain dict must still act, so nothing it needs can
    be hiding on the env, the canvas, or the task."""
    observation = MarketCanvasEnv().reset()[0]
    assert HeuristicPolicy().act(dict(observation))["type"] == "add_element"


def test_random_survives_its_own_output():
    """Arbitrary actions must fail cleanly rather than crash the episode."""
    result = episode(RandomPolicy(seed=7))
    assert result["steps"] > 0


def test_heuristic_is_deterministic():
    assert episode(HeuristicPolicy())["reward"] == episode(HeuristicPolicy())["reward"]


def test_heuristic_reacts_to_state_rather_than_replaying_a_plan():
    """Given a canvas that already has a background, it moves on instead of
    laying a second one."""
    env = MarketCanvasEnv()
    env.reset()
    env.step({"type": "add_element", "element_type": "shape", "role": "background",
              "x": 0, "y": 0, "width": 800, "height": 600, "color": "#1A237E"})

    action = HeuristicPolicy().act(env._observation())
    assert action["role"] != "background"


def test_heuristic_repairs_a_wrong_color():
    """Placement is not the only decision; an existing element can be wrong."""
    env = MarketCanvasEnv()
    env.reset()
    for action in [
        {"type": "add_element", "element_type": "shape", "role": "background",
         "x": 0, "y": 0, "width": 800, "height": 600, "color": "#1A237E"},
        {"type": "add_element", "element_type": "text", "role": "headline",
         "x": 200, "y": 140, "width": 400, "height": 80,
         "text_color": "#FFFFFF", "content": "SUMMER SALE"},
        {"type": "add_element", "element_type": "shape", "role": "cta_button",
         "x": 320, "y": 360, "width": 176, "height": 84, "color": "#9E9E9E"},
    ]:
        env.step(action)

    repair = HeuristicPolicy().act(env._observation())
    assert repair["type"] == "change_element_color"
    assert repair["color"] == "#FFD700"
