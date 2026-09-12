"""Policies: the thing that decides what to do next.

The environment defines what is possible; a policy decides what happens. Both
have to exist before anything can be called a rollout, and a scripted list of
actions is neither — it is an answer key, not a decision maker.

Every policy here reads the observation and nothing else. No policy touches
the Canvas, the Task, or the reward function, so any of them can be swapped
for a trained network later without the environment noticing.

Three are provided, and the gap between them is the point:

    RandomPolicy     what the task scores with no decision at all
    HeuristicPolicy  what it scores with hand-written rules
    LLMPolicy        what a language model does with the same observation

Random sets the floor, heuristic shows the reward is reachable, and the
spread between them is the room a learned policy has to work in. A reward
where random already scores well is too easy; one where the heuristic still
fails is unreachable, and PPO would never find a gradient in it.
"""

import json
import os
import random
from abc import ABC, abstractmethod

# Placement constants for the heuristic. Fractions of the canvas rather than
# pixels, so the same rules hold if the canvas is resized.
HEADLINE_BAND = (0.20, 0.40)
CTA_BAND = (0.58, 0.72)
HEADLINE_WIDTH = 0.50
CTA_WIDTH = 0.22

DARK = "#1A237E"
LIGHT = "#FFFFFF"
NAMED_HEX = {
    "yellow": "#FFD700",
    "red": "#DC2626",
    "blue": "#2563EB",
    "green": "#16A34A",
    "orange": "#F97316",
    "purple": "#9333EA",
}


class Policy(ABC):
    """Maps an observation to the next action.
    Returning None ends the episode, which is how a policy says "done"
    without the runner having to guess.
    """

    name: str

    def reset(self) -> None:
        """Clear any per-episode state. Stateless policies need not override."""

    @abstractmethod
    def act(self, observation: dict) -> dict | None:
        """Choose one action from the observation alone."""


class RandomPolicy(Policy):
    """Samples actions blindly, ignoring the observation entirely.
    A floor, not a contender. Its value is twofold. It proves the environment
    survives arbitrary input without crashing, and it measures what the task
    is worth to an agent that has learned nothing. 
    """

    name = "random"

    def __init__(self, seed: int = 0, steps: int = 12):
        self.rng = random.Random(seed)
        self.steps = steps
        self._taken = 0

    def reset(self) -> None:
        self._taken = 0

    def act(self, observation: dict) -> dict | None:
        if self._taken >= self.steps:
            return {"type": "submit"}
        self._taken += 1

        width = observation["canvas"]["width"]
        height = observation["canvas"]["height"]
        ids = [e["id"] for e in observation["elements"]]

        choices = ["add_element"]
        if ids:
            choices += ["move_element", "change_element_color"]

        kind = self.rng.choice(choices)
        if kind == "add_element":
            return {
                "type": "add_element",
                "element_type": self.rng.choice(["text", "shape", "image"]),
                "role": self.rng.choice(["headline", "cta_button", "decoration"]),
                "x": self.rng.randrange(0, width),
                "y": self.rng.randrange(0, height),
                "width": self.rng.randrange(20, 400),
                "height": self.rng.randrange(20, 200),
                "color": self._color(),
                "text_color": self._color(),
                "content": self.rng.choice(["", "SALE", "BUY"]),
            }
        if kind == "move_element":
            return {
                "type": "move_element",
                "element_id": self.rng.choice(ids),
                "x": self.rng.randrange(0, width),
                "y": self.rng.randrange(0, height),
            }
        return {
            "type": "change_element_color",
            "element_id": self.rng.choice(ids),
            "color": self._color(),
        }

    def _color(self) -> str:
        return "#{:06X}".format(self.rng.randrange(0x1000000))


class HeuristicPolicy(Policy):
    """Hand-written rules over the observation. No model, no training.
    Reads the brief the same way an agent would, by looking at the prompt
    text, then repairs whatever the canvas is missing: lay a background, add
    an absent element, recolor one whose color is wrong. One repair per step,
    so every decision is made against a fresh observation rather than a plan
    formed once at the start.
    """

    name = "heuristic"

    def act(self, observation: dict) -> dict | None:
        canvas = observation["canvas"]
        elements = observation["elements"]
        wanted = self._parse(observation["prompt"])
        by_role = {e["role"]: e for e in elements}

        if "background" not in by_role:
            return self._add("shape", "background", 0, 0,
                             canvas["width"], canvas["height"], DARK)

        for role, spec in wanted.items():
            if role not in by_role:
                return self._place(role, spec, canvas)

        # Everything is present, so fix colors that came out wrong.
        for role, spec in wanted.items():
            target = spec.get("color")
            if target and by_role[role]["color"] != NAMED_HEX[target]:
                return {
                    "type": "change_element_color",
                    "element_id": by_role[role]["id"],
                    "color": NAMED_HEX[target],
                }

        return {"type": "submit"}

    def _parse(self, prompt: str) -> dict:
        """Turn the brief into roles to satisfy. Keyword matching, nothing more."""
        text = prompt.lower()
        wanted: dict[str, dict] = {}
        if "headline" in text:
            wanted["headline"] = {"type": "text"}
        if "cta" in text or "button" in text:
            spec: dict = {"type": "shape"}
            for name in NAMED_HEX:
                if name in text:
                    spec["color"] = name
                    break
            wanted["cta_button"] = spec
        return wanted

    def _place(self, role: str, spec: dict, canvas: dict) -> dict:
        """Center the element in the band its role belongs to."""
        cw, ch = canvas["width"], canvas["height"]
        if role == "headline":
            top, bottom = HEADLINE_BAND
            width = int(cw * HEADLINE_WIDTH)
            text_color, fill = LIGHT, DARK
            content = "SUMMER SALE"
        else:
            top, bottom = CTA_BAND
            width = int(cw * CTA_WIDTH)
            fill = NAMED_HEX.get(spec.get("color", ""), LIGHT)
            text_color = DARK
            content = "SHOP NOW"

        height = int(ch * (bottom - top))
        return self._add(
            spec["type"], role,
            (cw - width) // 2, int(ch * top),
            width, height, fill, text_color, content,
        )

    def _add(self, element_type, role, x, y, width, height,
             color, text_color=LIGHT, content=""):
        return {
            "type": "add_element",
            "element_type": element_type,
            "role": role,
            "x": x, "y": y, "width": width, "height": height,
            "color": color, "text_color": text_color, "content": content,
        }


class LLMPolicy(Policy):
    """A language model choosing actions from the same observation.
    Optional: needs ANTHROPIC_API_KEY and the anthropic package. 
    """

    name = "llm"

    SYSTEM = (
        "You are designing on a 2D canvas. Each turn you receive the canvas "
        "state as JSON and reply with exactly one action as JSON, nothing "
        "else. Available actions: add_element (element_type, role, x, y, "
        "width, height, color, text_color, content), move_element "
        "(element_id, x, y), resize_element (element_id, width, height), "
        "change_element_color (element_id, color), change_text_color "
        "(element_id, text_color), submit. Colors are #RRGGBB. Use the role "
        "field to say what each element is for, matching the words in the "
        "brief. Reply with submit when the design satisfies the brief."
    )

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 512):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("pip install anthropic to use LLMPolicy") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def act(self, observation: dict) -> dict | None:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": json.dumps(observation)}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return self._parse_action(text)

    @staticmethod
    def _parse_action(text: str) -> dict:
        """Pull one JSON object out of the reply.
        Models wrap JSON in prose or fences often enough that failing on it
        would measure formatting compliance rather than design ability. A
        malformed reply becomes a noop, which costs a step like any other
        rejected action.
        """
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1:
            return {"type": "noop"}
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {"type": "noop"}


def run_episode(env, policy: Policy, verbose: bool = False) -> dict:
    """Drive one episode with a policy and report what happened."""
    policy.reset()
    observation, _ = env.reset()
    reward, steps, rejected = 0.0, 0, 0

    while True:
        action = policy.act(observation)
        if action is None:
            action = {"type": "submit"}

        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        rejected += int(not info["action_ok"])

        if verbose:
            status = "ok" if info["action_ok"] else f'FAILED ({info["action_message"]})'
            print(f'  {steps:2}. {action.get("type", "?"):22} {status}')

        if terminated or truncated:
            return {
                "policy": policy.name,
                "reward": reward,
                "steps": steps,
                "rejected": rejected,
                "elements": len(observation["elements"]),
                "breakdown": info.get("reward_breakdown", {}),
            }
