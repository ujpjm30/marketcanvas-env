"""Runs one scripted episode and prints the state and reward.

Uses a fixed action sequence rather than random actions: a random policy on
this task produces an empty or nonsensical canvas almost every time, which
shows nothing about whether the reward discriminates.
"""

import json
from pathlib import Path

from marketcanvas.env import MarketCanvasEnv
from marketcanvas.render import save_png

OUTPUT_DIR = Path("outputs")

SCRIPTED_EPISODE = [
    {
        "type": "add_element",
        "element_type": "shape",
        "role": "background",
        "x": 0, "y": 0, "width": 800, "height": 600,
        "color": "#1A237E",
    },
    {
        "type": "add_element",
        "element_type": "text",
        "role": "headline",
        "x": 200, "y": 140, "width": 400, "height": 80,
        "color": "#1A237E", "text_color": "#FFFFFF",
        "content": "SUMMER SALE",
    },
    {
        "type": "add_element",
        "element_type": "shape",
        "role": "cta_button",
        "x": 320, "y": 360, "width": 160, "height": 56,
        "color": "#FFD700", "text_color": "#1A237E",
        "content": "SHOP NOW",
    },
    {"type": "submit"},
]


def main() -> None:
    env = MarketCanvasEnv(task_id="summer_sale_banner")
    obs, info = env.reset(seed=0)

    print(f"Task: {obs['prompt']}\n")

    for i, action in enumerate(SCRIPTED_EPISODE, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        status = "ok" if info["action_ok"] else f"FAILED ({info['action_message']})"
        print(f"step {i}: {action['type']:16} {status}")
        if terminated or truncated:
            break

    print("\nFinal elements:")
    for element in obs["elements"]:
        print(
            f"  {element['id']:6} {element['type']:6} {element['role']:12}"
            f" ({element['x']}, {element['y']}) {element['width']}x{element['height']}"
            f" {element['color']}"
        )

    print(f"\nReward: {reward:.3f}")
    print(json.dumps(info["reward_breakdown"], indent=2))

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = save_png(env.canvas, str(OUTPUT_DIR / "final.png"))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()