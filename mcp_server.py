"""MCP server exposing MarketCanvas-Env to an LLM client.

One environment instance per server process, driven over stdio. Tools mirror
the RL interface: read the state, take an action, check the score.

This is the interactive path, not the training path. It exists so a model can
sit down at the canvas and try the task through ordinary tool calling. Under
PPO the environment is stepped in-process instead, because one stdio
subprocess per rollout does not survive contact with a few thousand of them.
"""

from mcp.server.mcpserver import MCPServer

from marketcanvas.env import MarketCanvasEnv
from marketcanvas.render import save_png
from marketcanvas.reward import compute_reward
from marketcanvas.tasks import TASKS

OUTPUT_PATH = "outputs/mcp_canvas.png"

mcp = MCPServer(
    "marketcanvas",
    instructions=(
        "A 2D design canvas. Call get_canvas_state to see what is on it and "
        "what the design brief asks for, execute_action to place or edit "
        "elements, and get_current_reward to check how the design scores. "
        "Call submit when the design is finished."
    ),
    version="0.1.0",
)

env = MarketCanvasEnv()
env.reset()


@mcp.tool()
def get_canvas_state() -> dict:
    """Current canvas: the design brief, every element with its properties,
    the spatial relations between them, and the remaining step budget."""
    return env._observation()


@mcp.tool()
def execute_action(action: dict) -> dict:
    """Apply one action and return the resulting state.

    High-level actions name an element and an intent, e.g.
    {"type": "add_element", "element_type": "text", "role": "headline",
     "x": 200, "y": 140, "width": 400, "height": 80,
     "text_color": "#FFFFFF", "content": "SUMMER SALE"}
    {"type": "move_element", "element_id": "el_1", "x": 300, "y": 200}
    {"type": "change_element_color", "element_id": "el_2", "color": "#FFD700"}

    Low-level actions mirror computer use: mouse_move, mouse_click,
    mouse_drag, keyboard_type.

    {"type": "submit"} ends the episode and scores the design.

    A rejected action still consumes a step; the reason comes back in
    action_message.
    """
    observation, reward, terminated, truncated, info = env.step(action)
    return {
        "ok": info["action_ok"],
        "message": info["action_message"],
        "done": terminated or truncated,
        "reward": reward if (terminated or truncated) else None,
        "reward_breakdown": info.get("reward_breakdown"),
        "state": observation,
    }


@mcp.tool()
def get_current_reward() -> dict:
    """Score the canvas as it stands, without ending the episode.

    The real reward is terminal, so this is a preview: it answers "what would
    this design score if I stopped now". Useful for a human or an agent
    inspecting its own work, and deliberately not part of the observation,
    since a policy that can read its own score optimizes the metric rather
    than the design.
    """
    return compute_reward(env.canvas, env.task).as_dict()


@mcp.tool()
def render_canvas() -> str:
    """Write the canvas to a PNG and return the path."""
    import os

    os.makedirs("outputs", exist_ok=True)
    return save_png(env.canvas, OUTPUT_PATH)


@mcp.tool()
def reset_canvas(task_id: str = "summer_sale_banner") -> dict:
    """Clear the canvas and start a fresh episode on the given task."""
    global env
    if task_id not in TASKS:
        return {"error": f"unknown task '{task_id}'", "available": sorted(TASKS)}
    env = MarketCanvasEnv(task_id=task_id)
    observation, _ = env.reset()
    return observation


if __name__ == "__main__":
    mcp.run(transport="stdio")