# MarketCanvas-Env

A deterministic 2D design canvas for training agents to build marketing
assets from a natural language brief. It exposes a Gymnasium-style RL
interface and runs as an MCP server, so the same environment can be stepped
programmatically or driven by an LLM client through tool calling.

Given a brief like "Create a Summer Sale email banner with a headline, a
yellow CTA button, and good contrast", an agent places elements on an
800x600 canvas and receives a single score between -1.0 and 1.0 at the end
of the episode.

See [WRITEUP.md](WRITEUP.md) for the design decisions behind the action
space, the reward function and its loopholes, and what would break at 10,000
parallel rollouts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer.

## Running it

```bash
python demo.py                   # compare the baseline policies
python demo.py --verbose         # print every action as it is taken
python demo.py --llm             # add the LLM policy (needs an API key)
pytest                           # 45 tests
```

`demo.py` runs each policy against the same task and prints a comparison.

```
policy        reward  steps  rejected  elements
-----------------------------------------------
random         -0.78     13         0         1
heuristic       1.00      4         0         3
```

The gap is the point. A policy that ignores the observation cannot reach the
reward by luck, and one that reads it and repairs what is missing can reach
the ceiling, so there is room in between for a learned policy to work in.
Renders are written to `outputs/`.

The LLM policy is optional and needs `pip install anthropic` plus
`ANTHROPIC_API_KEY` in the environment. Without either, it is skipped and
the other two still run.

## Using it as an environment

```python
from marketcanvas.env import MarketCanvasEnv

env = MarketCanvasEnv(task_id="summer_sale_banner")
observation, info = env.reset()

observation, reward, terminated, truncated, info = env.step({
    "type": "add_element", "element_type": "text", "role": "headline",
    "x": 200, "y": 140, "width": 400, "height": 80,
    "text_color": "#FFFFFF", "content": "SUMMER SALE",
})
```

A policy is anything with an `act(observation) -> action` method. Three are
included in `marketcanvas/policy.py`, and a trained network would go in the
same place.

## Connecting an LLM

```bash
python mcp_server.py
```

Tools exposed are `get_canvas_state`, `execute_action`,
`get_current_reward`, `render_canvas` and `reset_canvas`.

For Claude Desktop, add this to `claude_desktop_config.json` using absolute
paths, then restart the app.

```json
{
  "mcpServers": {
    "marketcanvas": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

MCP is the interactive path. Training should step the environment in process
instead, for reasons covered in the writeup.

## Layout

```
marketcanvas/
├── tasks.py         the brief and the spec it is graded against
├── elements.py      one positioned object and its geometry
├── canvas.py        the surface, and the only place state is mutated
├── observation.py   what the agent sees, with spatial relations precomputed
├── actions.py       high-level and low-level action layers
├── policy.py        random, rule-based and LLM policies
├── render.py        rasterization, kept off the step path
├── env.py           Gymnasium wrapper
└── reward/
    ├── constraints.py   is what the brief asked for present
    ├── contrast.py      can the text be read (WCAG)
    ├── layout.py        overlap, bounds, alignment
    ├── color.py         hex parsing and color math
    └── function.py      weights, and the scalar the environment returns
```

Each reward component is a separate scorer behind a shared interface, so
changing the weights or dropping a component is a configuration change
rather than an edit to the scoring code.

## Notes

Rendering is deliberately absent from the step path. Nothing in the
observation or the reward reads pixels, so `render()` runs only when an
image is actually wanted.

The canvas assigns element ids and z-order itself rather than letting the
caller choose, which is what keeps an action sequence reproducible.