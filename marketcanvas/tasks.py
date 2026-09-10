"""Task definitions for MarketCanvas-Env.

A task pairs a natural-language prompt (what the agent sees) with a
structured target specification (what the reward function grades against).
The prompt is never parsed at runtime: keeping the grading criteria explicit
makes the reward deterministic and reproducible across episodes.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ElementRequirement:
    """A single element the finished design must contain."""

    role: str                      # semantic slot, e.g. "headline"
    type: str                      # "text" | "shape" | "image"
    color: str | None = None       # named color constraint, e.g. "yellow"
    min_width: int | None = None
    min_height: int | None = None


@dataclass(frozen=True)
class Task:
    """A design goal: the prompt shown to the agent plus its grading spec."""

    task_id: str
    prompt: str
    required_elements: list[ElementRequirement]
    min_contrast_ratio: float = 4.5      # WCAG AA for normal-size text
    canvas_width: int = 800
    canvas_height: int = 600
    max_elements: int = 8                # guards against spam-the-canvas policies


SUMMER_SALE_BANNER = Task(
    task_id="summer_sale_banner",
    prompt=(
        "Create a Summer Sale email banner with a headline, "
        "a yellow CTA button, and good contrast."
    ),
    required_elements=[
        ElementRequirement(
            role="headline",
            type="text",
            min_width=200,
            min_height=40,
        ),
        ElementRequirement(
            role="cta_button",
            type="shape",
            color="yellow",
            min_width=120,
            min_height=40,
        ),
    ],
)


TASKS: dict[str, Task] = {
    SUMMER_SALE_BANNER.task_id: SUMMER_SALE_BANNER,
}


def get_task(task_id: str) -> Task:
    """Look up a task by id. Raises KeyError with the valid ids on a miss."""
    if task_id not in TASKS:
        raise KeyError(f"Unknown task '{task_id}'. Available: {sorted(TASKS)}")
    return TASKS[task_id]