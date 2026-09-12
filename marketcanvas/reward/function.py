"""Combines the scorers into the scalar the environment returns."""

from dataclasses import dataclass, field

from marketcanvas.canvas import Canvas
from marketcanvas.reward.base import Scorer
from marketcanvas.reward.constraints import ConstraintScorer
from marketcanvas.reward.contrast import ContrastScorer
from marketcanvas.reward.layout import LayoutScorer
from marketcanvas.tasks import Task

# Constraints dominate: a banner missing its CTA is a failed banner no matter
# how clean the layout is.
DEFAULT_WEIGHTS: list[tuple[Scorer, float]] = [
    (ConstraintScorer(), 0.6),
    (ContrastScorer(), 0.2),
    (LayoutScorer(), 0.2),
]

# The floor, not zero. At zero, doing nothing beats trying and failing, and a
# policy can settle into never acting.
EMPTY_CANVAS_REWARD = -1.0


@dataclass
class RewardBreakdown:
    """Per-component scores and the reasons behind them.
    A single number isn't debuggable. When a rollout scores 0.4, the useful
    question is which component lost the points.
    """

    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __getattr__(self, name: str) -> float:
        """Read a component by name, e.g. breakdown.contrast."""
        try:
            return self.__dict__["components"][name]
        except KeyError:
            raise AttributeError(name) from None

    def as_dict(self) -> dict:
        return {
            "total": round(self.total, 3),
            **{k: round(v, 3) for k, v in self.components.items()},
            "notes": self.notes,
        }


class RewardFunction:
    """A weighted mix of scorers, rescaled to [-1.0, 1.0].
    The scorer list is a constructor argument so an ablation is a different
    RewardFunction, not a different reward.py.
    """

    def __init__(self, weights: list[tuple[Scorer, float]] | None = None):
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS

    def __call__(self, canvas: Canvas, task: Task) -> RewardBreakdown:
        if not canvas.elements:
            return RewardBreakdown(
                total=EMPTY_CANVAS_REWARD, notes=["empty canvas"]
            )

        components = {}
        notes = []
        quality = 0.0
        for scorer, weight in self.weights:
            result = scorer.score(canvas, task)
            components[scorer.name] = result.value
            notes.extend(result.notes)
            quality += weight * result.value

        # Components are in [0, 1]; rescale so a canvas satisfying nothing
        # lands at the same floor as an empty one.
        total = 2.0 * quality - 1.0
        return RewardBreakdown(total=total, components=components, notes=notes)


_default = RewardFunction()


def compute_reward(canvas: Canvas, task: Task) -> RewardBreakdown:
    """Score a finished canvas with the default weights."""
    return _default(canvas, task)
