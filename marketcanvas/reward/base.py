"""The contract every reward component implements.

A scorer answers one question about a finished canvas and returns a value in
[0, 1] plus the notes behind it. Keeping the components independent is what
lets the weights be re-mixed, or a component dropped entirely, without
touching any scoring code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from marketcanvas.canvas import Canvas
from marketcanvas.tasks import Task


@dataclass
class ComponentScore:
    """One component's verdict: a normalized score and why."""

    value: float
    notes: list[str] = field(default_factory=list)


class Scorer(ABC):
    """Scores one aspect of a design."""

    name: str

    @abstractmethod
    def score(self, canvas: Canvas, task: Task) -> ComponentScore:
        """Return a score in [0, 1] for this aspect of the canvas."""
