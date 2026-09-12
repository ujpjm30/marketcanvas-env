"""Element primitives for the MarketCanvas design surface.

Every object on the canvas is an Element. Elements hold data only; the canvas
engine is responsible for mutating them. Colors are kept as "#RRGGBB" strings
so contrast can be computed from the values directly instead of rasterizing
the canvas first, which keeps the reward cheap enough to run in a tight loop.
"""

from dataclasses import dataclass
from enum import Enum


class ElementType(str, Enum):
    """Kinds of objects the canvas can hold."""

    TEXT = "text"
    SHAPE = "shape"
    IMAGE = "image"


@dataclass
class Element:
    """A single positioned object on the canvas."""

    element_id: str
    type: ElementType
    role: str
    x: int
    y: int
    width: int
    height: int
    z_index: int
    color: str = "#FFFFFF"
    text_color: str = "#000000"
    content: str = ""

    # Edges and center are derived rather than stored, so they stay correct
    # after a move without any bookkeeping.

    @property
    def left(self) -> int:
        return self.x

    @property
    def top(self) -> int:
        return self.y

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> int:
        return self.width * self.height

    def overlap_area(self, other: "Element") -> int:
        """Intersection area with another element, or 0 if they don't touch.

        Both boxes are axis-aligned, so the overlap is just the horizontal
        span times the vertical span. A non-positive span on either axis
        means there is no intersection at all.
        """
        dx = min(self.right, other.right) - max(self.left, other.left)
        dy = min(self.bottom, other.bottom) - max(self.top, other.top)
        if dx <= 0 or dy <= 0:
            return 0
        return dx * dy

    def to_dict(self) -> dict:
        """Flatten to the JSON shape the agent sees in its observation."""
        return {
            "id": self.element_id,
            "type": self.type.value,
            "role": self.role,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "z_index": self.z_index,
            "color": self.color,
            "text_color": self.text_color,
            "content": self.content,
        }