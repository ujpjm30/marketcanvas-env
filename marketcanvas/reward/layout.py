"""Is the design laid out sanely?"""

from marketcanvas.canvas import Canvas
from marketcanvas.reward.base import ComponentScore, Scorer
from marketcanvas.tasks import Task

# Overlap below this fraction of the smaller element is touching, not colliding.
OVERLAP_TOLERANCE = 0.05

# Pixels of slack before an element stops counting as centered. Exact equality
# would be unreachable for odd-width elements.
CENTER_TOLERANCE = 10

# Sub-weights within layout. Alignment stays small on purpose: rewarding it
# heavily pushes a policy toward stacking everything on the center line, which
# the overlap penalty then has to fight.
W_OVERLAP = 0.50
W_BOUNDS = 0.25
W_ALIGNMENT = 0.25


class LayoutScorer(Scorer):
    """Overlap, bounds, and alignment, combined on their own sub-weights."""

    name = "layout"

    def score(self, canvas: Canvas, task: Task) -> ComponentScore:
        notes = []
        overlap = self._overlap(canvas, notes)
        bounds = self._bounds(canvas, notes)
        alignment = self._alignment(canvas)

        value = W_OVERLAP * overlap + W_BOUNDS * bounds + W_ALIGNMENT * alignment
        return ComponentScore(value, notes)

    def _overlap(self, canvas: Canvas, notes: list[str]) -> float:
        """Penalized by the worst partial collision on the canvas.

        Full containment is skipped: text on a backing shape is composition,
        not collision. Only partial overlap counts.
        """
        worst = 0.0
        elements = canvas.elements
        for i, a in enumerate(elements):
            for b in elements[i + 1 :]:
                area = a.overlap_area(b)
                if not area:
                    continue
                smaller = min(a.area, b.area)
                if not smaller or area >= smaller:
                    continue
                ratio = area / smaller
                if ratio > OVERLAP_TOLERANCE:
                    worst = max(worst, ratio)
                    notes.append(f"{a.element_id}/{b.element_id} overlap {ratio:.0%}")
        return 1.0 - worst

    def _bounds(self, canvas: Canvas, notes: list[str]) -> float:
        """Fraction of elements fully inside the canvas."""
        out = [e for e in canvas.elements if canvas.is_out_of_bounds(e)]
        for element in out:
            notes.append(f"{element.element_id} out of bounds")
        return 1.0 - (len(out) / len(canvas.elements))

    def _alignment(self, canvas: Canvas) -> float:
        """Fraction of elements sharing the canvas center line."""
        centered = [
            e
            for e in canvas.elements
            if abs(e.center_x - canvas.width / 2) <= CENTER_TOLERANCE
        ]
        return len(centered) / len(canvas.elements)
