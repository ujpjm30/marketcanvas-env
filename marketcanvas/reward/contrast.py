"""Can the text actually be read?"""

from marketcanvas.canvas import Canvas
from marketcanvas.elements import Element, ElementType
from marketcanvas.reward.base import ComponentScore, Scorer
from marketcanvas.reward.color import contrast_ratio
from marketcanvas.tasks import Task

# Ratios at or below this score nothing; the task's target scores full marks.
CONTRAST_FLOOR = 1.0

# An element counts as the background only if it covers most of the text.
BACKING_COVERAGE = 0.5


class ContrastScorer(Scorer):
    """Readability of the least readable element carrying text.

    The minimum, not the mean. Accessibility isn't an average: one headline
    invisible against its background is a broken banner even if everything
    else is fine, and a mean would let an agent dilute one unreadable element
    by adding several legible ones. The cost is a sparser signal, since
    improving anything but the worst element moves nothing.

    Labelled shapes count too: a CTA whose text disappears into its own fill
    is unreadable whatever element type it happens to be.

    Ramped, not thresholded, so 4.4 and 4.6 aren't worlds apart and a policy
    gets signal for improving 1.5 toward 4.0.
    """

    name = "contrast"

    def score(self, canvas: Canvas, task: Task) -> ComponentScore:
        labelled = [e for e in canvas.elements if e.content]
        if not labelled:
            return ComponentScore(0.0, ["no text to evaluate"])

        target = task.min_contrast_ratio
        scores = []
        notes = []
        for element in labelled:
            background = self.background_of(element, canvas)
            ratio = contrast_ratio(element.text_color, background)
            normalized = (ratio - CONTRAST_FLOOR) / (target - CONTRAST_FLOOR)
            scores.append(max(0.0, min(1.0, normalized)))
            notes.append(f"{element.role} contrast {ratio:.1f}:1 on {background}")

        return ComponentScore(min(scores), notes)

    def background_of(self, element: Element, canvas: Canvas) -> str:
        """What an element's text actually sits on.

        Text sits on whatever is behind it; a labelled shape sits on its own
        fill. Falls back to the canvas when nothing covers it.
        """
        if element.type is not ElementType.TEXT:
            return element.color

        behind = [
            e
            for e in canvas.elements
            if e.z_index < element.z_index
            and element.area
            and e.overlap_area(element) / element.area > BACKING_COVERAGE
        ]
        if not behind:
            return canvas.background_color
        return max(behind, key=lambda e: e.z_index).color
