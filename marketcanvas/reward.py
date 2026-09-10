"""Terminal reward for a finished design.

Three components: does the design contain what was asked for, is it laid
out sanely, is the text readable.

All of it reads the semantic state. Contrast comes from the stored hex
values, not sampled pixels, so scoring never requires a render.
"""

from dataclasses import dataclass, field

from marketcanvas.canvas import Canvas
from marketcanvas.elements import Element, ElementType
from marketcanvas.tasks import ElementRequirement, Task

# Constraints dominate: a banner missing its CTA is a failed banner no
# matter how clean the layout is.
W_CONSTRAINTS = 0.6
W_CONTRAST = 0.2
W_LAYOUT = 0.2

# Alignment stays small. Rewarding it heavily pushes a policy toward
# stacking everything on the center line.
W_OVERLAP = 0.10
W_BOUNDS = 0.05
W_ALIGNMENT = 0.05

# The floor, not zero. At zero, doing nothing beats trying and failing.
EMPTY_CANVAS_REWARD = -1.0

CONTRAST_FLOOR = 1.0

NAMED_COLORS = {
    "yellow": (255, 215, 0),
    "red": (220, 38, 38),
    "blue": (37, 99, 235),
    "green": (22, 163, 74),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "orange": (249, 115, 22),
    "purple": (147, 51, 234),
}
COLOR_TOLERANCE = 90

OVERLAP_TOLERANCE = 0.05
CENTER_TOLERANCE = 10


@dataclass
class RewardBreakdown:
    """Per-component scores and the reasons behind them.

    A single number isn't debuggable. When a rollout scores 0.4, the useful
    question is which component lost the points.
    """

    constraints: float = 0.0
    contrast: float = 0.0
    layout: float = 0.0
    total: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total": round(self.total, 3),
            "constraints": round(self.constraints, 3),
            "contrast": round(self.contrast, 3),
            "layout": round(self.layout, 3),
            "notes": self.notes,
        }


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Parse "#RRGGBB" into 0-255 channels, falling back to black."""
    value = color.lstrip("#")
    if len(value) != 6:
        return (0, 0, 0)
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)


def relative_luminance(color: str) -> float:
    """WCAG relative luminance. Green dominates because our eyes do."""
    channels = []
    for raw in hex_to_rgb(color):
        c = raw / 255.0
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio, from 1.0 to 21.0."""
    l1 = relative_luminance(foreground)
    l2 = relative_luminance(background)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def color_matches(color: str, name: str) -> bool:
    """Whether a hex color reads as the named color.

    Flat per-channel RGB tolerance. Coarse, and it accepts some colors a
    person would name differently, but anything tighter rejects reasonable
    choices like #FFD700 for "yellow".
    """
    reference = NAMED_COLORS.get(name.lower())
    if reference is None:
        return False
    return all(abs(a - b) <= COLOR_TOLERANCE for a, b in zip(hex_to_rgb(color), reference))


def effective_background(element: Element, canvas: Canvas) -> str:
    """What sits directly behind an element, or the canvas if nothing does."""
    behind = [
        e
        for e in canvas.elements
        if e.z_index < element.z_index
        and element.area
        and e.overlap_area(element) / element.area > 0.5
    ]
    if not behind:
        return canvas.background_color
    return max(behind, key=lambda e: e.z_index).color


def _requirement_satisfied(
    requirement: ElementRequirement, canvas: Canvas
) -> tuple[bool, str]:
    """Check one required element, returning the reason on failure."""
    candidates = canvas.elements_by_role(requirement.role)
    if not candidates:
        return False, f"missing {requirement.role}"

    # Exactly one, so a policy can't spray duplicates hoping one lands.
    if len(candidates) > 1:
        return False, f"{len(candidates)} elements claim role {requirement.role}"

    element = candidates[0]
    if element.type.value != requirement.type:
        return False, f"{requirement.role} is {element.type.value}, expected {requirement.type}"
    if requirement.min_width and element.width < requirement.min_width:
        return False, f"{requirement.role} too narrow ({element.width}px)"
    if requirement.min_height and element.height < requirement.min_height:
        return False, f"{requirement.role} too short ({element.height}px)"
    if requirement.color and not color_matches(element.color, requirement.color):
        return False, f"{requirement.role} is {element.color}, expected {requirement.color}"

    return True, f"{requirement.role} ok"


def score_constraints(canvas: Canvas, task: Task) -> tuple[float, list[str]]:
    """Fraction of required elements present and valid.

    Partial credit is intentional. Demanding all of them before any reward
    leaves a fresh policy with no gradient to follow.
    """
    if not task.required_elements:
        return 1.0, []

    satisfied = 0
    notes = []
    for requirement in task.required_elements:
        ok, note = _requirement_satisfied(requirement, canvas)
        satisfied += int(ok)
        notes.append(note)

    return satisfied / len(task.required_elements), notes


def score_contrast(canvas: Canvas, task: Task) -> tuple[float, list[str]]:
    """Readability of the least readable element carrying text.

    The minimum, not the mean. Accessibility isn't an average: one headline
    invisible against its background is a broken banner even if everything
    else is fine, and a mean would let an agent dilute one unreadable
    element by adding several legible ones.

    The cost is a sparser signal. Improving anything other than the worst
    element moves nothing.

    Labelled shapes count too: a CTA whose text disappears into its own fill
    is unreadable whatever element type it happens to be. Text sits on
    whatever is behind it; a labelled shape sits on its own fill.

    Ramped, not thresholded, so 4.4 and 4.6 aren't worlds apart and a policy
    gets signal for improving 1.5 toward 4.0.
    """
    labelled = [e for e in canvas.elements if e.content]
    if not labelled:
        return 0.0, ["no text to evaluate"]

    target = task.min_contrast_ratio
    scores = []
    notes = []
    for element in labelled:
        if element.type is ElementType.TEXT:
            background = effective_background(element, canvas)
        else:
            background = element.color
        ratio = contrast_ratio(element.text_color, background)
        normalized = (ratio - CONTRAST_FLOOR) / (target - CONTRAST_FLOOR)
        scores.append(max(0.0, min(1.0, normalized)))
        notes.append(f"{element.role} contrast {ratio:.1f}:1 on {background}")

    return min(scores), notes


def score_layout(canvas: Canvas) -> tuple[float, list[str]]:
    """Overlap, bounds, and alignment, each on its own sub-weight."""
    notes = []
    elements = canvas.elements

    # Full containment is skipped: text on a backing shape is composition,
    # not collision. Only partial overlap counts.
    worst_overlap = 0.0
    for i, a in enumerate(elements):
        for b in elements[i + 1 :]:
            overlap = a.overlap_area(b)
            if not overlap:
                continue
            smaller_area = min(a.area, b.area)
            if not smaller_area or overlap >= smaller_area:
                continue
            ratio = overlap / smaller_area
            if ratio > OVERLAP_TOLERANCE:
                worst_overlap = max(worst_overlap, ratio)
                notes.append(f"{a.element_id}/{b.element_id} overlap {ratio:.0%}")
    overlap_score = 1.0 - worst_overlap

    out = [e for e in elements if canvas.is_out_of_bounds(e)]
    bounds_score = 1.0 - (len(out) / len(elements))
    for element in out:
        notes.append(f"{element.element_id} out of bounds")

    centered = [
        e for e in elements if abs(e.center_x - canvas.width / 2) <= CENTER_TOLERANCE
    ]
    alignment_score = len(centered) / len(elements)

    total = (
        W_OVERLAP * overlap_score
        + W_BOUNDS * bounds_score
        + W_ALIGNMENT * alignment_score
    )
    return total / W_LAYOUT, notes


def compute_reward(canvas: Canvas, task: Task) -> RewardBreakdown:
    """Score a finished canvas in [-1.0, 1.0]."""
    if not canvas.elements:
        return RewardBreakdown(total=EMPTY_CANVAS_REWARD, notes=["empty canvas"])

    constraints, constraint_notes = score_constraints(canvas, task)
    contrast, contrast_notes = score_contrast(canvas, task)
    layout, layout_notes = score_layout(canvas)

    # Components are in [0, 1]; rescale so a canvas satisfying nothing lands
    # at the same floor as an empty one.
    quality = W_CONSTRAINTS * constraints + W_CONTRAST * contrast + W_LAYOUT * layout
    total = 2.0 * quality - 1.0

    return RewardBreakdown(
        constraints=constraints,
        contrast=contrast,
        layout=layout,
        total=total,
        notes=constraint_notes + contrast_notes + layout_notes,
    )