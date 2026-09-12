"""Semantic observation: how the agent sees the canvas.

The observation is a JSON-serializable dict, not a pixel buffer. Raw
coordinates alone would force the model to do arithmetic every step to
answer questions it needs on every step ("do these overlap?", "is this
centered?"), so the spatial relations are precomputed here.
What is deliberately absent is the current reward and the scoring weights.
The task requirements reach the agent as natural language in the prompt.
Exposing the scoring function itself would let a policy optimize the metric
directly instead of the design.
"""

from marketcanvas.canvas import Canvas
from marketcanvas.elements import Element, ElementType
from marketcanvas.tasks import Task

# Fraction of the smaller element's area that must be covered before an
# overlap is reported. Below this, boxes are touching, not colliding.
OVERLAP_REPORT_THRESHOLD = 0.05

# Pixels of slack allowed when calling something "centered". Demanding exact
# equality would make alignment unreachable for odd-width elements.
CENTER_TOLERANCE = 10


def relative_position(a: Element, b: Element) -> str:
    """Coarse direction from a to b, based on center offsets.
    Whichever axis separates the two centers more decides the label, so a
    pair is described the way a person would describe it rather than by a
    diagonal nobody uses.
    """
    dx = b.center_x - a.center_x
    dy = b.center_y - a.center_y
    if abs(dx) >= abs(dy):
        return "right_of" if dx > 0 else "left_of"
    return "below" if dy > 0 else "above"


def describe_element(element: Element, canvas: Canvas) -> dict:
    """One element's own properties, plus facts about it alone."""
    data = element.to_dict()
    data["center_x"] = element.center_x
    data["center_y"] = element.center_y
    data["out_of_bounds"] = canvas.is_out_of_bounds(element)
    data["horizontally_centered"] = (
        abs(element.center_x - canvas.width / 2) <= CENTER_TOLERANCE
    )
    data["vertically_centered"] = (
        abs(element.center_y - canvas.height / 2) <= CENTER_TOLERANCE
    )
    return data


def describe_relations(canvas: Canvas) -> list[dict]:
    """Pairwise facts: direction, overlap, and alignment.
    Each unordered pair is reported once, from the earlier element's point
    of view, so the agent does not have to reconcile two mirror-image
    statements about the same pair.
    """
    relations = []
    elements = canvas.elements_in_z_order()

    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            overlap = a.overlap_area(b)
            smaller = min(a.area, b.area)
            overlap_ratio = overlap / smaller if smaller else 0.0

            relation = {
                "from": a.element_id,
                "to": b.element_id,
                "position": relative_position(a, b),
                "aligned_center_x": abs(a.center_x - b.center_x) <= CENTER_TOLERANCE,
                "aligned_left": a.left == b.left,
                "aligned_top": a.top == b.top,
            }

            if overlap_ratio > OVERLAP_REPORT_THRESHOLD:
                relation["overlaps"] = True
                relation["overlap_ratio"] = round(overlap_ratio, 3)
                # The higher z_index wins, and ties cannot happen because the
                # canvas assigns z itself.
                relation["occluded"] = a.element_id if a.z_index < b.z_index else b.element_id
            else:
                relation["overlaps"] = False

            relations.append(relation)

    return relations


def effective_background(element: Element, canvas: Canvas) -> str:
    """The color a text element actually sits on.
    Contrast is only meaningful against whatever is directly behind the text,
    which is the topmost element below it that covers most of its box, or the
    canvas background when nothing does.
    """
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


def build_observation(canvas: Canvas, task: Task, step: int, max_steps: int) -> dict:
    """Assemble the full observation the agent receives each step."""
    elements = [describe_element(e, canvas) for e in canvas.elements_in_z_order()]

    for data, element in zip(elements, canvas.elements_in_z_order()):
        if element.type is ElementType.TEXT:
            data["effective_background"] = effective_background(element, canvas)

    return {
        "prompt": task.prompt,
        "canvas": {
            "width": canvas.width,
            "height": canvas.height,
            "background_color": canvas.background_color,
        },
        "elements": elements,
        "relations": describe_relations(canvas),
        "budget": {
            "elements_used": len(canvas.elements),
            "elements_max": canvas.max_elements,
            "step": step,
            "max_steps": max_steps,
        },
    }