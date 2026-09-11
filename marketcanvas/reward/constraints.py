"""Did the design contain what the brief asked for?"""

from marketcanvas.canvas import Canvas
from marketcanvas.reward.base import ComponentScore, Scorer
from marketcanvas.reward.color import color_matches
from marketcanvas.tasks import ElementRequirement, Task


class ConstraintScorer(Scorer):
    """Fraction of the brief's required elements that are present and valid.

    Partial credit is intentional. Demanding every element before any reward
    leaves a fresh policy with no gradient to follow.
    """

    name = "constraints"

    def score(self, canvas: Canvas, task: Task) -> ComponentScore:
        if not task.required_elements:
            return ComponentScore(1.0, [])

        satisfied = 0
        notes = []
        for requirement in task.required_elements:
            ok, note = self._check(requirement, canvas)
            satisfied += int(ok)
            notes.append(note)

        return ComponentScore(satisfied / len(task.required_elements), notes)

    def _check(
        self, requirement: ElementRequirement, canvas: Canvas
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
            return (
                False,
                f"{requirement.role} is {element.type.value}, expected {requirement.type}",
            )
        if requirement.min_width and element.width < requirement.min_width:
            return False, f"{requirement.role} too narrow ({element.width}px)"
        if requirement.min_height and element.height < requirement.min_height:
            return False, f"{requirement.role} too short ({element.height}px)"
        if requirement.color and not color_matches(element.color, requirement.color):
            return (
                False,
                f"{requirement.role} is {element.color}, expected {requirement.color}",
            )

        return True, f"{requirement.role} ok"
