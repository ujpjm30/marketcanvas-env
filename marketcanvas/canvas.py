"""The canvas: an ordered collection of elements plus the operations on it.

Canvas owns every mutation. Elements are passive data, so all invariants
(unique ids, z-order, bounds) are enforced in one place, which is what makes
a rollout reproducible from a seed and an action sequence.
"""

from dataclasses import dataclass, field

from marketcanvas.elements import Element, ElementType


class CanvasError(Exception):
    """Raised when an action would leave the canvas in an invalid state."""


@dataclass
class Canvas:
    """A fixed-size design surface holding a flat list of elements."""

    width: int = 800
    height: int = 600
    background_color: str = "#FFFFFF"
    max_elements: int = 8
    elements: list[Element] = field(default_factory=list)

    _next_id: int = field(default=1, repr=False)
    _next_z: int = field(default=0, repr=False)

    def add_element(
        self,
        type: ElementType,
        role: str,
        x: int,
        y: int,
        width: int,
        height: int,
        color: str = "#FFFFFF",
        text_color: str = "#000000",
        content: str = "",
    ) -> Element:
        """Append a new element and return it.

        z-index is assigned by the canvas rather than the caller: each new
        element goes strictly on top. Letting the agent pick z would allow
        ties, and a tie has no defined winner, which breaks determinism.
        """
        if len(self.elements) >= self.max_elements:
            raise CanvasError(
                f"canvas already holds {self.max_elements} elements"
            )
        if width <= 0 or height <= 0:
            raise CanvasError(f"element must have positive size, got {width}x{height}")

        element = Element(
            element_id=f"el_{self._next_id}",
            type=type,
            role=role,
            x=x,
            y=y,
            width=width,
            height=height,
            z_index=self._next_z,
            color=color,
            text_color=text_color,
            content=content,
        )
        self.elements.append(element)
        self._next_id += 1
        self._next_z += 1
        return element

    def get_element(self, element_id: str) -> Element:
        """Look up an element by id, or raise with the ids that do exist."""
        for element in self.elements:
            if element.element_id == element_id:
                return element
        raise CanvasError(
            f"no element '{element_id}'. Present: {[e.element_id for e in self.elements]}"
        )

    def move_element(self, element_id: str, new_x: int, new_y: int) -> Element:
        """Reposition an element by its top-left corner."""
        element = self.get_element(element_id)
        element.x = new_x
        element.y = new_y
        return element

    def resize_element(self, element_id: str, new_width: int, new_height: int) -> Element:
        """Resize an element, keeping its top-left corner fixed."""
        if new_width <= 0 or new_height <= 0:
            raise CanvasError(
                f"element must have positive size, got {new_width}x{new_height}"
            )
        element = self.get_element(element_id)
        element.width = new_width
        element.height = new_height
        return element

    def set_color(self, element_id: str, color: str) -> Element:
        """Change an element's fill color."""
        element = self.get_element(element_id)
        element.color = color
        return element

    def set_text_color(self, element_id: str, text_color: str) -> Element:
        """Change an element's text color."""
        element = self.get_element(element_id)
        element.text_color = text_color
        return element

    def set_content(self, element_id: str, content: str) -> Element:
        """Change an element's text content."""
        element = self.get_element(element_id)
        element.content = content
        return element

    def elements_by_role(self, role: str) -> list[Element]:
        """Every element claiming the given role. Empty list if none."""
        return [e for e in self.elements if e.role == role]

    def elements_in_z_order(self) -> list[Element]:
        """Elements from back to front, the order a renderer should paint them."""
        return sorted(self.elements, key=lambda e: e.z_index)

    def is_out_of_bounds(self, element: Element) -> bool:
        """True if any part of the element falls outside the canvas."""
        return (
            element.left < 0
            or element.top < 0
            or element.right > self.width
            or element.bottom > self.height
        )

    def clear(self) -> None:
        """Reset to an empty canvas, including the id and z counters."""
        self.elements.clear()
        self._next_id = 1
        self._next_z = 0