"""Action space: what the agent is allowed to do.

Two layers over the same canvas mutations. High-level names elements and
intents directly, which keeps episodes short enough for credit assignment
to work. Low-level mimics real computer use and is what we care about at
deployment, but it stretches one semantic edit into several steps.

Both funnel into Canvas, so the layers can't disagree about what an action
means.
"""

from dataclasses import dataclass
from enum import Enum

from marketcanvas.canvas import Canvas, CanvasError
from marketcanvas.elements import Element, ElementType


class ActionType(str, Enum):
    """Every verb the environment accepts."""

    ADD_ELEMENT = "add_element"
    MOVE_ELEMENT = "move_element"
    RESIZE_ELEMENT = "resize_element"
    CHANGE_ELEMENT_COLOR = "change_element_color"
    CHANGE_TEXT_COLOR = "change_text_color"
    SET_CONTENT = "set_content"
    SUBMIT = "submit"
    NOOP = "noop"

    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DRAG = "mouse_drag"
    KEYBOARD_TYPE = "keyboard_type"


HIGH_LEVEL_ACTIONS = frozenset(
    {
        ActionType.ADD_ELEMENT,
        ActionType.MOVE_ELEMENT,
        ActionType.RESIZE_ELEMENT,
        ActionType.CHANGE_ELEMENT_COLOR,
        ActionType.CHANGE_TEXT_COLOR,
        ActionType.SET_CONTENT,
        ActionType.SUBMIT,
        ActionType.NOOP,
    }
)

LOW_LEVEL_ACTIONS = frozenset(
    {
        ActionType.MOUSE_MOVE,
        ActionType.MOUSE_CLICK,
        ActionType.MOUSE_DRAG,
        ActionType.KEYBOARD_TYPE,
    }
)


@dataclass
class ActionResult:
    """Outcome of one action.

    A rejected action is a normal event, not a crash: the agent is told why
    and the episode continues with a step consumed.
    """

    ok: bool
    message: str = ""
    element_id: str | None = None


@dataclass
class Cursor:
    """Pointer state for the low-level layer.

    Selection has to live somewhere for click-then-type to mean anything,
    and it isn't a property of the canvas.
    """

    x: int = 0
    y: int = 0
    selected_id: str | None = None


class ActionHandler:
    """Applies actions to a canvas, in either layer."""

    def __init__(self, canvas: Canvas, allow_low_level: bool = True):
        self.canvas = canvas
        self.allow_low_level = allow_low_level
        self.cursor = Cursor()

    def apply(self, action: dict) -> ActionResult:
        """Dispatch one action dict, turning invalid ones into failures."""
        raw_type = action.get("type")
        try:
            action_type = ActionType(raw_type)
        except ValueError:
            return ActionResult(False, f"unknown action type '{raw_type}'")

        if action_type in LOW_LEVEL_ACTIONS and not self.allow_low_level:
            return ActionResult(False, "low-level actions are disabled")

        handler = self._handlers().get(action_type)
        try:
            return handler(action)
        except CanvasError as exc:
            return ActionResult(False, str(exc))
        except (KeyError, TypeError, ValueError) as exc:
            return ActionResult(False, f"malformed action: {exc}")

    def _handlers(self) -> dict:
        return {
            ActionType.ADD_ELEMENT: self._add_element,
            ActionType.MOVE_ELEMENT: self._move_element,
            ActionType.RESIZE_ELEMENT: self._resize_element,
            ActionType.CHANGE_ELEMENT_COLOR: self._change_color,
            ActionType.CHANGE_TEXT_COLOR: self._change_text_color,
            ActionType.SET_CONTENT: self._set_content,
            ActionType.SUBMIT: self._submit,
            ActionType.NOOP: self._noop,
            ActionType.MOUSE_MOVE: self._mouse_move,
            ActionType.MOUSE_CLICK: self._mouse_click,
            ActionType.MOUSE_DRAG: self._mouse_drag,
            ActionType.KEYBOARD_TYPE: self._keyboard_type,
        }

    # High-level handlers

    def _add_element(self, action: dict) -> ActionResult:
        element = self.canvas.add_element(
            type=ElementType(action["element_type"]),
            role=action["role"],
            x=int(action["x"]),
            y=int(action["y"]),
            width=int(action["width"]),
            height=int(action["height"]),
            color=action.get("color", "#FFFFFF"),
            text_color=action.get("text_color", "#000000"),
            content=action.get("content", ""),
        )
        self.cursor.selected_id = element.element_id
        return ActionResult(True, "added", element.element_id)

    def _move_element(self, action: dict) -> ActionResult:
        element = self.canvas.move_element(
            action["element_id"], int(action["x"]), int(action["y"])
        )
        return ActionResult(True, "moved", element.element_id)

    def _resize_element(self, action: dict) -> ActionResult:
        element = self.canvas.resize_element(
            action["element_id"], int(action["width"]), int(action["height"])
        )
        return ActionResult(True, "resized", element.element_id)

    def _change_color(self, action: dict) -> ActionResult:
        element = self.canvas.set_color(action["element_id"], action["color"])
        return ActionResult(True, "recolored", element.element_id)

    def _change_text_color(self, action: dict) -> ActionResult:
        element = self.canvas.set_text_color(action["element_id"], action["text_color"])
        return ActionResult(True, "text recolored", element.element_id)

    def _set_content(self, action: dict) -> ActionResult:
        element = self.canvas.set_content(action["element_id"], action["content"])
        return ActionResult(True, "content set", element.element_id)

    def _submit(self, action: dict) -> ActionResult:
        """Declares the design finished. env.step reads the type to end it."""
        return ActionResult(True, "submitted")

    def _noop(self, action: dict) -> ActionResult:
        return ActionResult(True, "noop")

    # Low-level handlers

    def _element_at(self, x: int, y: int) -> Element | None:
        """Topmost element under the point, matching what a click hits."""
        hits = [
            e
            for e in self.canvas.elements
            if e.left <= x <= e.right and e.top <= y <= e.bottom
        ]
        if not hits:
            return None
        return max(hits, key=lambda e: e.z_index)

    def _mouse_move(self, action: dict) -> ActionResult:
        self.cursor.x = int(action["x"])
        self.cursor.y = int(action["y"])
        return ActionResult(True, f"cursor at ({self.cursor.x}, {self.cursor.y})")

    def _mouse_click(self, action: dict) -> ActionResult:
        element = self._element_at(self.cursor.x, self.cursor.y)
        self.cursor.selected_id = element.element_id if element else None
        if element is None:
            return ActionResult(True, "clicked empty canvas")
        return ActionResult(True, "selected", element.element_id)

    def _mouse_drag(self, action: dict) -> ActionResult:
        """Translates whatever is under the start point by the delta."""
        x1, y1 = int(action["x1"]), int(action["y1"])
        x2, y2 = int(action["x2"]), int(action["y2"])
        element = self._element_at(x1, y1)
        if element is None:
            self.cursor.x, self.cursor.y = x2, y2
            return ActionResult(False, "nothing to drag at start point")

        self.canvas.move_element(
            element.element_id, element.x + (x2 - x1), element.y + (y2 - y1)
        )
        self.cursor.x, self.cursor.y = x2, y2
        self.cursor.selected_id = element.element_id
        return ActionResult(True, "dragged", element.element_id)

    def _keyboard_type(self, action: dict) -> ActionResult:
        """Appends to the selected text element, as a text field would."""
        if self.cursor.selected_id is None:
            return ActionResult(False, "nothing selected")
        element = self.canvas.get_element(self.cursor.selected_id)
        if element.type is not ElementType.TEXT:
            return ActionResult(False, "selected element is not text")
        self.canvas.set_content(
            element.element_id, element.content + str(action["text"])
        )
        return ActionResult(True, "typed", element.element_id)