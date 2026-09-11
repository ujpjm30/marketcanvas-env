"""Action layer tests.

The point of these is the claim the write-up makes: that the low-level layer
is a complete action space on its own, not a set of edits that still needs a
high-level call to create anything. The first test is the one that matters —
a full episode from a blank canvas, low-level only, scoring 1.0.
"""

import pytest

from marketcanvas.actions import ActionHandler, ActionType, LOW_LEVEL_ACTIONS
from marketcanvas.canvas import Canvas
from marketcanvas.elements import ElementType
from marketcanvas.env import MarketCanvasEnv

LOW_LEVEL_EPISODE = [
    {"type": "select_tool", "tool": "shape", "role": "background", "color": "#1A237E"},
    {"type": "mouse_drag", "x1": 0, "y1": 0, "x2": 800, "y2": 600},
    {"type": "select_tool", "tool": "text", "role": "headline",
     "color": "#1A237E", "text_color": "#FFFFFF"},
    {"type": "mouse_drag", "x1": 200, "y1": 140, "x2": 600, "y2": 220},
    {"type": "keyboard_type", "text": "SUMMER SALE"},
    {"type": "select_tool", "tool": "shape", "role": "cta_button",
     "color": "#FFD700", "text_color": "#1A237E"},
    {"type": "mouse_drag", "x1": 320, "y1": 360, "x2": 480, "y2": 416},
    {"type": "keyboard_type", "text": "SHOP NOW"},
    {"type": "submit"},
]


def handler() -> ActionHandler:
    return ActionHandler(Canvas())


# The completeness claim

def test_low_level_alone_can_finish_the_task():
    """A blank canvas to a scoring banner without one high-level action."""
    env = MarketCanvasEnv()
    env.reset()

    for action in LOW_LEVEL_EPISODE:
        assert ActionType(action["type"]) in LOW_LEVEL_ACTIONS or action["type"] == "submit"
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["action_ok"], f'{action["type"]} failed: {info["action_message"]}'

    assert len(obs["elements"]) == 3
    assert reward == pytest.approx(1.0)


def test_drawing_requires_a_tool():
    """Without a toolbar there is nothing to grab on a blank canvas."""
    h = handler()
    result = h.apply({"type": "mouse_drag", "x1": 10, "y1": 10, "x2": 100, "y2": 100})
    assert not result.ok
    assert len(h.canvas.elements) == 0


# Tool semantics

def test_active_tool_draws_even_over_an_existing_element():
    """Draw mode is modal. A tool means draw, whatever is underneath."""
    h = handler()
    h.apply({"type": "select_tool", "tool": "shape", "role": "background"})
    h.apply({"type": "mouse_drag", "x1": 0, "y1": 0, "x2": 800, "y2": 600})
    h.apply({"type": "select_tool", "tool": "text", "role": "headline"})
    h.apply({"type": "mouse_drag", "x1": 200, "y1": 140, "x2": 600, "y2": 220})

    assert len(h.canvas.elements) == 2
    assert h.canvas.get_element("el_1").x == 0  # background was not dragged


def test_clearing_the_tool_returns_to_select_mode():
    h = handler()
    h.apply({"type": "select_tool", "tool": "shape", "role": "box"})
    h.apply({"type": "mouse_drag", "x1": 100, "y1": 100, "x2": 300, "y2": 200})
    h.apply({"type": "select_tool", "tool": None})
    h.apply({"type": "mouse_drag", "x1": 150, "y1": 150, "x2": 350, "y2": 350})

    element = h.canvas.get_element("el_1")
    assert (element.x, element.y) == (300, 300)


@pytest.mark.parametrize(
    "x1,y1,x2,y2",
    [(100, 100, 300, 200), (300, 200, 100, 100), (300, 100, 100, 200)],
)
def test_drag_direction_does_not_change_the_box(x1, y1, x2, y2):
    """Dragging up-left must produce the same element as down-right."""
    h = handler()
    h.apply({"type": "select_tool", "tool": "shape", "role": "box"})
    h.apply({"type": "mouse_drag", "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    element = h.canvas.get_element("el_1")
    assert (element.x, element.y, element.width, element.height) == (100, 100, 200, 100)


def test_zero_area_drag_draws_nothing():
    h = handler()
    h.apply({"type": "select_tool", "tool": "shape", "role": "box"})
    result = h.apply({"type": "mouse_drag", "x1": 100, "y1": 100, "x2": 100, "y2": 100})
    assert not result.ok
    assert len(h.canvas.elements) == 0


def test_tool_carries_role_and_color():
    h = handler()
    h.apply({"type": "select_tool", "tool": "shape", "role": "cta_button",
             "color": "#FFD700", "text_color": "#1A237E"})
    h.apply({"type": "mouse_drag", "x1": 320, "y1": 360, "x2": 480, "y2": 416})

    element = h.canvas.get_element("el_1")
    assert element.role == "cta_button"
    assert element.color == "#FFD700"
    assert element.type is ElementType.SHAPE


# Pointer and typing

def test_click_selects_the_topmost_element():
    h = handler()
    h.apply({"type": "select_tool", "tool": "shape", "role": "back"})
    h.apply({"type": "mouse_drag", "x1": 0, "y1": 0, "x2": 800, "y2": 600})
    h.apply({"type": "select_tool", "tool": "shape", "role": "front"})
    h.apply({"type": "mouse_drag", "x1": 100, "y1": 100, "x2": 300, "y2": 200})
    h.apply({"type": "select_tool", "tool": None})

    h.apply({"type": "mouse_move", "x": 200, "y": 150})
    assert h.apply({"type": "mouse_click"}).element_id == "el_2"


def test_typing_appends_and_works_on_labelled_shapes():
    """A CTA has to be labellable, so content is not TEXT-only."""
    h = handler()
    h.apply({"type": "select_tool", "tool": "shape", "role": "cta_button"})
    h.apply({"type": "mouse_drag", "x1": 320, "y1": 360, "x2": 480, "y2": 416})
    h.apply({"type": "keyboard_type", "text": "SHOP "})
    h.apply({"type": "keyboard_type", "text": "NOW"})

    assert h.canvas.get_element("el_1").content == "SHOP NOW"


def test_typing_needs_a_selection():
    assert not handler().apply({"type": "keyboard_type", "text": "X"}).ok


# Layer gating and robustness

def test_low_level_can_be_disabled_for_ablation():
    h = ActionHandler(Canvas(), allow_low_level=False)
    assert not h.apply({"type": "select_tool", "tool": "shape"}).ok
    assert h.apply({"type": "add_element", "element_type": "shape", "role": "box",
                    "x": 0, "y": 0, "width": 10, "height": 10}).ok


def test_malformed_actions_fail_without_crashing():
    h = handler()
    for action in [
        {"type": "not_a_real_action"},
        {"type": "mouse_drag", "x1": 0},
        {"type": "move_element", "element_id": "nope", "x": 0, "y": 0},
        {"type": "select_tool", "tool": "banana"},
    ]:
        assert not h.apply(action).ok
