"""Reward tests.

Two things matter here. First, that a good design scores high and a bad one
scores low. Second, that the specific ways an agent could game the reward
are actually blocked: duplicate roles, token-sized elements, text hidden in
its own fill. Each of those has a test below, and each is a loophole that
was open at some point during development.
"""

import pytest

from marketcanvas.canvas import Canvas
from marketcanvas.elements import ElementType
from marketcanvas.reward import compute_reward, contrast_ratio
from marketcanvas.tasks import get_task

TASK = get_task("summer_sale_banner")


def blank() -> Canvas:
    return Canvas(
        width=TASK.canvas_width,
        height=TASK.canvas_height,
        max_elements=TASK.max_elements,
    )


def good_banner() -> Canvas:
    """The reference design: everything the brief asks for, laid out well."""
    canvas = blank()
    canvas.add_element(
        ElementType.SHAPE, "background", 0, 0, 800, 600, color="#1A237E"
    )
    canvas.add_element(
        ElementType.TEXT, "headline", 200, 140, 400, 80,
        color="#1A237E", text_color="#FFFFFF", content="SUMMER SALE",
    )
    canvas.add_element(
        ElementType.SHAPE, "cta_button", 320, 360, 160, 56,
        color="#FFD700", text_color="#1A237E", content="SHOP NOW",
    )
    return canvas


def score(canvas: Canvas) -> float:
    return compute_reward(canvas, TASK).total


# Endpoints

def test_empty_canvas_is_the_floor():
    assert score(blank()) == -1.0


def test_good_banner_scores_high():
    assert score(good_banner()) > 0.9


def test_irrelevant_canvas_scores_low():
    """Elements that satisfy nothing in the brief earn nothing."""
    canvas = blank()
    canvas.add_element(ElementType.SHAPE, "decoration", 10, 10, 50, 50)
    assert score(canvas) < -0.4


def test_reward_stays_in_range():
    for canvas in (blank(), good_banner()):
        assert -1.0 <= score(canvas) <= 1.0


# Partial credit

def test_headline_only_beats_nothing_and_loses_to_both():
    canvas = blank()
    canvas.add_element(
        ElementType.TEXT, "headline", 200, 140, 400, 80,
        text_color="#000000", content="SUMMER SALE",
    )
    partial = score(canvas)
    assert score(blank()) < partial < score(good_banner())


# Constraint checks

def test_grey_button_fails_the_color_requirement():
    canvas = good_banner()
    canvas.set_color("el_3", "#9E9E9E")
    assert score(canvas) < score(good_banner())


def test_gold_passes_as_yellow():
    """#FFD700 is a normal choice for "yellow" and must not be rejected."""
    canvas = good_banner()
    canvas.set_color("el_3", "#FFEB3B")
    assert score(canvas) > 0.9


def test_wrong_element_type_fails():
    """A text element claiming the CTA role doesn't satisfy a shape slot."""
    canvas = blank()
    canvas.add_element(
        ElementType.TEXT, "cta_button", 320, 360, 160, 56,
        color="#FFD700", content="SHOP NOW",
    )
    assert score(canvas) < 0.0


# Gaming attempts

def test_token_sized_button_is_rejected():
    """A 4x4 yellow dot is not a call to action."""
    canvas = good_banner()
    canvas.resize_element("el_3", 4, 4)
    assert score(canvas) < score(good_banner())


def test_duplicate_roles_are_rejected():
    """Spraying headlines to hope one lands should cost, not pay."""
    canvas = good_banner()
    canvas.add_element(
        ElementType.TEXT, "headline", 100, 20, 400, 80,
        text_color="#FFFFFF", content="ALSO SUMMER SALE",
    )
    assert score(canvas) < score(good_banner())


def test_text_hidden_in_its_own_fill_is_punished():
    """Yellow-on-yellow renders as nothing and must not score as readable."""
    canvas = good_banner()
    canvas.set_text_color("el_3", "#FFD700")
    assert score(canvas) < score(good_banner())


def test_offscreen_element_is_punished():
    """Pushing an element out of frame is not a way to avoid a penalty."""
    canvas = good_banner()
    canvas.move_element("el_3", 750, 360)
    assert score(canvas) < score(good_banner())


def test_partial_overlap_is_punished():
    canvas = good_banner()
    canvas.move_element("el_3", 320, 190)
    assert score(canvas) < score(good_banner())


def test_backing_shape_is_not_an_overlap():
    """Text on a background is composition. The reference design would not
    reach 1.0 if containment counted as a collision."""
    assert score(good_banner()) > 0.9


# Contrast

def test_one_invisible_element_sinks_the_score():
    """A legible CTA must not rescue an invisible headline.

    This is why contrast takes the minimum rather than the mean: under a
    mean, this canvas scored 0.5 with a headline nobody could read.
    """
    canvas = good_banner()
    canvas.set_color("el_1", "#FFFFFF")
    canvas.set_color("el_2", "#FFFFFF")
    assert compute_reward(canvas, TASK).contrast < 0.1


@pytest.mark.parametrize(
    "fg,bg,expected",
    [
        ("#000000", "#FFFFFF", 21.0),
        ("#FFFFFF", "#FFFFFF", 1.0),
        ("#000000", "#000000", 1.0),
    ],
)
def test_contrast_ratio_endpoints(fg, bg, expected):
    """Black on white is the WCAG maximum; a color on itself is the minimum."""
    assert contrast_ratio(fg, bg) == pytest.approx(expected, abs=0.1)


def test_contrast_is_symmetric():
    assert contrast_ratio("#123456", "#ABCDEF") == pytest.approx(
        contrast_ratio("#ABCDEF", "#123456")
    )