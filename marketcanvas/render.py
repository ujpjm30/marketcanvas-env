"""Rasterizes a canvas to RGB pixels.

Rendering is deliberately kept out of the step path. Nothing in the
observation or the reward reads pixels, so this runs only when a caller
actually wants an image, which is what lets the environment stay cheap
under parallel rollouts.
"""

from PIL import Image, ImageDraw, ImageFont

import numpy as np

from marketcanvas.canvas import Canvas
from marketcanvas.elements import Element, ElementType

# Images are drawn as a flat fill plus a border so they read as a distinct
# element type without needing real image assets.
IMAGE_BORDER_COLOR = "#666666"

# Text sizing. Start at half the box height, then shrink until the label fits
# the width with a little breathing room on each side.
FONT_HEIGHT_RATIO = 0.5
TEXT_FIT_MARGIN = 0.9
MIN_FONT_SIZE = 8


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Best available font at the requested size, falling back to the default.

    Font availability differs per machine, and a missing font should degrade
    the picture rather than fail the run.
    """
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_font(
    draw: ImageDraw.ImageDraw, text: str, box_width: int, box_height: int
) -> ImageFont.FreeTypeFont:
    """Largest font from the box height down that still fits the box width.

    Sizing on height alone overflows as soon as the label is long, and a
    banner whose CTA reads "HOP NOW" is broken however well it scores. The
    reward does not measure this, so the renderer has to.
    """
    size = max(MIN_FONT_SIZE, int(box_height * FONT_HEIGHT_RATIO))
    while size > MIN_FONT_SIZE:
        font = _load_font(size)
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        if right - left <= box_width * TEXT_FIT_MARGIN:
            return font
        size -= 2
    return _load_font(MIN_FONT_SIZE)


def _draw_element(draw: ImageDraw.ImageDraw, element: Element) -> None:
    box = [element.left, element.top, element.right, element.bottom]

    if element.type is ElementType.IMAGE:
        draw.rectangle(box, fill=element.color, outline=IMAGE_BORDER_COLOR, width=2)
        return

    draw.rectangle(box, fill=element.color)

    if not element.content:
        return

    font = _fit_font(draw, element.content, element.width, element.height)
    left, top, right, bottom = draw.textbbox((0, 0), element.content, font=font)
    draw.text(
        (element.center_x - (right - left) / 2 - left,
         element.center_y - (bottom - top) / 2 - top),
        element.content,
        fill=element.text_color,
        font=font,
    )


def render(canvas: Canvas) -> Image.Image:
    """Paint the canvas back to front and return the image."""
    image = Image.new("RGB", (canvas.width, canvas.height), canvas.background_color)
    draw = ImageDraw.Draw(image)
    for element in canvas.elements_in_z_order():
        _draw_element(draw, element)
    return image


def render_array(canvas: Canvas) -> np.ndarray:
    """Render to an (H, W, 3) uint8 array for multimodal observations."""
    return np.asarray(render(canvas), dtype=np.uint8)


def save_png(canvas: Canvas, path: str) -> str:
    """Render and write a PNG, returning the path written."""
    render(canvas).save(path)
    return path
