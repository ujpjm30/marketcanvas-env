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


def _draw_element(draw: ImageDraw.ImageDraw, element: Element) -> None:
    box = [element.left, element.top, element.right, element.bottom]

    if element.type is ElementType.IMAGE:
        draw.rectangle(box, fill=element.color, outline=IMAGE_BORDER_COLOR, width=2)
        return

    if element.type is ElementType.SHAPE:
        draw.rectangle(box, fill=element.color)

    if element.type is ElementType.TEXT:
        draw.rectangle(box, fill=element.color)

    if not element.content:
        return

    # Size the text to the box height, then center it. Real layout engines do
    # line breaking here; a single centered line is enough for this task.
    font = _load_font(max(10, int(element.height * 0.5)))
    left, top, right, bottom = draw.textbbox((0, 0), element.content, font=font)
    text_x = element.center_x - (right - left) / 2
    text_y = element.center_y - (bottom - top) / 2
    fill = element.text_color if element.type is ElementType.TEXT else element.text_color
    draw.text((text_x, text_y), element.content, fill=fill, font=font)


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