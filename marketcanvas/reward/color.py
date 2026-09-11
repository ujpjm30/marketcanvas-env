"""Color math shared by the scorers.

Lives apart from any one scorer because two of them need it: contrast reads
luminance, and the constraint check matches named colors. Everything works on
"#RRGGBB" strings, so no scorer ever has to rasterize the canvas.
"""

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
    return all(
        abs(a - b) <= COLOR_TOLERANCE for a, b in zip(hex_to_rgb(color), reference)
    )
