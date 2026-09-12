"""Terminal reward for a finished design.

Split by responsibility, each scorer owns one question about the canvas and
nothing else, and RewardFunction owns only how their answers are mixed. That
separation is what makes an ablation a different weight list rather than an
edit to the scoring code.
All of it reads the semantic state. Contrast comes from the stored hex values,
not sampled pixels, so scoring never requires a render.
"""

from marketcanvas.reward.base import ComponentScore, Scorer
from marketcanvas.reward.color import (
    color_matches,
    contrast_ratio,
    hex_to_rgb,
    relative_luminance,
)
from marketcanvas.reward.constraints import ConstraintScorer
from marketcanvas.reward.contrast import ContrastScorer
from marketcanvas.reward.function import (
    DEFAULT_WEIGHTS,
    EMPTY_CANVAS_REWARD,
    RewardBreakdown,
    RewardFunction,
    compute_reward,
)
from marketcanvas.reward.layout import LayoutScorer

__all__ = [
    "ComponentScore",
    "ConstraintScorer",
    "ContrastScorer",
    "DEFAULT_WEIGHTS",
    "EMPTY_CANVAS_REWARD",
    "LayoutScorer",
    "RewardBreakdown",
    "RewardFunction",
    "Scorer",
    "color_matches",
    "compute_reward",
    "contrast_ratio",
    "hex_to_rgb",
    "relative_luminance",
]
