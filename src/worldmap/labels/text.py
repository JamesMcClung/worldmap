"""Text as outlines (matplotlib TextPath), in poster pixels: no figure is needed until it is drawn."""

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath, TextToPath

from worldmap.frame import DPI

PX_PER_PT = DPI / 72


def font(size: float, italic: bool = False) -> FontProperties:
    return FontProperties(size=size, style="italic" if italic else "normal")


def text_width(text: str, prop: FontProperties) -> float:
    """Advance width of `text` in points."""
    return TextToPath().get_text_width_height_descent(text, prop, ismath=False)[0] if text else 0.0


def lines(text: str, size: float, italic: bool = False) -> tuple[Path, float, float]:
    """`text` at `size` pt as outlines, px relative to its middle, each line centred under the one before, and the
    widest line's advance and the lines' height (px)."""
    prop = font(size, italic)
    _, line, descent = TextToPath().get_text_width_height_descent("lp", prop, ismath=False)
    rows = text.split("\n")
    top = len(rows) * line / 2
    widths = [text_width(t, prop) for t in rows]
    path = Path.make_compound_path(*(TextPath((-w / 2, top - (k + 1) * line + descent), t, prop=prop) for k, (t, w) in enumerate(zip(rows, widths))))
    return Path(np.asarray(path.vertices) * PX_PER_PT, path.codes), max(widths) * PX_PER_PT, len(rows) * line * PX_PER_PT
