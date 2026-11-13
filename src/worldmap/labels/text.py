"""Text as outlines (matplotlib TextPath), in poster pixels: no figure is needed until it is drawn."""

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

from worldmap.frame import DPI

PX_PER_PT = DPI / 72


def font(size: float, italic: bool = False) -> FontProperties:
    return FontProperties(size=size, style="italic" if italic else "normal")


def text_width(text: str, prop: FontProperties) -> float:
    """Advance width of `text` in points."""
    return TextToPath().get_text_width_height_descent(text, prop, ismath=False)[0] if text else 0.0
