"""Pins: golden spheres drawn at (lon, lat) points, and an interactive picker for adding them."""

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from worldmap.frame import POSTER_WIDTH_IN, MapFrame

LonLat = tuple[float, float]

PIN_ZORDER = 10  # above labels
PICK_WINDOW_IN = 14  # width of the --pick window
PICK_DOWNSAMPLE = 4  # the rendered map is shown at 1/4 size in the --pick window


@dataclass(frozen=True)
class PinStyle:
    """Look of the pin sprite. Distances are in sphere radii; the reflected view ray's elevation runs from -1
    (straight down, the ground) to 1 (straight up, the sky)."""

    pad: float = 1.4  # sprite canvas half-width (room for the drop shadow)
    # The metal reflects an environment: gold ground, horizon, a dark band above it, a bright cream sky.
    dome_bend: float = 0.7  # how much the horizon curves into a dome towards the sides
    ground_level: float = 0.68
    ground_dip: float = 0.2  # ground darkens by this much far below the horizon...
    ground_dip_edges: tuple[float, float] = (-1.0, -0.3)  # ...fading out between these elevations
    sky_level: float = 0.8
    sky_rise: float = 0.2  # sky brightens by this much high up...
    sky_rise_edges: tuple[float, float] = (0.4, 1.0)  # ...between these elevations
    horizon_edges: tuple[float, float] = (0.05, 0.5)  # ground -> sky blend
    band_elevation: float = -0.2  # bright band at this elevation
    band_strength: float = 0.1
    band_width_below: float = 0.07  # sharp below...
    band_width_above: float = 0.3  # ...soft above
    # Window-like bright strips down each side.
    strip_right: float = 0.55
    strip_left: float = 0.3
    strip_position: float = 0.85  # |horizontal reflected direction| at the strip centre
    strip_width: float = 0.14
    strip_fade_edges: tuple[float, float] = (0.0, 0.35)  # strips fade in above the horizon
    # Shade (0..1) -> gold colour ramp.
    ramp_stops: tuple[float, ...] = (0, 0.35, 0.65, 0.9, 1)
    ramp_colors: tuple[tuple[float, float, float], ...] = ((0.30, 0.17, 0.02), (0.65, 0.42, 0.05), (0.92, 0.70, 0.16), (0.98, 0.80, 0.28), (1.0, 0.93, 0.68))
    rim_floor: float = 0.7  # brightness at the rim; centre is rim_floor + rim_gain
    rim_gain: float = 0.3
    # Specular highlight from a key light in the upper left: broad soft glare plus a hot core.
    key_light: tuple[float, float, float] = (-0.45, 0.6, 0.65)
    glare: float = 0.4
    glare_power: float = 12
    core: float = 0.8
    core_power: float = 70
    # Black drop shadow, offset down and right.
    shadow_offset: tuple[float, float] = (0.12, -0.16)
    shadow_radius: float = 1.15
    shadow_softness: float = 0.35
    shadow_strength: float = 0.45


def load_pins(path: str | Path) -> list[LonLat]:
    """(lon, lat) pairs from a text file with one 'lon,lat' per line; '#' starts a comment."""
    pins = []
    for n, line in enumerate(open(path), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        try:
            lon, lat = (float(s) for s in line.replace(",", " ").split()[:2])
        except ValueError:
            sys.exit("%s:%d: expected 'lon,lat', got %r" % (path, n, line))
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            sys.exit("%s:%d: coordinates out of range: %r" % (path, n, line))
        pins.append((lon, lat))
    return pins


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def pin_sprite(diameter_px: float, style: PinStyle = PinStyle()) -> np.ndarray:
    """RGBA (float, 0..1) gold sphere with a chrome-like reflected environment, specular glare and a soft drop shadow."""
    radius_px = diameter_px / 2
    n = int(round(2 * style.pad * radius_px))
    axis = np.linspace(-style.pad, style.pad, n)
    x, y = np.meshgrid(axis, -axis)  # in sphere radii, y up
    r = np.hypot(x, y)
    z = np.sqrt(np.clip(1 - r**2, 0, 1))  # sphere surface height (0 outside)
    normal = np.stack([x, y, z], -1)

    # Metal: shade by what the surface reflects, not by diffuse light.
    reflect_y = 2 * z * y + style.dome_bend * x**2  # elevation of the reflected view ray
    reflect_x = 2 * z * x
    ground = style.ground_level - style.ground_dip * (1 - smoothstep(*style.ground_dip_edges, reflect_y))
    sky = style.sky_level + style.sky_rise * smoothstep(*style.sky_rise_edges, reflect_y)
    shade = ground + (sky - ground) * smoothstep(*style.horizon_edges, reflect_y)
    band_width = np.where(reflect_y < style.band_elevation, style.band_width_below, style.band_width_above)
    shade += style.band_strength * np.exp(-(((reflect_y - style.band_elevation) / band_width) ** 2))
    strip_strength = np.where(reflect_x > 0, style.strip_right, style.strip_left)
    strip = strip_strength * np.exp(-(((np.abs(reflect_x) - style.strip_position) / style.strip_width) ** 2)) * smoothstep(*style.strip_fade_edges, reflect_y)
    shade = np.clip(shade + strip, 0, 1)
    ramp = np.array(style.ramp_colors)
    rgb = np.stack([np.interp(shade, list(style.ramp_stops), ramp[:, k]) for k in range(3)], -1)
    rgb *= (style.rim_floor + style.rim_gain * np.sqrt(z))[..., None]

    key = np.array(style.key_light)
    key /= np.linalg.norm(key)
    half_vector = key + [0, 0, 1]
    highlight = np.clip(normal @ (half_vector / np.linalg.norm(half_vector)), 0, None)
    spec = np.clip(style.glare * highlight**style.glare_power + style.core * highlight**style.core_power, 0, 1)[..., None]
    rgb = rgb + (1 - rgb) * spec

    sphere_alpha = np.clip((1 - r) * radius_px + 0.5, 0, 1)  # anti-aliased edge
    shadow_dx, shadow_dy = style.shadow_offset
    shadow = np.clip((style.shadow_radius - np.hypot(x - shadow_dx, y - shadow_dy)) / style.shadow_softness, 0, 1)
    shadow_alpha = style.shadow_strength * shadow * shadow * (3 - 2 * shadow)
    alpha = sphere_alpha + shadow_alpha * (1 - sphere_alpha)
    rgb = np.where(alpha[..., None] > 0, rgb * sphere_alpha[..., None] / np.maximum(alpha, 1e-9)[..., None], 0)
    return np.dstack([rgb, alpha])


def draw_pins(ax: Axes, frame: MapFrame, pins: Sequence[LonLat], sprite: np.ndarray, diameter_in: float, style: PinStyle = PinStyle()) -> list[AxesImage]:
    """Draw pins northernmost first, so southern ones overlap them. Returns the image artists."""
    half = style.pad * diameter_in / 2 * frame.width / POSTER_WIDTH_IN
    out = []
    for lon, lat in sorted(pins, key=lambda p: -p[1]):
        x, y = frame.to_map.transform(lon, lat)
        out.append(ax.imshow(sprite, extent=(x - half, x + half, y - half, y + half), zorder=PIN_ZORDER, interpolation="bilinear"))
    return out


def pick_pins(path: str, rendered_png: str, frame: MapFrame, diameter_in: float, dpi: float) -> None:
    """Show the rendered map in a window; left-click adds a pin, right-click or `u` undoes one. Closing the window
    appends the new pins to `path` (existing lines are left alone)."""
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    from PIL import Image

    im = Image.open(rendered_png).convert("RGBA").reduce(PICK_DOWNSAMPLE)
    existing = load_pins(path) if os.path.exists(path) else []
    new: list[LonLat] = []
    sprite = pin_sprite(diameter_in * dpi)
    outline = MplPath(np.array(frame.outline))
    fig = plt.figure(figsize=(PICK_WINDOW_IN, PICK_WINDOW_IN * frame.height / frame.width))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(np.asarray(im), extent=(frame.xmin, frame.xmax, frame.ymin, frame.ymax))
    ax.axis("off")
    artists: list[AxesImage] = []

    def redraw() -> None:
        for a in artists:
            a.remove()
        artists[:] = draw_pins(ax, frame, existing + new, sprite, diameter_in)
        ax.set_xlim(frame.xmin, frame.xmax)
        ax.set_ylim(frame.ymin, frame.ymax)
        fig.canvas.draw_idle()

    def undo() -> None:
        if new:
            print("removed %.4f,%.4f" % new.pop())

    def on_click(e) -> None:
        toolbar = fig.canvas.toolbar
        if e.inaxes is not ax or (toolbar is not None and toolbar.mode):  # ignore clicks while zoom/pan is active
            return
        if e.button == 3:
            undo()
        elif e.button == 1 and outline.contains_point((e.xdata, e.ydata)):
            new.append(frame.to_lonlat.transform(e.xdata, e.ydata))
            print("added   %.4f,%.4f" % new[-1])
        redraw()

    def on_key(e) -> None:
        if e.key == "u":
            undo()
            redraw()

    def save(_) -> None:
        if not new:
            print("no new pins")
            return
        text = open(path).read() if os.path.exists(path) else ""
        with open(path, "a") as f:
            if text and not text.endswith("\n"):
                f.write("\n")
            f.writelines("%.4f,%.4f\n" % p for p in new)
        print("added %d pins to %s (%d total)" % (len(new), path, len(existing) + len(new)))

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", save)
    redraw()
    print("left-click: add pin; right-click or u: undo; close the window to save to", path)
    plt.show()
