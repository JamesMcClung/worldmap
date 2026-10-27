"""Shared helpers for the regression tests: render the map in a subprocess and compare pixel fingerprints."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TESTS = Path(__file__).parent
FINGERPRINTS = TESTS / "fingerprints.json"
REFERENCE_DIR = TESTS.parent / ".cache" / "reference"


def render(tmp_path: Path, *args: str) -> Path:
    """Run `python -m worldmap out.png ARGS...` in tmp_path and return the output path."""
    subprocess.run([sys.executable, "-m", "worldmap", "out.png", *args], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path / "out.png"


def fingerprint(png: Path) -> str:
    """SHA-256 of the decoded pixels (shape and dtype included), so PNG metadata doesn't matter."""
    arr = np.asarray(Image.open(png))
    return hashlib.sha256(f"{arr.shape}|{arr.dtype}".encode() + arr.tobytes()).hexdigest()


def environment() -> dict[str, str]:
    """Versions of everything that can change the rendered pixels."""
    import geopandas
    import matplotlib
    import matplotlib.ft2font
    import PIL
    import pyproj
    import shapely

    return {
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "freetype": matplotlib.ft2font.__freetype_version__,
        "numpy": np.__version__,
        "shapely": shapely.__version__,
        "geos": shapely.geos_version_string,
        "geopandas": geopandas.__version__,
        "pyproj": pyproj.__version__,
        "pillow": PIL.__version__,
    }


def check_fingerprint(mode: str, png: Path, store: Path = FINGERPRINTS) -> None:
    """Compare png's fingerprint with the one stored for `mode`.

    UPDATE_REFERENCE=1 records it instead. A missing entry fails; an entry made with different library
    versions skips, since the pixels legitimately depend on them. Every run keeps a copy of the PNG in
    .cache/reference/ so a mismatch can be compared by eye."""
    import pytest

    entries = json.loads(store.read_text()) if store.exists() else {}
    current = {"sha256": fingerprint(png), "env": environment()}
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("UPDATE_REFERENCE") == "1":
        entries[mode] = current
        store.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
        shutil.copy(png, REFERENCE_DIR / f"{mode}.png")
        return
    if mode not in entries:
        pytest.fail(f"no fingerprint for {mode!r} in {store.name}: run with UPDATE_REFERENCE=1 to record one")
    recorded = entries[mode]
    changed = {k: (recorded["env"].get(k), v) for k, v in current["env"].items() if recorded["env"].get(k) != v}
    if changed:
        pytest.skip(f"{mode!r} fingerprint was made with other versions: " + ", ".join(f"{k} {old} -> {new}" for k, (old, new) in changed.items()))
    if recorded["sha256"] != current["sha256"]:
        new_copy = REFERENCE_DIR / f"{mode}.new.png"
        shutil.copy(png, new_copy)
        pytest.fail(f"{mode!r} output changed: compare {REFERENCE_DIR / f'{mode}.png'} with {new_copy}")
