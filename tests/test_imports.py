import subprocess
import sys

# Imports every worldmap module, then reports whether pyplot got imported along the way.
SCRIPT = """
import pkgutil, importlib, sys, worldmap
for m in pkgutil.walk_packages(worldmap.__path__, "worldmap."):
    if m.name != "worldmap.__main__":
        importlib.import_module(m.name)
print("matplotlib.pyplot" in sys.modules)
"""


def test_import_has_no_pyplot() -> None:
    """Importing pyplot picks a backend; `--pick` needs to choose the interactive one first."""
    out = subprocess.run([sys.executable, "-c", SCRIPT], check=True, capture_output=True, text=True).stdout
    assert out.strip() == "False"
