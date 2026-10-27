# worldmap

Renders a poster-sized political world map: a 7200 x 3693 px RGBA PNG (36 in wide at 200 dpi) in the Robinson
projection, with country borders, state/province borders for the largest countries, and non-overlapping country labels.
It can also draw pins (golden spheres) at a list of locations, e.g. to digitise a pin-your-hometown poster.

## Install

```
git clone <repo-url> worldmap
cd worldmap
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install with `-e` (editable): map data is downloaded into the repo's `.cache/` directory on first use.

## Use

```
make_map                      # writes worldmap.png
make_map out.png              # choose the output file
make_map --earth              # natural colours (sand to green) from Natural Earth land-cover imagery
make_map --pick pins.txt      # click on the rendered map to add pins to pins.txt
make_map --pins pins.txt      # draw the pins; writes worldmap_pins.png
```

| Option | Effect |
| --- | --- |
| `--earth` | Colour each region by what it looks like from space instead of distinct political colours. The first run downloads a ~75 MB raster. |
| `--pins FILE` | Draw pins from `FILE`, one `lon,lat` per line (`#` starts a comment). |
| `--pin-size IN` | Pin diameter in inches on the 36 in poster (default 0.24). |
| `--pick FILE` | Open the already-rendered map in a window. Left-click adds a pin, right-click or `u` undoes it, and closing the window appends the new pins to `FILE`. |
| `--recolor` | Recompute the country colouring instead of using the cached one. |
| `--ocean` | Experimental label placement that lets labels spill into open water (slow). |

A run takes about 10 s once the data is downloaded. Output goes to the current directory.

## Data

[Natural Earth](https://www.naturalearthdata.com/) 10m cultural vectors and the Natural Earth I land-cover raster,
both public domain. They are downloaded on first use from naciscdn.org into `.cache/`, which also holds the colouring
caches. Delete `.cache/` to start fresh.

## License

MIT; see [LICENSE](LICENSE).
