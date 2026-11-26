"""Natural Earth inputs: download/cache, load, filter and project."""

import os
import sys
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

# Data files and caches live in the repo's .cache/ (found from this file, so it needs an editable install).
CACHE = Path(__file__).resolve().parent.parent.parent / ".cache"
NE_URL = "https://naciscdn.org/naturalearth/10m/"
COUNTRIES = "ne_10m_admin_0_countries.zip"
STATES = "ne_10m_admin_1_states_provinces_lines.zip"
LAKES = "ne_10m_lakes.zip"
DISPUTED = "ne_10m_admin_0_disputed_areas.zip"
RASTER = "ne1/NE1_LR_LC.tif"  # unzipped from NE1_LR_LC.zip
GLACIERS = "ne_10m_glaciated_areas.zip"
ICE_SHELVES = "ne_10m_antarctic_ice_shelves_polys.zip"
LAND_OUTLINE = "ne_10m_land.zip"  # --type topographic: decides land vs water (heights alone get the Dead Sea wrong)
MARINE = "ne_10m_geography_marine_polys.zip"  # named oceans, seas, bays, straits... (label areas, not coastlines)
REGIONS = "ne_10m_geography_regions_polys.zip"  # named ranges, plateaus, deserts, plains... (label areas too)
RIVERS = "ne_10m_rivers_lake_centerlines.zip"  # --rivers: centrelines, joined up through the lakes they cross
PHYSICAL = {LAKES, GLACIERS, ICE_SHELVES, LAND_OUTLINE, MARINE, REGIONS, RIVERS}  # under physical/ rather than cultural/
# ETOPO 2022 (NOAA, public domain): land heights and sea depths in metres, 60 arc-second grid, top of the ice.
ETOPO = "ETOPO_2022_v1_60s_N90W180_surface.tif"
# ETOPO_BED: the same with the ice sheets (Greenland, Antarctica) taken off, the rock under them (without --ice).
ETOPO_BED = "ETOPO_2022_v1_60s_N90W180_bed.tif"
ETOPO_URL = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_surface_elev_gtif/" + ETOPO
ETOPO_BED_URL = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_bed_elev_gtif/" + ETOPO_BED
# Its 15 arc-second version, in 15-degree tiles (4-33 MB each), for zoomed-in crops; kept in .cache/etopo15/.
ETOPO15_URL = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/15s/15s_surface_elev_gtif/"
ETOPO15_BED_URL = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/15s/15s_bed_elev_gtif/"
ETOPO15_DIR = "etopo15"
# The only 15" tiles with a bed version (north-west corners): elsewhere the bed is the surface.
BED_TILES = frozenset("N60W030 N60W045 N60W060 N75W015 N75W030 N75W045 N75W060 N75W075 N90E000 N90W015 N90W030 N90W045 N90W060 N90W075 N90W090 N90W105 S60E000 S60E015 S60E030 S60E045 S60E060 S60E075 S60E090 S60E105 S60E120 S60E135 S60E150 S60E165 S60W015 S60W030 S60W045 S60W060 S60W075 S60W090 S60W105 S60W120 S60W135 S60W150 S75E000 S75E015 S75E030 S75E045 S75E060 S75E075 S75E090 S75E105 S75E120 S75E135 S75E150 S75E165 S75W015 S75W030 S75W045 S75W060 S75W075 S75W090 S75W105 S75W120 S75W135 S75W150 S75W165 S75W180".split())

DRY_LAKES = {"Lake Eyre North", "Lake Torrens", "Lake Gairdner", "Lake Mackay", "Lake Frome", "Lake Barlee"}  # Australian salt pans, usually dry
EQUAL_AREA_CRS = "+proj=eqearth"  # for every km2 (with_km2)

CRS = "+proj=robin +lon_0=0 +datum=WGS84"  # the map's projection
STATE_COUNTRIES = {
    "United States of America",
    "Canada",
    "Russia",
    "China",
    "Brazil",
    "Australia",
    "India",
    "Mexico",
    "Argentina",
    "Kazakhstan",
    "Indonesia",
}

NOT_LABELED = {"Northern Cyprus", "Turkish Republic of Northern Cyprus", "Somaliland"}
LABELED_EXTRA = {"Palestine", "Kosovo", "Western Sahara", "Israel"}
LABELED_TYPES = ("Sovereign country", "Country", "Sovereignty")
NOT_DEPENDENCY_TYPES = ("Lease", "Indeterminate")


def etopo15_tile(north: int, west: int, ice: bool = True) -> str:
    """The 15" ETOPO tile whose north-west corner is at (west, north), as a fetch name: the top of the ice, or
    without `ice` the bed (the surface tile where the bed has none)."""
    ns = f"N{north:02d}" if north >= 0 else f"S{-north:02d}"
    ew = f"E{west:03d}" if west >= 0 else f"W{-west:03d}"
    kind = "surface" if ice or ns + ew not in BED_TILES else "bed"
    return f"{ETOPO15_DIR}/ETOPO_2022_v1_15s_{ns}{ew}_{kind}.tif"


def download_size(url: str) -> int | None:
    """The file's size in bytes, from a HEAD request; None if the server doesn't say."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            length = r.headers.get("Content-Length")
    except OSError:
        return None
    return int(length) if length else None


def confirm_download(urls: list[str], folder: Path, optional: bool = False) -> bool:
    """Ask before downloading `urls` into `folder`, saying how big they are; without a terminal to ask on, stop.
    Declined: stop, or with `optional` return False (the caller can do without)."""
    sizes = [download_size(url) for url in urls]
    total = "size unknown" if None in sizes else f"{sum(s or 0 for s in sizes) / 1e6:.0f} MB"
    files = urls[0] if len(urls) == 1 else f"{len(urls)} files from {os.path.dirname(urls[0])}/"
    what = f"{files} ({total}), into {folder}"
    if not sys.stdin.isatty():
        sys.exit(f"make_map: needs to download {what}; run in a terminal to confirm, or download it there yourself")
    if input(f"make_map needs to download {what}. Download? [y/N] ").strip().lower() in ("y", "yes"):
        return True
    if optional:
        return False
    sys.exit("make_map: download declined")


def url_of(name: str) -> str:
    if name == RASTER:
        return NE_URL + "raster/NE1_LR_LC.zip"
    if name == ETOPO:
        return ETOPO_URL
    if name == ETOPO_BED:
        return ETOPO_BED_URL
    if name.startswith(ETOPO15_DIR + "/"):
        return (ETOPO15_BED_URL if name.endswith("_bed.tif") else ETOPO15_URL) + name.split("/", 1)[1]
    return NE_URL + ("physical/" if name in PHYSICAL else "cultural/") + name


def fetch(name: str) -> Path:
    """Path of data file `name` in CACHE, downloading it first (once the user agrees) if it isn't there yet."""
    paths = fetch_all([name])
    assert paths is not None
    return paths[0]


def fetch_all(names: list[str], optional: bool = False) -> list[Path] | None:
    """Paths of the data files `names` in CACHE, downloading the missing ones first, asking once for them all.
    With `optional`, None if the user declines."""
    missing = [name for name in names if not (CACHE / name).exists()]
    if missing:
        folders = {(CACHE / name).parent for name in missing}
        folder = folders.pop() if len(folders) == 1 else CACHE
        if not confirm_download([url_of(name) for name in missing], folder, optional):
            return None
        for name in missing:
            download(name)
    return [CACHE / name for name in names]


def download(name: str) -> None:
    path = CACHE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = url_of(name)
    print("downloading", url)
    part = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(url, part)
    if name == RASTER:
        with zipfile.ZipFile(part) as z:
            z.extractall(path.parent)
        os.remove(part)
    else:
        os.replace(part, path)


def is_labeled(row: pd.Series) -> bool:
    """Whether a country feature gets a name label (and counts as a 'real' country for across-water colouring).
    Dependencies are labelled only with --label territories (labels/regions.py)."""
    if is_dependency(row):
        return False
    return row.NAME_EN in LABELED_EXTRA or (row.TYPE in LABELED_TYPES and row.NAME_EN not in NOT_LABELED)


def is_dependency(row: pd.Series) -> bool:
    """A territory of another country (Greenland, Puerto Rico, Hong Kong...), coloured as part of its parent.
    Leases (Baikonur) and indeterminate areas (Palestine, Siachen) are not dependencies."""
    return row.SOVEREIGNT != row.ADMIN and row.TYPE not in NOT_DEPENDENCY_TYPES


def parent_index(countries: gpd.GeoDataFrame) -> dict[int, int]:
    """Each dependency's row index -> its parent's (the feature whose ADMIN is the dependency's SOVEREIGNT)."""
    index_of = dict(zip(countries.ADMIN, countries.index))
    parents = {}
    for i, row in countries.iterrows():
        if is_dependency(row):
            if row.SOVEREIGNT not in index_of:
                raise ValueError(f"no parent feature {row.SOVEREIGNT!r} for dependency {row.NAME_EN!r}")
            parents[i] = index_of[row.SOVEREIGNT]
    return parents


def km2(geoms: Sequence[BaseGeometry]) -> np.ndarray:
    """Each lon/lat geometry's area in km2."""
    return gpd.GeoSeries(geoms, crs="EPSG:4326").to_crs(EQUAL_AREA_CRS).area.to_numpy() / 1e6


def with_km2(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """`gdf` (lon/lat) with a `km2` column: each row's whole area, which projecting and clipping leave as it is."""
    gdf["km2"] = km2(gdf.geometry.values)
    return gdf


def load_countries() -> gpd.GeoDataFrame:
    return with_km2(gpd.read_file(fetch(COUNTRIES)).reset_index(drop=True))


def load_disputed() -> gpd.GeoDataFrame:
    """Natural Earth's disputed areas (Kashmir, Crimea, Taiwan...), see disputed.py."""
    return with_km2(gpd.read_file(fetch(DISPUTED)).reset_index(drop=True))


def load_states() -> gpd.GeoDataFrame:
    """State/province border lines of the STATE_COUNTRIES."""
    states = gpd.read_file(fetch(STATES))
    return states[states.ADM0_NAME.isin(STATE_COUNTRIES)]


def load_lakes() -> gpd.GeoDataFrame:
    """Every lake but the usually-dry DRY_LAKES (render.drawable_lakes keeps those big enough to see)."""
    lakes = with_km2(gpd.read_file(fetch(LAKES)))
    return lakes[~lakes.name.isin(DRY_LAKES)]


def load_rivers() -> gpd.GeoDataFrame:
    """Rivers, canals and their centrelines through lakes, of every scalerank (their labels decide which are drawn)."""
    return gpd.read_file(fetch(RIVERS))[["name", "name_en", "featurecla", "scalerank", "geometry"]]


def load_marine() -> gpd.GeoDataFrame:
    """Natural Earth's marine label areas (oceans, seas, gulfs, straits...), unfiltered: see labels/water.py."""
    return with_km2(gpd.read_file(fetch(MARINE))[["name", "featurecla", "scalerank", "geometry"]])


def load_ice_shelves() -> gpd.GeoDataFrame:
    """Antarctica's floating ice shelves. The islands Natural Earth cuts out of them are ice-covered too, so the
    shelves lose their holes."""
    raw = gpd.read_file(fetch(ICE_SHELVES))
    shelves = shapely.polygons(shapely.get_exterior_ring(shapely.get_parts(raw.geometry)))
    return gpd.GeoDataFrame(geometry=shelves, crs=raw.crs)


def load_ice() -> gpd.GeoDataFrame:
    """Glaciers and ice sheets, plus the ice shelves (--type topographic)."""
    parts = [gpd.read_file(fetch(GLACIERS))[["geometry"]], load_ice_shelves()]
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs)


def load_land() -> gpd.GeoDataFrame:
    """Natural Earth's land polygons (no lakes cut out; the Caspian is water)."""
    return gpd.read_file(fetch(LAND_OUTLINE))[["geometry"]]


def load_regions() -> gpd.GeoDataFrame:
    """Natural Earth's land label areas (ranges, plateaus, deserts, plains, islands...), unfiltered: see
    labels/land.py. The file's columns are upper case; these are lower-cased like the marine file's."""
    regions = gpd.read_file(fetch(REGIONS))[["FEATURECLA", "NAME", "NAME_EN", "SCALERANK", "geometry"]]
    return with_km2(regions.rename(columns={"FEATURECLA": "featurecla", "NAME": "name", "NAME_EN": "name_en", "SCALERANK": "scalerank"}))
