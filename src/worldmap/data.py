"""Natural Earth inputs: download/cache, load, filter and project."""

import os
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import box

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
RIVERS = "ne_10m_rivers_lake_centerlines.zip"  # --rivers: centrelines, joined up through the lakes they cross
PHYSICAL = {LAKES, GLACIERS, ICE_SHELVES, LAND_OUTLINE, MARINE, RIVERS}  # under physical/ rather than cultural/
# ETOPO 2022 (NOAA, public domain): land heights and sea depths in metres, 60 arc-second grid, top of the ice.
ETOPO = "ETOPO_2022_v1_60s_N90W180_surface.tif"
ETOPO_URL = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_surface_elev_gtif/" + ETOPO

LAKE_MIN_KM2 = 2000  # lakes at least this big are drawn as water (~8 px across on the poster)
DRY_LAKES = {"Lake Eyre North", "Lake Torrens", "Lake Gairdner", "Lake Mackay", "Lake Frome", "Lake Barlee"}  # Australian salt pans, usually dry
RIVER_MAX_RANK = 8  # rivers of scalerank (0 = biggest) up to this are drawn; 9+ are short tributaries
EQUAL_AREA_CRS = "+proj=eqearth"  # for lake areas

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


def fetch(name: str) -> Path:
    """Path of Natural Earth file `name` in CACHE, downloading it first if it isn't there yet."""
    path = CACHE / name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == RASTER:
        url = NE_URL + "raster/NE1_LR_LC.zip"
    elif name == ETOPO:
        url = ETOPO_URL
    else:
        url = NE_URL + ("physical/" if name in PHYSICAL else "cultural/") + name
    print("downloading", url)
    part = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(url, part)
    if name == RASTER:
        with zipfile.ZipFile(part) as z:
            z.extractall(path.parent)
        os.remove(part)
    else:
        os.replace(part, path)
    return path


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


def load_countries() -> gpd.GeoDataFrame:
    return gpd.read_file(fetch(COUNTRIES)).reset_index(drop=True)


def load_disputed() -> gpd.GeoDataFrame:
    """Natural Earth's disputed areas (Kashmir, Crimea, Taiwan...), see disputed.py."""
    return gpd.read_file(fetch(DISPUTED)).reset_index(drop=True)


def load_states() -> gpd.GeoDataFrame:
    """State/province border lines of the STATE_COUNTRIES."""
    states = gpd.read_file(fetch(STATES))
    return states[states.ADM0_NAME.isin(STATE_COUNTRIES)]


def load_lakes() -> gpd.GeoDataFrame:
    """Lakes of at least LAKE_MIN_KM2, minus the usually-dry DRY_LAKES."""
    lakes = gpd.read_file(fetch(LAKES))
    big = lakes.to_crs(EQUAL_AREA_CRS).area >= LAKE_MIN_KM2 * 1e6
    return lakes[big & ~lakes.name.isin(DRY_LAKES)]


def load_rivers() -> gpd.GeoDataFrame:
    """Rivers (and their centrelines through lakes) of scalerank <= RIVER_MAX_RANK; canals left out."""
    rivers = gpd.read_file(fetch(RIVERS))[["name", "featurecla", "scalerank", "geometry"]]
    return rivers[(rivers.scalerank <= RIVER_MAX_RANK) & (rivers.featurecla != "Canal")]


def load_marine() -> gpd.GeoDataFrame:
    """Natural Earth's marine label areas (oceans, seas, gulfs, straits...), unfiltered: see labels/water.py."""
    return gpd.read_file(fetch(MARINE))[["name", "featurecla", "scalerank", "geometry"]]


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


def project(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip to the +-180/90 lon/lat box and project to CRS."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.intersection(box(-180, -90, 180, 90))
    return gdf.to_crs(CRS)
