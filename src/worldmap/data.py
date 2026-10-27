"""Natural Earth inputs: download/cache, load, filter and project."""

import os
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# Data files and caches live in the repo's .cache/ (found from this file, so it needs an editable install).
CACHE = Path(__file__).resolve().parent.parent.parent / ".cache"
NE_URL = "https://naciscdn.org/naturalearth/10m/"
COUNTRIES = "ne_10m_admin_0_countries.zip"
STATES = "ne_10m_admin_1_states_provinces_lines.zip"
LAKES = "ne_10m_lakes.zip"
RASTER = "ne1/NE1_LR_LC.tif"  # unzipped from NE1_LR_LC.zip

LAKE_MIN_KM2 = 2000  # lakes at least this big are drawn as water (~8 px across on the poster)
DRY_LAKES = {"Lake Eyre North", "Lake Torrens", "Lake Gairdner", "Lake Mackay", "Lake Frome", "Lake Barlee"}  # Australian salt pans, usually dry
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

NOT_LABELED = {"Jersey", "Guernsey", "Isle of Man", "Åland", "Macau", "Hong Kong", "Sint Maarten", "Aruba", "Curaçao", "Northern Cyprus", "Turkish Republic of Northern Cyprus", "Somaliland"}
LABELED_EXTRA = {"Palestine", "Kosovo", "Western Sahara", "Israel", "Puerto Rico", "Falkland Islands", "Faroe Islands"}
LABELED_TYPES = ("Sovereign country", "Country", "Sovereignty")


def fetch(name: str) -> Path:
    """Path of Natural Earth file `name` in CACHE, downloading it first if it isn't there yet."""
    path = CACHE / name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == RASTER:
        url = NE_URL + "raster/NE1_LR_LC.zip"
    else:
        url = NE_URL + ("physical/" if name == LAKES else "cultural/") + name
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
    """Whether a country feature gets a name label (and counts as a 'real' country for across-water colouring)."""
    return row.NAME_EN in LABELED_EXTRA or (row.TYPE in LABELED_TYPES and row.NAME_EN not in NOT_LABELED)


def load_countries() -> gpd.GeoDataFrame:
    return gpd.read_file(fetch(COUNTRIES)).reset_index(drop=True)


def load_states() -> gpd.GeoDataFrame:
    """State/province border lines of the STATE_COUNTRIES."""
    states = gpd.read_file(fetch(STATES))
    return states[states.ADM0_NAME.isin(STATE_COUNTRIES)]


def load_lakes() -> gpd.GeoDataFrame:
    """Lakes of at least LAKE_MIN_KM2, minus the usually-dry DRY_LAKES."""
    lakes = gpd.read_file(fetch(LAKES))
    big = lakes.to_crs(EQUAL_AREA_CRS).area >= LAKE_MIN_KM2 * 1e6
    return lakes[big & ~lakes.name.isin(DRY_LAKES)]


def project(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip to the +-180/90 lon/lat box and project to CRS."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.intersection(box(-180, -90, 180, 90))
    return gdf.to_crs(CRS)
