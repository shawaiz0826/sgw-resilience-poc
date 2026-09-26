"""Advisory store (PLAN §5A, FR5, R4).

Reads every archived wind speed probability snapshot (raw/wsp/<storm>/*.zip), converts contour bands
to probabilities (band midpoint), and writes:
  processed/advisories.parquet       one row per (storm, advisory_time, product): resolution, source file
  processed/advisory_county.parquet  area-weighted mean p34/p64 for every Florida county per advisory
  processed/advisory_cells.parquet   p34/p64 at the centre of each 0.1-degree cell over SW Florida (FR12)
Snapshots are never overwritten in raw/; processed tables are rebuilt deterministically from them.
"""
from __future__ import annotations

import re
import zipfile
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from src.pull.common import PROCESSED, RAW, STORMS

EQUAL_AREA = 5070  # CONUS Albers, for area weighting
FL_CLIP = box(-87.8, 24.3, -79.8, 31.1)
# 0.1-degree cells over SW Florida: study counties plus the placeholder staging sites.
REGION = (-83.2, 25.8, -80.8, 28.2)
CELL = 0.1
PRODUCTS = {"wsp34": "34knt", "wsp64": "64knt"}


def band_mid(label: str) -> float:
    """'<5%' -> 0.025, '10-20%' -> 0.15, '>90%' -> 0.95."""
    s = label.strip().replace("%", "")
    if s.startswith("<"):
        return float(s[1:]) / 200
    if s.startswith(">"):
        return (float(s[1:]) + 100) / 200
    lo, hi = s.split("-")
    return (float(lo) + float(hi)) / 200


def read_bands(zpath, knots: str) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(zpath) as z:
        shp = next(n for n in z.namelist() if n.endswith(".shp") and f"wsp{knots}" in n)
    g = gpd.read_file(f"zip://{zpath}!{shp}")
    # NHC ships these on an authalic sphere; the offset from WGS84 is far below the 5 km grid.
    g = g.set_crs(4326, allow_override=True)
    g = g[g.intersects(FL_CLIP)].copy()
    g["geometry"] = g.geometry.intersection(FL_CLIP)
    g["p"] = g["PERCENTAGE"].astype(str).map(band_mid).astype(float)
    return g[["p", "geometry"]]


def cell_grid() -> gpd.GeoDataFrame:
    lons = np.round(np.arange(REGION[0] + CELL / 2, REGION[2], CELL), 3)
    lats = np.round(np.arange(REGION[1] + CELL / 2, REGION[3], CELL), 3)
    rows = [(f"c{lon:.2f}_{lat:.2f}", lon, lat) for lat in lats for lon in lons]
    df = pd.DataFrame(rows, columns=["cell_id", "lon", "lat"])
    return gpd.GeoDataFrame(df, geometry=[Point(x, y) for x, y in zip(df.lon, df.lat)], crs=4326)


def county_means(counties_ea: gpd.GeoDataFrame, bands: gpd.GeoDataFrame) -> pd.Series:
    inter = gpd.overlay(counties_ea[["fips", "geometry"]], bands.to_crs(EQUAL_AREA), how="intersection",
                        keep_geom_type=True)
    inter["w"] = inter.geometry.area * inter["p"]
    num = inter.groupby("fips")["w"].sum()
    area = counties_ea.set_index("fips").geometry.area
    return (num.reindex(area.index).fillna(0.0) / area).clip(0, 1)


def cell_values(cells: gpd.GeoDataFrame, bands: gpd.GeoDataFrame) -> pd.Series:
    j = gpd.sjoin(cells[["cell_id", "geometry"]], bands, how="left", predicate="within")
    return j.groupby("cell_id")["p"].max().reindex(cells.cell_id).fillna(0.0)


def parse_stamp(name: str) -> tuple[datetime, str]:
    m = re.match(r"(\d{10})_wsp_120hr(\w+)\.zip", name)
    return datetime.strptime(m.group(1), "%Y%m%d%H"), m.group(2)


def main() -> None:
    counties = gpd.read_file(RAW / "census" / "fl_counties.geojson")
    counties_ea = counties.to_crs(EQUAL_AREA)
    cells = cell_grid()
    idx_rows, county_rows, cell_rows = [], [], []
    for storm_id, s in STORMS.items():
        landfall = pd.Timestamp(s["landfall"])
        for z in sorted((RAW / "wsp" / storm_id).glob("*.zip")):
            t, res = parse_stamp(z.name)
            ts = pd.Timestamp(t, tz="UTC")
            h = round((landfall - ts).total_seconds() / 3600, 2)
            base = {"storm_id": storm_id, "advisory_time": ts, "hours_to_landfall": h, "resolution": res}
            vals_c, vals_g = {}, {}
            for prod, knots in PRODUCTS.items():
                bands = read_bands(z, knots)
                vals_c[prod] = county_means(counties_ea, bands)
                vals_g[prod] = cell_values(cells, bands)
                idx_rows.append({**base, "product": prod, "source_file": f"wsp/{storm_id}/{z.name}"})
            for fips in counties.fips:
                county_rows.append({**base, "fips": fips, "p34": float(vals_c["wsp34"][fips]),
                                    "p64": float(vals_c["wsp64"][fips])})
            cell_rows.append(pd.DataFrame({**base, "cell_id": cells.cell_id.values, "lon": cells.lon.values,
                                           "lat": cells.lat.values, "p34": vals_g["wsp34"].values,
                                           "p64": vals_g["wsp64"].values}))
            print(f"[advisory_store] {storm_id} {t:%Y-%m-%d %HZ} (T{-h:+.0f}h) {res}", flush=True)
    pd.DataFrame(idx_rows).to_parquet(PROCESSED / "advisories.parquet", index=False)
    pd.DataFrame(county_rows).round({"p34": 5, "p64": 5}).to_parquet(PROCESSED / "advisory_county.parquet", index=False)
    pd.concat(cell_rows).round({"p34": 5, "p64": 5}).to_parquet(PROCESSED / "advisory_cells.parquet", index=False)
    print(f"[advisory_store] {len(idx_rows)} snapshots, {len(county_rows)} county rows")


if __name__ == "__main__":
    main()
