"""D0 — Census cartographic boundary counties (2023, 1:500k), Florida only."""
from __future__ import annotations

import geopandas as gpd

from .common import FL_STATEFP, RAW, download, write_pull_log

URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"


def main() -> None:
    z = download(URL, RAW / "census" / "cb_2023_us_county_500k.zip", timeout=300)
    gdf = gpd.read_file(f"zip://{z}")
    fl = gdf[gdf["STATEFP"] == FL_STATEFP][["GEOID", "NAME", "geometry"]].to_crs(4326)
    fl = fl.rename(columns={"GEOID": "fips", "NAME": "county"}).reset_index(drop=True)
    fl.to_file(RAW / "census" / "fl_counties.geojson", driver="GeoJSON")
    write_pull_log("D0", {"source": URL, "counties": len(fl), "vintage": "2023"})
    print(f"[counties] {len(fl)} Florida counties")


if __name__ == "__main__":
    main()
