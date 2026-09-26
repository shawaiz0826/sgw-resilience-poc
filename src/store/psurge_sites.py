"""P-Surge at substation sites (FR6, C5) — samples every archived Ian P-Surge snapshot at each substation.

Writes processed/psurge_sites.parquet (storm, advisory_time, asset_id, threshold_ft, prob) and adds the
P-Surge snapshots to processed/advisories.parquet so the advisory store indexes every product it holds.
A substation outside every contour gets prob 0 only if the snapshot's contours reach the study area;
the app shows prob_source=PSURGE only where a snapshot exists for that advisory.
"""
from __future__ import annotations

import re
import zipfile

import geopandas as gpd
import pandas as pd

from src.pull.common import PROCESSED, RAW, STORMS

SRC = RAW / "psurge" / "ian"


def read_contours(zpath) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(zpath) as z:
        shp = next(n for n in z.namelist() if n.endswith(".shp"))
    g = gpd.read_file(f"zip://{zpath}!{shp}").set_crs(4326, allow_override=True)
    col = next(c for c in g.columns if c.lower().startswith("psurge"))
    g["prob"] = pd.to_numeric(g[col], errors="coerce").astype(float) / 100.0
    return g[["prob", "geometry"]]


def main() -> None:
    reg = pd.read_parquet(PROCESSED / "registry.parquet")
    subs = reg[reg["type"] == "substation"]
    pts = gpd.GeoDataFrame(subs[["asset_id"]], geometry=gpd.points_from_xy(subs.lon, subs.lat), crs=4326)
    landfall = pd.Timestamp(STORMS["ian"]["landfall"])
    rows, idx = [], []
    for z in sorted(SRC.glob("*.zip")):
        m = re.match(r"al092022_psurge(\d+)_(\d{10})\.zip", z.name)
        n_ft, ts = int(m.group(1)), pd.Timestamp(pd.to_datetime(m.group(2), format="%Y%m%d%H"), tz="UTC")
        c = read_contours(z)
        j = gpd.sjoin(pts, c, how="left", predicate="within").groupby("asset_id")["prob"].max().fillna(0.0)
        h = round((landfall - ts).total_seconds() / 3600, 2)
        for aid, p in j.items():
            rows.append({"storm_id": "ian", "advisory_time": ts, "hours_to_landfall": h, "asset_id": aid,
                         "threshold_ft": n_ft, "threshold_m": round(n_ft * 0.3048, 3), "prob": round(float(p), 4)})
        idx.append({"storm_id": "ian", "advisory_time": ts, "hours_to_landfall": h, "resolution": "SLOSH contours",
                    "product": f"psurge_{n_ft}ft", "source_file": f"psurge/ian/{z.name}"})
    out = pd.DataFrame(rows).sort_values(["advisory_time", "threshold_ft", "asset_id"])
    out.to_parquet(PROCESSED / "psurge_sites.parquet", index=False)
    adv = pd.read_parquet(PROCESSED / "advisories.parquet")
    adv = pd.concat([adv[~adv["product"].str.startswith("psurge")], pd.DataFrame(idx)], ignore_index=True)
    adv.sort_values(["storm_id", "advisory_time", "product"]).to_parquet(PROCESSED / "advisories.parquet", index=False)
    print(f"[psurge_sites] {out.advisory_time.nunique()} cycles x {out.threshold_ft.nunique()} thresholds x "
          f"{out.asset_id.nunique()} substations = {len(out)} rows")


if __name__ == "__main__":
    main()
