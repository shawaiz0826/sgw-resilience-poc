"""Shared inputs for the decision layer: processed tables (read-only, cached), advisory lookup, C1 county
estimates, cell lookup, and the canonical JSON / hash used to prove a record replays identically."""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from src.models.features import COASTAL_FIPS, county_frame
from src.pull.common import PROCESSED, STORMS, STUDY_COUNTIES, STUDY_FIPS

REGION_LON0, REGION_LAT0, CELL = -83.2, 25.8, 0.1  # must match src/store/advisory_store.py
UNITS = "customers (meters)"
EXPOSURE_LABEL = "exposure ranking, county-validated"


@lru_cache(maxsize=None)
def table(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


@lru_cache(maxsize=None)
def county_names() -> dict:
    import geopandas as gpd
    return gpd.read_parquet(PROCESSED / "counties.parquet").set_index("fips")["county"].to_dict()


def advisories(storm_id: str) -> pd.DataFrame:
    a = table("advisories")
    a = a[(a["storm_id"] == storm_id) & (a["product"] == "wsp64")]
    return a.sort_values("advisory_time").reset_index(drop=True)


def parse_advisory(storm_id: str, when: str | pd.Timestamp) -> pd.Timestamp:
    """Accept an ISO time or 'T-72h' (latest advisory issued at or before that lead)."""
    adv = advisories(storm_id)
    if isinstance(when, str) and when.upper().startswith("T-"):
        lead = float(when[2:].rstrip("hH"))
        return adv[adv["hours_to_landfall"] >= lead]["advisory_time"].max()
    t = pd.Timestamp(when)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    if t not in set(adv["advisory_time"]):
        raise ValueError(f"no {storm_id} advisory at {t}; available: {[str(x) for x in adv['advisory_time']]}")
    return t


def advisory_meta(storm_id: str, t: pd.Timestamp) -> dict:
    a = table("advisories")
    rows = a[(a["storm_id"] == storm_id) & (a["advisory_time"] == t)]
    wsp = rows[rows["product"].str.startswith("wsp")]
    return {
        "storm_id": storm_id, "storm_name": STORMS[storm_id]["name"], "advisory_time": t.isoformat(),
        "hours_to_landfall": float(wsp["hours_to_landfall"].iloc[0]), "resolution": str(wsp["resolution"].iloc[0]),
        "snapshot_files": sorted(set(rows["source_file"])),
    }


def cell_id_for(lon: float, lat: float) -> str:
    i, j = math.floor((lon - REGION_LON0) / CELL), math.floor((lat - REGION_LAT0) / CELL)
    return f"c{round(REGION_LON0 + (i + 0.5) * CELL, 3):.2f}_{round(REGION_LAT0 + (j + 0.5) * CELL, 3):.2f}"


def cells_at(storm_id: str, t: pd.Timestamp) -> pd.DataFrame:
    c = table("advisory_cells")
    return c[(c["storm_id"] == storm_id) & (c["advisory_time"] == t)].set_index("cell_id")


def county_estimates(storm_id: str, t: pd.Timestamp, model) -> pd.DataFrame:
    """C1 at this advisory for every Florida county: fraction and customers out at P10/P50/P90."""
    f = county_frame([storm_id])
    f = f[f["advisory_time"] == t].reset_index(drop=True)
    p = model.predict(f)
    out = pd.concat([f[["fips", "customers", "coastal", "p34", "p64"]], p.rename(columns=lambda c: f"frac_{c}")], axis=1)
    for q in ("p10", "p50", "p90"):
        out[f"out_{q}"] = (out[f"frac_{q}"] * out["customers"]).round(0)
    out["county"] = out["fips"].map(county_names())
    out["study_zone"] = out["fips"].isin(STUDY_FIPS)
    return out.sort_values("fips").reset_index(drop=True)


def study_customers() -> dict:
    o = table("outages_county")
    return o.drop_duplicates("fips").set_index("fips")["customers"].to_dict()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if o is pd.NA or o is pd.NaT:
        return None
    return o


def canonical(obj) -> str:
    return json.dumps(_clean(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def output_hash(estimates: dict, recommendation: dict) -> str:
    return hashlib.sha256((canonical(estimates) + canonical(recommendation)).encode()).hexdigest()[:16]


def records(df: pd.DataFrame) -> list[dict]:
    return _clean(df.to_dict(orient="records"))


__all__ = ["COASTAL_FIPS", "STUDY_COUNTIES", "STUDY_FIPS", "UNITS", "EXPOSURE_LABEL"]
