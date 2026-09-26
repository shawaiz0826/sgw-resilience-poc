"""C1 features (PRD C1: at most four) and training rows (PLAN §5B, PRD R5).

Row = (county, storm, advisory) inside the decision horizon (advisories issued between landfall and
T-78h). Features: p64, p34 (area-weighted mean of the cumulative 120-h probability over the county),
customers (EAGLE-I MCC), coastal flag. Target: the storm's peak customers out as a fraction of customers,
paired with every advisory of that storm (no interpolation across the 15-min / 6-h cadence gap).
"""
from __future__ import annotations

import pandas as pd

from src.pull.common import PROCESSED, STORMS

FEATURES = ["p64", "p34", "customers", "coastal"]
MONOTONE = [1, 1, 0, 0]  # output may only rise as p64 or p34 rises
HORIZON_H = (0.0, 78.0)  # hours before landfall kept as training rows

# Florida counties touching the Gulf or the Atlantic (35 of 67), hand-coded.
COASTAL_FIPS = {
    "12033", "12113", "12091", "12131", "12005", "12045", "12037", "12129", "12065", "12123", "12029", "12075",
    "12017", "12053", "12101", "12103", "12057", "12081", "12115", "12015", "12071", "12021", "12087", "12086",
    "12011", "12099", "12085", "12111", "12061", "12009", "12127", "12035", "12109", "12031", "12089",
}


def county_frame(storm_ids: list[str] | None = None) -> pd.DataFrame:
    """Every (storm, advisory, county) with features and, where observed, the target."""
    adv = pd.read_parquet(PROCESSED / "advisory_county.parquet")
    out = pd.read_parquet(PROCESSED / "outages_county.parquet")
    if storm_ids:
        adv = adv[adv["storm_id"].isin(storm_ids)]
    df = adv.merge(out[["storm_id", "fips", "customers", "peak_out_net", "frac_out"]], on=["storm_id", "fips"], how="left")
    df["coastal"] = df["fips"].isin(COASTAL_FIPS).astype(int)
    return df


def training_rows(storm_ids: list[str]) -> pd.DataFrame:
    df = county_frame(storm_ids)
    lo, hi = HORIZON_H
    return df[(df["hours_to_landfall"] > lo) & (df["hours_to_landfall"] <= hi)].reset_index(drop=True)


def advisory_at_lead(storm_id: str, lead_hours: float) -> pd.Timestamp:
    """Latest advisory issued at or before T-lead (e.g. T-72h for Decision A, T-12h for Decision B)."""
    adv = pd.read_parquet(PROCESSED / "advisories.parquet")
    a = adv[(adv["storm_id"] == storm_id) & (adv["product"] == "wsp64") & (adv["hours_to_landfall"] >= lead_hours)]
    return a["advisory_time"].max()


def train_storms() -> list[str]:
    return [k for k, s in STORMS.items() if s["role"] == "train"]


def holdout_storms() -> list[str]:
    return [k for k, s in STORMS.items() if s["role"] == "holdout"]
