"""Observed outcome per (storm, county) from EAGLE-I — the C1 target and the Milton backtest truth.

Pairing rule (R5): each advisory is paired with the storm's peak customers out per county; nothing is
interpolated. Peak = max 15-min customers_out in [landfall - 24 h, landfall + 96 h], net of a
pre-storm baseline (median of the first 24 h of the pulled window). Units: customers (meters), R7.

Denominator: MCC (modeled customers per county) published with the EAGLE-I Sci Data release. The 2024
file's total_customers is kept for reference; it overstates some counties (Miami-Dade 1.9x MCC).
Writes processed/outages_county.parquet and processed/outage_series_study.parquet (hourly, Lee+Charlotte).
"""
from __future__ import annotations

import pandas as pd

from src.pull.common import PROCESSED, RAW, STORMS, STUDY_FIPS


def load_mcc() -> pd.Series:
    m = pd.read_csv(RAW / "eaglei" / "mcc.csv", encoding="utf-8-sig")
    m["fips"] = m["County_FIPS"].astype(str).str.zfill(5)
    return m.set_index("fips")["Customers"].astype(int)


def load_storm(storm_id: str) -> pd.DataFrame:
    d = pd.read_csv(RAW / "eaglei" / f"fl_{storm_id}.csv.gz", dtype={"fips_code": str})
    d["fips"] = d["fips_code"].str.zfill(5)
    d["t"] = pd.to_datetime(d["run_start_time"]).dt.tz_localize("UTC")
    return d


def main() -> None:
    mcc = load_mcc()
    eaglei_2024 = load_storm("milton").groupby("fips")["total_customers"].max()
    rows, series = [], []
    for storm_id, s in STORMS.items():
        d = load_storm(storm_id)
        lf = pd.Timestamp(s["landfall"])
        t0 = d["t"].min()
        base = d[d["t"] < t0 + pd.Timedelta("24h")].groupby("fips")["customers_out"].median()
        win = d[(d["t"] >= lf - pd.Timedelta("24h")) & (d["t"] <= lf + pd.Timedelta("96h"))]
        g = win.groupby("fips")["customers_out"]
        peak, peak_t = g.max(), win.loc[g.idxmax(), ["fips", "t"]].set_index("fips")["t"]
        for fips in mcc.index[mcc.index.str.startswith("12")]:
            p = float(peak.get(fips, 0.0))
            b = float(base.get(fips, 0.0))
            net = max(p - b, 0.0)
            rows.append({
                "storm_id": storm_id, "fips": fips, "customers": int(mcc[fips]),
                "customers_eaglei_2024": int(eaglei_2024.get(fips, 0)) or None,
                "baseline_out": b, "peak_out": p, "peak_out_net": net,
                "frac_out": min(net / mcc[fips], 1.0), "peak_time": peak_t.get(fips, pd.NaT),
                "observed": fips in peak.index,
            })
        st = d[d["fips"].isin(STUDY_FIPS)].set_index("t").groupby("fips")["customers_out"].resample("1h").max()
        series.append(st.reset_index().assign(storm_id=storm_id))
        print(f"[outages] {storm_id}: {int(peak.gt(0).sum())} counties with outages; "
              f"Lee peak {int(peak.get('12071', 0)):,}, Charlotte {int(peak.get('12015', 0)):,}")
    pd.DataFrame(rows).to_parquet(PROCESSED / "outages_county.parquet", index=False)
    pd.concat(series).to_parquet(PROCESSED / "outage_series_study.parquet", index=False)


if __name__ == "__main__":
    main()
