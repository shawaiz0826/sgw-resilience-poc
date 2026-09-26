"""Second truth source for Decision B (FR22): manual inspection of NOAA NGS post-Ian aerial imagery.

`python -m src.models.imagery_truth` creates data/processed/imagery_truth_ian.csv once (it never overwrites):
one row per substation in the screening set or flagged outside it by P-Surge at Ian T-12h (R3). A person fills
`flooded` with Y, N or UNCLEAR by opening `lookup` (the NGS Ian viewer, centred on the site: #zoom/lat/lon).
The backtest adds an "imagery_manual" truth variant as soon as any row says Y or N.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.decisions import c5_inundation as c5
from src.models.features import advisory_at_lead
from src.pull.common import PROCESSED

PATH = PROCESSED / "imagery_truth_ian.csv"
VIEWER = "https://storms.ngs.noaa.gov/storms/ian/index.html#17/{lat:.5f}/{lon:.5f}"
COLUMNS = ["asset_id", "name", "lon", "lat", "fema_zone", "flooded", "note", "lookup"]


def create() -> pd.DataFrame:
    if PATH.exists():
        raise FileExistsError(f"{PATH} exists; it holds manual labels and is never overwritten")
    cfg = load_config()
    reg = pd.read_parquet(PROCESSED / "registry.parquet")
    ps = pd.read_parquet(PROCESSED / "psurge_sites.parquet")
    t12 = advisory_at_lead("ian", cfg["decisionB_lead_hours"])
    allf = c5.inundation(reg, ps, "ian", t12, cfg, screen=False)
    in_set = allf["fema_zone"].isin(cfg["screening_zones"])
    flagged = ~in_set & (allf["prob"] >= c5.threshold_in_force(cfg))
    rows = allf[in_set | flagged].copy()
    rows["note"] = ["screening set" if s else f"outside screening set; P-Surge {p:.2f} at Ian T-12h"
                    for s, p in zip(in_set[in_set | flagged], rows["prob"])]
    rows["flooded"] = ""
    rows["lookup"] = [VIEWER.format(lat=la, lon=lo) for la, lo in zip(rows["lat"], rows["lon"])]
    out = rows[COLUMNS].sort_values("asset_id").reset_index(drop=True)
    out.to_csv(PATH, index=False)
    return out


def labelled() -> pd.DataFrame:
    """Rows labelled Y or N (UNCLEAR and blanks excluded), with observed_flooded as bool."""
    if not PATH.exists():
        return pd.DataFrame(columns=["asset_id", "observed", "observed_flooded"])
    d = pd.read_csv(PATH, dtype=str).fillna("")
    d["flooded"] = d["flooded"].str.strip().str.upper()
    d = d[d["flooded"].isin(["Y", "N"])]
    return pd.DataFrame({"asset_id": d["asset_id"], "observed": True, "observed_flooded": d["flooded"].eq("Y")})


if __name__ == "__main__":
    t = create()
    print(f"[imagery_truth] wrote {len(t)} rows to {PATH} (flooded left blank for manual labelling)")
