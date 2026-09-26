"""C3 — per-asset exposure ranking (PRD C3, FR12, FR13). A rule on C1's output; the weights are not learned.

cell_prob = C1's P50 fraction for the asset's 0.1-degree cell: the cell's p64/p34 with the county's customers
and coastal flag. score = cell_prob x w_zone x w_elev x w_voltage (config c3_weights). Public attributes only
(PRD Appendix A); maintenance attributes enter in production where their completeness flag passes.
Two assets in the same cell with the same attributes get the same score. Label: "exposure ranking, county-validated".
"""
from __future__ import annotations

import pandas as pd

from src.decisions.common import EXPOSURE_LABEL, cell_id_for
from src.models.features import COASTAL_FIPS


def zone_weight(zone: str, subtype, w: dict) -> tuple[float, str]:
    if zone == "X" and isinstance(subtype, str) and subtype.startswith("0.2 Percent"):
        return w["X_shaded"], "X (0.2% shaded)"
    return w.get(zone, w["X"]), zone


def elev_weight(elev_m, bands: list[dict]) -> tuple[float, str]:
    if pd.isna(elev_m):
        return 1.0, "elevation UNKNOWN"
    for b in bands:
        if elev_m < b["max_m"]:
            return b["w"], f"< {b['max_m']} m"
    return 1.0, "n/a"


def exposure(registry: pd.DataFrame, cells: pd.DataFrame, customers: dict, model, cfg: dict) -> pd.DataFrame:
    w = cfg["c3_weights"]
    a = registry.copy()
    a["cell_id"] = [cell_id_for(x, y) for x, y in zip(a["lon"], a["lat"])]
    a["p64"] = a["cell_id"].map(cells["p64"]).fillna(0.0)
    a["p34"] = a["cell_id"].map(cells["p34"]).fillna(0.0)
    a["customers"] = a["county_fips"].map(customers)
    a["coastal"] = a["county_fips"].isin(COASTAL_FIPS).astype(int)
    a["cell_prob"] = model.predict(a[["p64", "p34", "customers", "coastal"]])["p50"].to_numpy()
    zw = [zone_weight(z, s, w["fema_zone"]) for z, s in zip(a["fema_zone"], a["fema_zone_subtype"])]
    ew = [elev_weight(e, w["elev_bands"]) for e in a["ground_elev_m"]]
    a["w_zone"], a["zone_basis"] = [x[0] for x in zw], [x[1] for x in zw]
    a["w_elev"], a["elev_basis"] = [x[0] for x in ew], [x[1] for x in ew]
    electric = a["type"].isin(["substation", "line"])
    a["w_voltage"] = [w["voltage"].get(v, 1.0) if e else 1.0 for v, e in zip(a["voltage_class"], electric)]
    a["score"] = (a["cell_prob"] * a["w_zone"] * a["w_elev"] * a["w_voltage"]).round(6)
    a = a.sort_values(["score", "asset_id"], ascending=[False, True]).reset_index(drop=True)
    a["rank"] = range(1, len(a) + 1)
    a["label"] = EXPOSURE_LABEL
    keep = ["rank", "asset_id", "type", "name", "county_fips", "lon", "lat", "cell_id", "p64", "p34", "cell_prob",
            "fema_zone", "zone_basis", "w_zone", "ground_elev_m", "elev_basis", "w_elev", "voltage_class", "w_voltage",
            "score", "source", "source_vintage", "label"]
    return a[keep]
