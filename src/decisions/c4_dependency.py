"""C4 — dependent assets inherit their feeding substation's exposure (PRD C4, FR3, FR15). A lookup, no ML.

Prototype feed = nearest substation by straight line (PRD Appendix A), labelled PLACEHOLDER FEED; beyond
placeholder_feed_max_km there is no plausible feed and the asset is flagged, never guessed (R10).
Production reads the recorded feed from the Phase 0 registry and never assigns by distance.
"""
from __future__ import annotations

import pandas as pd

DEPENDENT_TYPES = ("pumping", "plant", "hospital")


def dependencies(registry: pd.DataFrame, exposure: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    max_km = float(cfg["placeholder_feed_max_km"])
    dep = registry[registry["type"].isin(DEPENDENT_TYPES)].copy()
    subs = exposure[exposure["type"] == "substation"].set_index("asset_id")
    names = registry.set_index("asset_id")["name"]
    ok = dep["feed_asset_id"].notna() & (dep["feed_distance_km"] <= max_km)
    dep["feed_kind"] = ok.map({True: "PLACEHOLDER", False: "NONE"})
    dep["feed_asset_id"] = dep["feed_asset_id"].where(ok)
    dep["feed_name"] = dep["feed_asset_id"].map(names)
    dep["feed_label"] = dep["feed_kind"].map({"PLACEHOLDER": "PLACEHOLDER FEED",
                                              "NONE": f"no plausible feed (> {max_km:g} km)"})
    dep["inherited_score"] = dep["feed_asset_id"].map(subs["score"])
    dep["inherited_cell_prob"] = dep["feed_asset_id"].map(subs["cell_prob"])
    dep["feed_rank"] = dep["feed_asset_id"].map(subs["rank"])
    dep["critical_load"] = dep["type"].isin(["hospital", "pumping"])
    dep["backup_status"] = "unknown (treated as no backup, A18)"
    dep = dep.sort_values(["inherited_score", "asset_id"], ascending=[False, True], na_position="last")
    keep = ["asset_id", "type", "name", "county_fips", "lon", "lat", "feed_asset_id", "feed_name", "feed_distance_km",
            "feed_kind", "feed_label", "feed_rank", "inherited_score", "inherited_cell_prob", "critical_load",
            "backup_status", "source", "source_vintage"]
    return dep[keep].reset_index(drop=True)
