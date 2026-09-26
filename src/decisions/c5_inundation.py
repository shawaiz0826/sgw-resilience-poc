"""C5 — substation inundation flag, Decision B (PRD C5, FR18-FR21). A threshold on a probabilistic input.

- Screening set: substations whose FEMA zone is in config screening_zones (fixed in Phase 0).
- Probability: P-Surge at the site for the advisory (prob_source=PSURGE) when a snapshot exists; else the
  STATIC fallback from the FEMA zone, bumped when the NAVD88 BFE is above the switchgear (prob_source=STATIC).
- Switchgear height: never in the public registry, so always ESTIMATED = ground (3DEP) + height_offset_m.
  P-Surge is "surge above ground", so the offset maps to the P-Surge threshold at or below it (conservative).
- Threshold in force = decisionB_threshold + decisionB_margin while the height is ESTIMATED.
- Recommend DE-ENERGIZE if prob >= threshold and no critical load on the feed is left without backup; the
  prototype knows no backup status, so any hospital or pumping station on the feed forces WATCH (A18).
Pure functions: config and tables in, DataFrame out.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FT = 0.3048
CRITICAL_TYPES = ("hospital", "pumping")


def psurge_threshold_ft(height_offset_m: float) -> int:
    """Largest whole-foot P-Surge threshold at or below the switchgear offset (1.0 m -> 3 ft)."""
    return max(1, math.floor(height_offset_m / FT + 1e-9))


def screening_set(registry: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    subs = registry[registry["type"] == "substation"]
    return subs[subs["fema_zone"].isin(cfg["screening_zones"])].copy()


def static_probability(row: pd.Series, cfg: dict, switchgear_m: float) -> tuple[float, str]:
    sp = cfg["static_prob"]
    zone = row["fema_zone"]
    bfe = row.get("bfe_m_navd88")
    if zone == "AE" and pd.notna(bfe) and bfe > switchgear_m:
        return sp["AE_bfe_above_switchgear"], "AE, BFE above switchgear"
    if zone in sp:
        note = "BFE has no usable datum" if zone == "AE" and pd.notna(row.get("bfe_ft")) and pd.isna(bfe) else zone
        return sp[zone], note
    return sp["other"], "other zone"


def inundation(registry: pd.DataFrame, psurge_sites: pd.DataFrame | None, storm_id: str,
               advisory_time: pd.Timestamp, cfg: dict, screen: bool = True) -> pd.DataFrame:
    """Probability of inundation above switchgear for each substation (screening set by default)."""
    subs = screening_set(registry, cfg) if screen else registry[registry["type"] == "substation"].copy()
    offset = float(cfg["height_offset_m"])
    n_ft = psurge_threshold_ft(offset)
    ps = pd.Series(dtype=float)
    if psurge_sites is not None and len(psurge_sites):
        snap = psurge_sites[(psurge_sites["storm_id"] == storm_id) & (psurge_sites["advisory_time"] == advisory_time)
                            & (psurge_sites["threshold_ft"] == n_ft)]
        ps = snap.set_index("asset_id")["prob"]
    rows = []
    for _, r in subs.iterrows():
        switchgear = r["ground_elev_m"] + offset if pd.notna(r["ground_elev_m"]) else np.nan
        if r["asset_id"] in ps.index:
            prob, source, basis = float(ps[r["asset_id"]]), "PSURGE", f"P(surge > {n_ft} ft above ground)"
        else:
            prob, basis = static_probability(r, cfg, switchgear)
            source = "STATIC"
        rows.append({
            "asset_id": r["asset_id"], "name": r["name"], "lon": r["lon"], "lat": r["lat"], "fema_zone": r["fema_zone"],
            "ground_elev_m": r["ground_elev_m"], "switchgear_m_navd88": round(switchgear, 3) if pd.notna(switchgear) else None,
            "height_source": "ESTIMATED", "height_offset_m": offset, "prob": round(prob, 4), "prob_source": source,
            "prob_basis": basis, "psurge_threshold_ft": n_ft if source == "PSURGE" else None,
        })
    return pd.DataFrame(rows)


def threshold_in_force(cfg: dict, height_source: str = "ESTIMATED") -> float:
    t = float(cfg["decisionB_threshold"])
    return round(t + float(cfg["decisionB_margin"]), 4) if height_source == "ESTIMATED" else t


def critical_loads(registry: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Hospitals and pumping stations on a plausible (placeholder) feed; same distance rule as C4."""
    max_km = float(cfg["placeholder_feed_max_km"])
    dep = registry[registry["type"].isin(CRITICAL_TYPES) & registry["feed_asset_id"].notna()
                   & (registry["feed_distance_km"] <= max_km)]
    return dep[["asset_id", "type", "name", "feed_asset_id", "feed_distance_km"]]


def recommend(flags: pd.DataFrame, registry: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add threshold, critical-load check and DE-ENERGIZE / WATCH to the output of inundation()."""
    crit = critical_loads(registry, cfg)
    out = flags.copy()
    out["threshold"] = [threshold_in_force(cfg, h) for h in out["height_source"]]
    out["above_threshold"] = out["prob"] >= out["threshold"]
    loads = crit.groupby("feed_asset_id").apply(
        lambda g: [f"{t}:{a}" for t, a in zip(g["type"], g["asset_id"])], include_groups=False)
    out["critical_loads"] = out["asset_id"].map(loads).apply(lambda v: v if isinstance(v, list) else [])
    out["critical_load_check"] = np.where(out["critical_loads"].str.len() > 0, "critical load, backup unknown",
                                          "no critical load on feed")
    rec, reason = [], []
    for _, r in out.iterrows():
        if not r["above_threshold"]:
            rec.append("WATCH")
            reason.append(f"prob {r['prob']:.2f} below threshold {r['threshold']:.2f}")
        elif r["critical_loads"]:
            rec.append("WATCH")
            reason.append(f"prob {r['prob']:.2f} >= {r['threshold']:.2f} but critical load, backup unknown "
                          f"({', '.join(r['critical_loads'])})")
        else:
            rec.append("DE-ENERGIZE")
            reason.append(f"prob {r['prob']:.2f} >= {r['threshold']:.2f}, no critical load on feed")
    out["recommendation"], out["reason"] = rec, reason
    out["sensor_value"] = None  # FR21: no historian in the prototype; blank and marked blank
    return out
