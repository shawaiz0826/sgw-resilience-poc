"""Run both decisions for one advisory and write the decision record (PRD "order of a run", FR7, FR25).

Order: C1 (county estimates) -> C2 (staging plan) -> C3 (exposure) -> C4 (dependents) = Decision A;
C5 = Decision B, whenever a P-Surge snapshot exists for the advisory (or, with no snapshot loaded, from
decisionB_starts_hours on the STATIC fallback). Everything goes into the record; the briefing (C6) reads it after.

  python -m src.decisions.run milton T-72h            # latest Milton advisory at or before T-72h
  python -m src.decisions.run ian 2022-09-28T06:00Z --show
  python -m src.decisions.run --replay <record_id>    # recompute from the record's inputs, compare output hash
"""
from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd
import yaml

from src.config import CONFIG_PATH, load_config
from src.decisions import c5_inundation as c5
from src.decisions.c2_staging import staging_plan
from src.decisions.c3_exposure import exposure
from src.decisions.c4_dependency import dependencies
from src.decisions.common import (EXPOSURE_LABEL, UNITS, advisory_meta, canonical, cells_at, cell_id_for,
                                  county_estimates, output_hash, parse_advisory, records, study_customers, table)
from src.models.c1 import active_model_kind, load_model, model_version
from src.pull.common import PROCESSED, STUDY_FIPS
from src.store import record as rec_store

PSURGE_NOT_LOADED = "STATIC (P-Surge grid not loaded)"


def _record_id(decision: str, storm_id: str, t: pd.Timestamp, inputs: dict, cfg_version: str, model_ver: str) -> str:
    key = hashlib.sha256((canonical(inputs) + cfg_version + model_ver).encode()).hexdigest()[:8]
    return f"{decision}-{storm_id}-{t:%Y%m%d%H}-{key}"


def banner_for(model_kind: str, cfg: dict) -> dict:
    """FR17: the banner state of the model version in use, from its backtest on the held-out storm."""
    b = json.loads((PROCESSED / "backtest.json").read_text())["fr16"][model_kind]
    on = b["top_n_match"] < cfg["banner_topn_match"] or b["p90_coverage"] < cfg["banner_p90_coverage"]
    return {"on": bool(on), "model_kind": model_kind, "top_n_match": b["top_n_match"], "p90_coverage": b["p90_coverage"],
            "targets": {"top_n_match": cfg["banner_topn_match"], "p90_coverage": cfg["banner_p90_coverage"]},
            "held_out_storm": "milton"}


def decision_a(storm_id: str, t: pd.Timestamp, cfg: dict, model_kind: str | None = None,
               own_crew_hours: float | None = None) -> dict:
    model_kind = model_kind or active_model_kind()
    model, mver = load_model(model_kind), model_version(model_kind)
    meta = advisory_meta(storm_id, t)
    est = county_estimates(storm_id, t, model)                                          # C1
    cells = cells_at(storm_id, t)
    site_p64 = {s["site_id"]: float(cells["p64"].get(cell_id_for(s["lon"], s["lat"]), 0.0)) for s in cfg["staging_sites"]}
    own = float(own_crew_hours) if own_crew_hours is not None else float(cfg["own_crew_hours_default"])
    own_src = "entered by P1" if own_crew_hours is not None else "PLACEHOLDER default, config own_crew_hours_default"
    zones = [{"zone_fips": r.fips, "zone_name": r.county, "out_p50": r.out_p50, "out_p90": r.out_p90}
             for r in est[est["study_zone"]].itertuples()]
    plan = staging_plan(zones, site_p64, own, cfg)                                       # C2
    reg = table("registry")
    expo = exposure(reg, cells, study_customers(), model, cfg)                           # C3
    deps = dependencies(reg, expo, cfg)                                                  # C4
    inputs = {**meta, "decision": "A", "model_kind": model_kind, "own_crew_hours": own, "own_crew_hours_source": own_src,
              "zones": list(STUDY_FIPS), "c1_features": ["p64", "p34", "customers", "coastal"]}
    estimates = {"units": UNITS, "counties": records(est), "assets": records(expo), "dependents": records(deps),
                 "exposure_label": EXPOSURE_LABEL}
    recommendation = {"plan": plan, "banner": banner_for(model_kind, cfg), "units": UNITS}
    return {
        "record_id": _record_id("A", storm_id, t, inputs, cfg["_version"], mver), "storm_id": storm_id,
        "advisory_time": meta["advisory_time"], "hours_to_landfall": meta["hours_to_landfall"],
        "resolution": meta["resolution"], "decision": "A", "config_version": cfg["_version"], "model_version": mver,
        "inputs": inputs, "estimates": estimates, "recommendation": recommendation,
        "crew_hours_entered": own, "crew_hours_source": own_src,
    }


def decision_b(storm_id: str, t: pd.Timestamp, cfg: dict) -> dict | None:
    meta = advisory_meta(storm_id, t)
    ps = table("psurge_sites")
    snap = ps[(ps["storm_id"] == storm_id) & (ps["advisory_time"] == t)]
    has_ps = len(snap) > 0
    h = meta["hours_to_landfall"]
    if not has_ps and not (0 < h <= float(cfg["decisionB_starts_hours"])):
        return None  # no P-Surge and outside the watch window: Decision B does not run, P2 sees nothing from C5
    reg = table("registry")
    flags = c5.recommend(c5.inundation(reg, snap if has_ps else None, storm_id, t, cfg), reg, cfg)
    srcs = set(flags["prob_source"])
    prob_source = "PSURGE" if srcs == {"PSURGE"} else ("STATIC" if srcs == {"STATIC"} else "MIXED")
    n_ft = c5.psurge_threshold_ft(float(cfg["height_offset_m"]))
    psurge_files = [f for f in meta["snapshot_files"] if f.startswith("psurge/") and f"psurge{n_ft}_" in f]
    inputs = {**meta, "decision": "B", "screening_zones": cfg["screening_zones"], "height_offset_m": cfg["height_offset_m"],
              "psurge_snapshot": psurge_files, "psurge_threshold_ft": n_ft if has_ps else None,
              "prob_source_note": "P-Surge as issued" if has_ps else PSURGE_NOT_LOADED}
    est_cols = ["asset_id", "name", "lon", "lat", "fema_zone", "ground_elev_m", "switchgear_m_navd88", "height_source",
                "height_offset_m", "prob", "prob_source", "prob_basis", "psurge_threshold_ft", "static_tier"]
    rec_cols = ["asset_id", "threshold", "above_threshold", "critical_loads", "critical_load_check", "recommendation",
                "escalation", "reason", "sensor_value"]
    counts = flags["recommendation"].value_counts().to_dict()
    recommendation = {"substations": records(flags[rec_cols]), "threshold_in_force": c5.threshold_in_force(cfg),
                      "threshold": cfg["decisionB_threshold"], "margin": cfg["decisionB_margin"],
                      "margin_applied_because": "switchgear height ESTIMATED", "counts": counts,
                      "prob_source": prob_source if has_ps else PSURGE_NOT_LOADED}
    estimates = {"substations": records(flags[est_cols]), "sensor_note": "no historian in the prototype; blank and marked blank (FR21)"}
    mver = "none (C5 is a rule on P-Surge; no fitted model)"
    return {
        "record_id": _record_id("B", storm_id, t, inputs, cfg["_version"], mver), "storm_id": storm_id,
        "advisory_time": meta["advisory_time"], "hours_to_landfall": h, "resolution": meta["resolution"], "decision": "B",
        "config_version": cfg["_version"], "model_version": mver, "inputs": inputs, "estimates": estimates,
        "recommendation": recommendation, "height_source": "ESTIMATED", "prob_source": prob_source,
        "sensor_value": None, "sensor_note": "BLANK: no flood sensor historian in the prototype (FR21)",
    }


def _finalise(r: dict) -> dict:
    r["output_hash"] = output_hash(r["estimates"], r["recommendation"])
    r["inputs_json"], r["estimates_json"], r["recommendation_json"] = (
        canonical(r["inputs"]), canonical(r["estimates"]), canonical(r["recommendation"]))
    return r


def run(storm_id: str, when, own_crew_hours: float | None = None, model_kind: str | None = None,
        conn=None, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    conn = conn or rec_store.connect()
    rec_store.save_config(conn, cfg["_version"], CONFIG_PATH.read_text())
    t = parse_advisory(storm_id, when)
    out = []
    for r in (decision_a(storm_id, t, cfg, model_kind, own_crew_hours), decision_b(storm_id, t, cfg)):
        if r is None:
            continue
        r = _finalise(r)
        created = rec_store.insert_record(conn, r)
        out.append({"record_id": r["record_id"], "decision": r["decision"], "output_hash": r["output_hash"],
                    "created": created, "record": r})
    return out


def replay(record_id: str, conn=None) -> dict:
    """NFR reproducibility: same advisory snapshot + same configuration version -> same output."""
    conn = conn or rec_store.connect()
    orig = rec_store.get_record(conn, record_id)
    if orig is None:
        raise KeyError(record_id)
    text = rec_store.load_config_text(conn, orig["config_version"])
    cfg = yaml.safe_load(text)
    cfg["_version"] = orig["config_version"]
    t = pd.Timestamp(orig["advisory_time"])
    inp = orig["inputs"]
    if orig["decision"] == "A":
        if model_version(inp["model_kind"]) != orig["model_version"]:
            return {"record_id": record_id, "identical": False,
                    "reason": f"model {orig['model_version']} is no longer the deployed {inp['model_kind']} version"}
        r = decision_a(orig["storm_id"], t, cfg, inp["model_kind"],
                       inp["own_crew_hours"] if inp["own_crew_hours_source"] == "entered by P1" else None)
    else:
        r = decision_b(orig["storm_id"], t, cfg)
    r = _finalise(r)
    return {"record_id": record_id, "original_hash": orig["output_hash"], "replay_hash": r["output_hash"],
            "identical": r["output_hash"] == orig["output_hash"], "config_version": orig["config_version"],
            "model_version": orig["model_version"]}


def _summary(r: dict) -> str:
    if r["decision"] == "A":
        p, b = r["recommendation"]["plan"], r["recommendation"]["banner"]
        zones = "; ".join(f"{z['zone_name']}: P50 {z['customers_out_p50']:,.0f} / P90 {z['customers_out_p90']:,.0f} out, "
                          f"site {z['assigned_site_name']}{' (DISPLACED from ' + z['mapped_site_name'] + ')' if z['displaced'] else ''}"
                          for z in p["zones"])
        return (f"A  {r['record_id']}  model {r['model_version']}  banner {'ON' if b['on'] else 'off'}\n"
                f"   {zones}\n   mutual aid request: {p['mutual_aid_crews']:,} crews ({p['mutual_aid_workers']:,} workers); "
                f"own crew-hours {p['own_crew_hours']:,.0f} [{r['crew_hours_source']}]")
    rc = r["recommendation"]
    return (f"B  {r['record_id']}  prob source {rc['prob_source']}  threshold in force {rc['threshold_in_force']}\n"
            f"   {rc['counts']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("storm", nargs="?")
    ap.add_argument("when", nargs="?", help="ISO time of an advisory, or T-<hours>h")
    ap.add_argument("--crew-hours", type=float, default=None, help="SGW's own available crew-hours (FR11)")
    ap.add_argument("--model", choices=["glm", "lgbm"], default=None)
    ap.add_argument("--show", action="store_true", help="print the records as JSON")
    ap.add_argument("--replay", metavar="RECORD_ID")
    a = ap.parse_args()
    if a.replay:
        print(json.dumps(replay(a.replay), indent=2))
        return
    for o in run(a.storm, a.when, a.crew_hours, a.model):
        print(("new " if o["created"] else "existing ") + _summary(o["record"]) + f"\n   output_hash {o['output_hash']}")
        if a.show:
            r = o["record"]
            print(json.dumps({k: r[k] for k in ("record_id", "decision", "storm_id", "advisory_time", "resolution",
                                               "config_version", "model_version", "output_hash")} |
                             {"inputs": r["inputs"], "recommendation": r["recommendation"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
