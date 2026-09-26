"""Backtest suite (FR16, FR17, FR22, FR30). Writes docs/backtest_report.md, data/processed/backtest.json and
the per-county / per-substation tables the app's Backtest tab reads.

FR16  C1 on Milton (never seen in training) at the latest advisory issued at or before T-72h, all 67
      Florida counties: MAE of customers out, Spearman rank correlation of county order, top-N match
      (N = config top_n), P90 coverage. Also every Milton advisory in the horizon (lead-time curve).
FR30  Selection on leave-one-storm-out between the two training storms (models/c1/loso_cv.json, written by
      c1_fit): the active model has the higher mean Spearman across the held-out training storms; ties go to
      the GLM. Milton is not an input to selection; it is reported once, as the holdout.
FR17  Banner flag for the active model, on the holdout: top-N match < banner_topn_match or P90 coverage <
      banner_p90_coverage.
FR22  Decision B on Ian at the latest advisory at or before T-12h: precision and recall of the P-Surge inundation
      flag and of DE-ENERGIZE against USGS high-water marks (observed flooded = an HWM within
      fr22_hwm_radius_m whose water surface is above the ESTIMATED switchgear elevation). The STATIC
      variant routes every site to P2's judgment with a tier, so it reports that count, not precision/recall.
Pairing rule (R5): every advisory is paired with the storm's peak customers out; nothing is interpolated.
"""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.stats import spearmanr

from src.config import ROOT, load_config
from src.decisions import c5_inundation as c5
from src.models.c1 import MODEL_DIR, load_model, model_version
from src.models.features import advisory_at_lead, county_frame
from src.pull.common import PROCESSED, STUDY_FIPS

UTM17 = 26917


def county_metrics(df: pd.DataFrame, top_n: int) -> dict:
    pred_out = df["p50"] * df["customers"]
    obs_out = df["frac_out"] * df["customers"]
    top_pred = set(pred_out.nlargest(top_n).index)
    top_obs = set(obs_out.nlargest(top_n).index)
    return {
        "counties": int(len(df)),
        "mae_customers": round(float((pred_out - obs_out).abs().mean()), 1),
        "mae_frac": round(float((df["p50"] - df["frac_out"]).abs().mean()), 4),
        "spearman": round(float(spearmanr(pred_out, obs_out).statistic), 3),
        "top_n": top_n,
        "top_n_match": round(len(top_pred & top_obs) / top_n, 3),
        "top_n_predicted": sorted(df.loc[list(top_pred), "fips"]),
        "top_n_observed": sorted(df.loc[list(top_obs), "fips"]),
        "p90_coverage": round(float((df["frac_out"] <= df["p90"] + 1e-9).mean()), 3),
        "p10_p90_coverage": round(float(((df["frac_out"] >= df["p10"] - 1e-9) & (df["frac_out"] <= df["p90"] + 1e-9)).mean()), 3),
    }


def loso() -> dict:
    """Leave-one-storm-out skill between the training storms: {kind: {held_out_storm: metrics}}."""
    cv = json.loads((MODEL_DIR / "loso_cv.json").read_text())
    out: dict = {}
    for key, v in cv.items():
        kind, held = key.split("_trained_without_")
        out.setdefault(kind, {})[held] = v
    return out


def select() -> tuple[str, str, dict]:
    cv = loso()
    means = {k: float(np.mean([v["spearman_frac"] for v in cv[k].values()])) for k in ("glm", "lgbm")}
    active = "glm" if means["glm"] >= means["lgbm"] else "lgbm"
    reason = (f"Leave-one-storm-out on the training storms (Ian, Idalia): mean Spearman GLM {means['glm']:.3f}, "
              f"LightGBM {means['lgbm']:.3f}; rule: the higher mean is active, ties to the GLM. Milton is not an "
              "input to selection; it is reported once as the holdout")
    return active, reason, means


def fr16(cfg: dict) -> dict:
    top_n = int(cfg["top_n"])
    frame = county_frame(["milton"])
    t72 = advisory_at_lead("milton", cfg["decisionA_lead_hours"])
    res, per_county, curve = {"advisory_time": str(t72)}, [], []
    at = frame[frame["advisory_time"] == t72].reset_index(drop=True)
    res["hours_to_landfall"] = float(at["hours_to_landfall"].iloc[0])
    for kind in ("lgbm", "glm"):
        model = load_model(kind)
        p = model.predict(at)
        df = pd.concat([at, p], axis=1)
        res[kind] = {"model_version": model_version(kind), **county_metrics(df, top_n)}
        per_county.append(df.assign(model=kind))
        for t, g in frame[(frame["hours_to_landfall"] > 0) & (frame["hours_to_landfall"] <= 120)].groupby("advisory_time"):
            g = g.reset_index(drop=True)
            gm = county_metrics(pd.concat([g, model.predict(g)], axis=1), top_n)
            curve.append({"model": kind, "advisory_time": t, "hours_to_landfall": float(g["hours_to_landfall"].iloc[0]),
                          **{k: gm[k] for k in ("mae_customers", "spearman", "top_n_match", "p90_coverage")}})
    active, reason, means = select()
    a = res[active]
    banner = a["top_n_match"] < cfg["banner_topn_match"] or a["p90_coverage"] < cfg["banner_p90_coverage"]
    res["selection"] = {"active": active, "model_version": res[active]["model_version"], "reason": reason,
                        "basis": "leave-one-storm-out", "loso_mean_spearman": {k: round(v, 3) for k, v in means.items()}}
    res["banner"] = {"on": bool(banner), "top_n_match": a["top_n_match"], "p90_coverage": a["p90_coverage"],
                     "targets": {"top_n_match": cfg["banner_topn_match"], "p90_coverage": cfg["banner_p90_coverage"]}}
    pc = pd.concat(per_county, ignore_index=True)
    pc["pred_out_p10"], pc["pred_out_p50"], pc["pred_out_p90"] = (pc[q] * pc["customers"] for q in ("p10", "p50", "p90"))
    pc["obs_out"] = pc["frac_out"] * pc["customers"]
    counties = gpd.read_parquet(PROCESSED / "counties.parquet")[["fips", "county"]]
    pc = pc.merge(counties, on="fips", how="left")
    keep = ["model", "fips", "county", "customers", "coastal", "p64", "p34", "p10", "p50", "p90", "frac_out",
            "pred_out_p10", "pred_out_p50", "pred_out_p90", "obs_out"]
    pc[keep].to_parquet(PROCESSED / "backtest_milton_counties.parquet", index=False)
    pd.DataFrame(curve).to_parquet(PROCESSED / "backtest_leadtime.parquet", index=False)
    res["study_counties"] = pc[pc["fips"].isin(STUDY_FIPS)][keep].round(4).to_dict(orient="records")
    return res


def observed_flooding(subs: pd.DataFrame, hwm: pd.DataFrame, radius_m: float, offset_m: float,
                      method: str = "max") -> pd.DataFrame:
    """Observed water surface at each substation from Ian HWMs within radius_m.

    method "max": any HWM within the radius above switchgear (PLAN rule, 500 m).
    method "idw": inverse-distance-weighted (power 2) water surface of the HWMs within the radius; used as a
    sensitivity check because HWMs are sparse (median substation is ~4 km from the nearest one).
    """
    s = gpd.GeoDataFrame(subs[["asset_id", "ground_elev_m"]], geometry=gpd.points_from_xy(subs.lon, subs.lat), crs=4326).to_crs(UTM17)
    h = gpd.GeoDataFrame(hwm[["hwm_id", "water_elev_m_navd88"]], geometry=gpd.points_from_xy(hwm.lon, hwm.lat), crs=4326).to_crs(UTM17)
    si, hi = h.sindex.query(s.geometry.buffer(radius_m), predicate="contains")
    pairs = pd.DataFrame({"asset_id": s["asset_id"].to_numpy()[si], "water": h["water_elev_m_navd88"].to_numpy()[hi],
                          "d": shapely.distance(s.geometry.to_numpy()[si], h.geometry.to_numpy()[hi])})
    pairs["w"] = 1.0 / np.maximum(pairs["d"], 1.0) ** 2
    agg = pairs.groupby("asset_id").apply(lambda g: pd.Series({
        "n_hwm": len(g), "nearest_hwm_m": g["d"].min(),
        "water_m": g["water"].max() if method == "max" else float(np.average(g["water"], weights=g["w"]))}),
        include_groups=False)
    out = s[["asset_id", "ground_elev_m"]].merge(agg, on="asset_id", how="left")
    out["switchgear_m"] = out["ground_elev_m"] + offset_m
    out["observed"] = out["n_hwm"].fillna(0) > 0
    out["observed_flooded"] = np.where(out["observed"], out["water_m"] > out["switchgear_m"], np.nan)
    return out.drop(columns="ground_elev_m")


def _pr(pred: pd.Series, truth: pd.Series) -> dict:
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": int((~pred & ~truth).sum()),
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None}


TRUTH_DEFS = [("max_500m", "any HWM within 500 m above switchgear (PLAN rule)", 500, "max"),
              ("idw_1km", "IDW water surface of HWMs within 1 km above switchgear", 1000, "idw"),
              ("idw_3km", "IDW water surface of HWMs within 3 km above switchgear", 3000, "idw")]
VARIANTS = [("psurge_screening_set", "P-Surge, screening set (as designed)", True, True),
            ("static_screening_set", "STATIC fallback, screening set", False, True),
            ("psurge_all_substations", "P-Surge, all substations", True, False)]


def fr22(cfg: dict) -> dict:
    reg = pd.read_parquet(PROCESSED / "registry.parquet")
    hwm = pd.read_parquet(PROCESSED / "hwm_ian.parquet")
    ps = pd.read_parquet(PROCESSED / "psurge_sites.parquet")
    t12 = advisory_at_lead("ian", cfg["decisionB_lead_hours"])
    offset = float(cfg["height_offset_m"])
    subs = reg[reg["type"] == "substation"]
    out = {"advisory_time": str(t12), "height_offset_m": offset, "threshold_in_force": c5.threshold_in_force(cfg),
           "primary_truth": TRUTH_DEFS[0][0], "truth_defs": {k: lab for k, lab, _, _ in TRUTH_DEFS}, "results": {}}
    flags = {v: c5.recommend(c5.inundation(reg, ps if use_ps else None, "ian", t12, cfg, screen=screen), reg, cfg)
             for v, _, use_ps, screen in VARIANTS}
    tables = []
    for tkey, _, radius, method in TRUTH_DEFS:
        truth = observed_flooding(subs, hwm, radius, offset, method)
        out["results"][tkey] = {}
        for v, _, _, _ in VARIANTS:
            f = flags[v].merge(truth, on="asset_id", how="left")
            obs = f[f["observed"].fillna(False).astype(bool)]
            t = obs["observed_flooded"].astype(bool)
            r = {"substations": int(len(f)), "with_hwm_observation": int(len(obs)), "observed_flooded": int(t.sum()),
                 "prob_sources": f["prob_source"].value_counts().to_dict(),
                 "recommendations": f["recommendation"].value_counts().to_dict()}
            if f["prob_source"].eq("STATIC").all():
                # STATIC is a tier routed to P2's judgment, not a thresholded probability: no precision/recall.
                r["routes_to_judgment"] = int(f["recommendation"].eq("JUDGMENT").sum())
                r["static_tiers"] = f["static_tier"].value_counts().to_dict()
            else:
                r["inundation_flag"] = _pr(obs["above_threshold"].astype(bool), t)
                r["de_energize"] = _pr(obs["recommendation"].eq("DE-ENERGIZE"), t)
            out["results"][tkey][v] = r
            tables.append(f.assign(variant=v, truth=tkey))
        allsubs = tables[-1]
        missed = allsubs[(allsubs["observed_flooded"] == True) & ~allsubs["fema_zone"].isin(cfg["screening_zones"])]  # noqa: E712
        out["results"][tkey]["flooded_outside_screening_set"] = missed[["asset_id", "name", "fema_zone", "prob"]].to_dict(orient="records")
    fr = pd.concat(tables, ignore_index=True)
    fr["critical_loads"] = fr["critical_loads"].apply(lambda v: ";".join(v))
    fr.to_parquet(PROCESSED / "fr22_ian.parquet", index=False)
    # Substations the FEMA-zone screening set never looks at, but P-Surge flags anyway (LIMITATIONS §4).
    allf = flags["psurge_all_substations"]
    outside = allf[~allf["fema_zone"].isin(cfg["screening_zones"]) & allf["prob_source"].eq("PSURGE")]
    above = outside[outside["prob"] >= out["threshold_in_force"]].sort_values("prob", ascending=False)
    out["outside_screening_above_threshold"] = above[["asset_id", "name", "fema_zone", "prob", "ground_elev_m"]].to_dict(orient="records")
    out["zone_x_psurge_ge_080"] = int((outside["fema_zone"].eq("X") & (outside["prob"] >= 0.8)).sum())
    return out


def write_report(b: dict, cfg: dict) -> None:
    f16, f22 = b["fr16"], b["fr22"]
    names = gpd.read_parquet(PROCESSED / "counties.parquet").set_index("fips")["county"].to_dict()
    nm = lambda fl: ", ".join(names.get(x, x) for x in fl)  # noqa: E731
    cv = loso()
    rng = {k: (min(v["spearman_frac"] for v in cv[k].values()), max(v["spearman_frac"] for v in cv[k].values()))
           for k in ("lgbm", "glm")}
    label = {"glm": "GLM (fractional logit)", "lgbm": "LightGBM (pinball, monotone)"}
    other = {"ian": "Idalia", "idalia": "Ian"}
    L = ["# Backtest report", "",
         f"Config `{cfg['_version']}`. Generated by `src/models/backtest.py`. Units: customers (meters), not people.", "",
         "## Leave-one-storm-out on the training storms (selection basis)", "",
         "Each model is fitted on one training storm and scored on the other (Spearman rank correlation of the "
         "fraction out, all county-advisory rows in the decision horizon). This, not Milton, selects the active model "
         "(FR30).", "",
         "| Model | Trained on | Scored on (held out) | Spearman | P90 coverage |", "|---|---|---|---|---|"]
    for k in ("glm", "lgbm"):
        for held, v in sorted(cv[k].items()):
            L.append(f"| {label[k]} | {other[held]} | {held.title()} | {v['spearman_frac']} | {v['p90_coverage']} |")
    L += ["", f"Between the two training storms, skill is low (Spearman {rng['lgbm'][0]:.2f}–{rng['lgbm'][1]:.2f} LightGBM, "
          f"{rng['glm'][0]:.2f}–{rng['glm'][1]:.2f} GLM). Milton's result is stronger because its track resembled Ian's; "
          "a storm on a different coast would look like these numbers. Two training storms is the model's main limit.", "",
          f"- Selection: {f16['selection']['reason']}.",
          f"- **Active model: `{f16['selection']['model_version']}`.**", "",
          "## FR16 — C1 on Milton (held out, reported once), all 67 Florida counties", "",
          f"Advisory: {f16['advisory_time']} (T-{f16['hours_to_landfall']:.1f}h, latest advisory at or before T-72h). "
          "Milton was not used in training, tuning or selection.", "",
          "| Metric | LightGBM (pinball, monotone) | GLM (fractional logit) | TabPFN v2 (challenger, not monotone) | Target |",
          "|---|---|---|---|---|"]
    tab = b.get("tabpfn") or {}
    for k, lab, tgt in [("mae_customers", "MAE, customers out per county", ""), ("mae_frac", "MAE, fraction out", ""),
                        ("spearman", "Spearman, county order", ""),
                        ("top_n_match", f"Top-{cfg['top_n']} match", f">= {cfg['banner_topn_match']}"),
                        ("p90_coverage", "P90 coverage", f">= {cfg['banner_p90_coverage']}"),
                        ("p10_p90_coverage", "P10-P90 coverage", "")]:
        L.append(f"| {lab} | {f16['lgbm'][k]} | {f16['glm'][k]} | {tab.get(k, 'not run')} | {tgt} |")
    L += ["", f"- Observed top {cfg['top_n']}: {nm(f16['lgbm']['top_n_observed'])}.",
          f"- LightGBM top {cfg['top_n']}: {nm(f16['lgbm']['top_n_predicted'])}.",
          f"- GLM top {cfg['top_n']}: {nm(f16['glm']['top_n_predicted'])}."]
    if tab:
        L += [f"- TabPFN top {cfg['top_n']}: {nm(tab['top_n_predicted'])}.", "",
              f"TabPFN column: `{tab['model']}` (`{tab['checkpoint']}`, sha256 `{tab['sha256'][:12]}...`), "
              f"in-context on the same {tab['context_rows']} training rows, no fitting. It has no monotone constraints, so "
              "it cannot be the active C1 model; it is a benchmark only. **Built with PriorLabs-TabPFN** "
              "(Prior Labs License 1.1, see docs/licenses/TabPFN-LICENSE.txt)."]
    L += ["",
          "### Low-confidence banner (FR17), on the holdout", "",
          f"- Active model top-{cfg['top_n']} match {f16['banner']['top_n_match']} (target {cfg['banner_topn_match']}), "
          f"P90 coverage {f16['banner']['p90_coverage']} (target {cfg['banner_p90_coverage']}).",
          f"- **Banner: {'ON' if f16['banner']['on'] else 'OFF'}.** "
          + ("The P1 screen shows the county ranking with a low-confidence banner and P1 decides on judgment."
             if f16["banner"]["on"] else "Targets met on the held-out storm."), "",
          "### Lee and Charlotte at Milton T-72h", "",
          "| Model | County | Customers | P10 out | P50 out | P90 out | Observed out |", "|---|---|---|---|---|---|---|"]
    for r in f16["study_counties"]:
        L.append(f"| {r['model']} | {r['county']} | {r['customers']:,} | {r['pred_out_p10']:,.0f} | {r['pred_out_p50']:,.0f} | "
                 f"{r['pred_out_p90']:,.0f} | {r['obs_out']:,.0f} |")
    L += ["", "## FR22 — Decision B on Ian", "",
          f"Advisory: {f22['advisory_time']} (latest at or before T-12h). Switchgear height ESTIMATED (ground + "
          f"{f22['height_offset_m']} m); threshold in force {f22['threshold_in_force']} (threshold + margin). "
          "Flag = P-Surge probability at or above the threshold in force. DE-ENERGIZE also requires no critical load "
          "(hospital or pumping station) on the placeholder feed; backup status is unknown, so any such load forces "
          "WATCH. PRD M4 target: precision 0.7, recall 0.8. The STATIC fallback is a tier (HIGH / MEDIUM / LOW from "
          "FEMA zone and BFE), not a probability: every STATIC site routes to P2's judgment, so it has no precision "
          "or recall.", "",
          "USGS high-water marks are sparse: the median substation is about 4 km from the nearest one. The PLAN rule "
          "(500 m) is the primary truth; 1 km and 3 km inverse-distance-weighted water surfaces are sensitivity checks. "
          "Read every row with its sample size.", ""]
    for tkey, lab in f22["truth_defs"].items():
        L += [f"### Truth: {lab}", "",
              "| Variant | Substations | With HWM obs. | Observed flooded | Flag precision | Flag recall | DE-ENERGIZE precision | DE-ENERGIZE recall |",
              "|---|---|---|---|---|---|---|---|"]
        for v, vlab, _, _ in VARIANTS:
            r = f22["results"][tkey][v]
            if "routes_to_judgment" in r:
                L.append(f"| {vlab} | {r['substations']} | {r['with_hwm_observation']} | {r['observed_flooded']} | "
                         f"routes to judgment (n={r['routes_to_judgment']}) | — | — | — |")
                continue
            L.append(f"| {vlab} | {r['substations']} | {r['with_hwm_observation']} | {r['observed_flooded']} | "
                     f"{r['inundation_flag']['precision']} | {r['inundation_flag']['recall']} | "
                     f"{r['de_energize']['precision']} | {r['de_energize']['recall']} |")
        m = f22["results"][tkey]["flooded_outside_screening_set"]
        if m:
            L.append("")
            L.append("Observed flooded but outside the FEMA screening set: " + "; ".join(
                f"{r['asset_id']} {r['name'] or '(unnamed)'} (zone {r['fema_zone']}, P-Surge {r['prob']:.2f})" for r in m) + ".")
        L.append("")
    ox = f22["outside_screening_above_threshold"]
    L += ["### Outside the screening set at Ian T-12h", "",
          f"Substations not in the FEMA-zone screening set whose P-Surge probability is at or above the threshold in force "
          f"({f22['threshold_in_force']}): {len(ox)}. Of these, {f22['zone_x_psurge_ge_080']} are in Zone X with P-Surge "
          ">= 0.8. The screening set (PRD C5) never looks at them; Phase 0 should screen on P-Surge coverage as well as "
          "FEMA zone (LIMITATIONS §4).", "",
          "| Substation | FEMA zone | P-Surge prob | Ground elevation (m NAVD88) |", "|---|---|---|---|"]
    for r in ox:
        L.append(f"| {r['name'] or r['asset_id']} ({r['asset_id']}) | {r['fema_zone']} | {r['prob']:.2f} | {r['ground_elev_m']:.2f} |")
    L += ["", "## Pairing rule (R5)", "", "Each advisory is paired with the storm's peak customers out per county "
          "(max of the 15-minute EAGLE-I series from landfall - 24 h to + 96 h, net of the first-day median); nothing is "
          "interpolated between the 15-minute and 6-hour cadences.", ""]
    (ROOT / "docs" / "backtest_report.md").write_text("\n".join(L))


def main() -> None:
    cfg = load_config()
    b = {"config_version": cfg["_version"], "fr16": fr16(cfg), "fr22": fr22(cfg)}
    tab = PROCESSED / "backtest_tabpfn.json"  # optional challenger, produced by src/models/tabpfn_challenger.py
    if tab.exists():
        b["tabpfn"] = json.loads(tab.read_text())
    sel = b["fr16"]["selection"]
    (MODEL_DIR / "selection.json").write_text(json.dumps({**sel, "banner": b["fr16"]["banner"]}, indent=2))
    (PROCESSED / "backtest.json").write_text(json.dumps(b, indent=2, default=str))
    write_report(b, cfg)
    print((ROOT / "docs" / "backtest_report.md").read_text())


if __name__ == "__main__":
    main()
