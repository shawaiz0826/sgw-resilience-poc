"""FR34 retrieval over the decision record and the asset registry, with source rows shown.

Template mode answers only by retrieval: it finds the rows the question names (an asset, a county, a topic) and
shows them. Anything it cannot ground in a record row gets "Not in the record." It never produces a rationale
of its own (PRD Section 8: "C6 answers 'why is this substation on watch' by retrieving those rows").
"""
from __future__ import annotations

import re

import pandas as pd

NOT_IN_RECORD = "Not in the record. I can only answer from the decision records and the asset registry for this advisory."

TOPICS = {
    "crews": ["mutual aid", "crew", "crews", "workers", "staging", "stage", "site"],
    "substations": ["de-energize", "deenergize", "de-energise", "watch", "surge", "inundation", "substation", "switchgear"],
    "pumping": ["pumping", "generator", "lift station", "water", "wastewater", "plant"],
    "banner": ["banner", "confidence", "backtest", "accuracy"],
    "counties": ["county", "counties", "customers out", "outage", "outages"],
}


def answer(question: str, a: dict, b: dict | None, registry: pd.DataFrame) -> tuple[str, pd.DataFrame | None, list[str]]:
    q = question.lower().strip()
    ids = [a["record_id"]] + ([b["record_id"]] if b else [])
    subs_est = pd.DataFrame(b["estimates"]["substations"]) if b else pd.DataFrame()
    subs_rec = pd.DataFrame(b["recommendation"]["substations"]) if b else pd.DataFrame()
    subs = subs_est.merge(subs_rec, on="asset_id") if b else pd.DataFrame()
    assets = pd.DataFrame(a["estimates"]["assets"])
    counties = pd.DataFrame(a["estimates"]["counties"])

    # 1. A county, when the question names one as a county.
    for _, c in counties.iterrows():
        if "count" in q and c["county"] and re.search(rf"\b{re.escape(c['county'].lower())}\b", q):
            return (f"{c['county']} County at this advisory (customers, meters):",
                    counties[counties["fips"] == c["fips"]][["county", "customers", "out_p10", "out_p50", "out_p90", "p64", "p34"]], [a["record_id"]])

    # 2. A named asset (by ID or by name).
    for aid in re.findall(r"\b(?:sub|lin|wwt|pmp|hsp)-\d{4}\b", q):
        aid = aid.upper()
        if len(subs) and aid in set(subs["asset_id"]):
            row = subs[subs["asset_id"] == aid]
            return f"{aid}: {row['recommendation'].iloc[0]} — {row['reason'].iloc[0]}.", row, [b["record_id"]]
        if aid in set(assets["asset_id"]):
            return f"{aid} exposure (county-validated ranking only):", assets[assets["asset_id"] == aid], [a["record_id"]]
    named = registry[registry["name"].fillna("").str.len() > 3]
    for _, r in named.iterrows():
        nm = r["name"].lower().replace(" substation", "").strip()
        if len(nm) > 3 and nm in q:
            if len(subs) and r["asset_id"] in set(subs["asset_id"]):
                row = subs[subs["asset_id"] == r["asset_id"]]
                return f"{r['name']}: {row['recommendation'].iloc[0]} — {row['reason'].iloc[0]}.", row, [b["record_id"]]
            if r["asset_id"] in set(assets["asset_id"]):
                return f"{r['name']} exposure (county-validated ranking only):", assets[assets["asset_id"] == r["asset_id"]], [a["record_id"]]

    # 3. A topic.
    hit = next((t for t, words in TOPICS.items() if any(w in q for w in words)), None)
    plan = a["recommendation"]["plan"]
    if hit == "crews":
        rows = pd.DataFrame(plan["zones"])[["zone_name", "customers_out_p50", "customers_out_p90", "crew_hours_p90",
                                            "assigned_site_name", "displaced", "note"]]
        return (f"Mutual aid request: {plan['mutual_aid_crews']:,} crews ({plan['mutual_aid_workers']:,} workers). "
                f"Rule: {plan['formula']}.", rows, [a["record_id"]])
    if hit == "substations":
        if not b:
            return "Decision B has not run for this advisory, so there is no substation recommendation in the record.", None, ids
        return f"Decision B: {b['recommendation']['counts']}.", subs[["asset_id", "name", "prob", "prob_source", "threshold",
                                                                          "recommendation", "reason"]], [b["record_id"]]
    if hit == "pumping":
        d = pd.DataFrame(a["estimates"]["dependents"])
        return "Dependent assets and the substation score they inherit (PLACEHOLDER FEED):", d[
            ["asset_id", "type", "name", "feed_name", "feed_label", "inherited_score"]], [a["record_id"]]
    if hit == "banner":
        bn = a["recommendation"]["banner"]
        return (f"Banner {'ON' if bn['on'] else 'off'}: top-5 match {bn['top_n_match']} (target {bn['targets']['top_n_match']}), "
                f"P90 coverage {bn['p90_coverage']} (target {bn['targets']['p90_coverage']}) on {bn['held_out_storm'].title()}.",
                None, [a["record_id"]])
    if hit == "counties":
        c = counties.sort_values("out_p50", ascending=False).head(10)
        return "Top counties by P50 customers out at this advisory:", c[["county", "out_p10", "out_p50", "out_p90"]], [a["record_id"]]
    return NOT_IN_RECORD, None, []
