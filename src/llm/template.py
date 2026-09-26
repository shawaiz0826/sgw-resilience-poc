"""FR36 — fixed briefing template filled from the decision record. The default C6 mode: no LLM, no key.

Every number is copied from the records; the record IDs used are listed. P5 edits the draft and sends it.
"""
from __future__ import annotations

from src.decisions.c2_staging import scale_line
from src.decisions.common import EXPOSURE_LABEL, UNITS


def _n(x) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


def briefing(a: dict, b: dict | None, actions_a: list[dict] | None = None) -> str:
    inp, plan, banner = a["inputs"], a["recommendation"]["plan"], a["recommendation"]["banner"]
    L = [f"SGW STORM BRIEFING — {inp['storm_name'].upper()}, advisory {inp['advisory_time'][:16].replace('T', ' ')}Z "
         f"(T-{inp['hours_to_landfall']:.0f}h)", "",
         f"Source records: {a['record_id']}" + (f", {b['record_id']}" if b else "") +
         f". Configuration {a['config_version']}; model {a['model_version']}. Units: {UNITS}.", ""]
    if banner["on"]:
        L += [f"LOW CONFIDENCE: the model in use missed its backtest targets on held-out storm {banner['held_out_storm'].title()} "
              f"(top-5 match {banner['top_n_match']}, P90 coverage {banner['p90_coverage']}). The Storm Director decides on judgment.", ""]
    L += ["1. Expected outages (Lee and Charlotte)"]
    for z in plan["zones"]:
        L.append(f"   - {z['zone_name']}: P50 {_n(z['customers_out_p50'])}, P90 {_n(z['customers_out_p90'])} customers (meters) out.")
    L += ["", "2. Mutual aid and staging"]
    L.append(f"   - Recommended request: {_n(plan['mutual_aid_crews'])} crews of {plan['crew_size']} "
             f"({_n(plan['mutual_aid_workers'])} workers), sized on P90 crew-hours {_n(plan['total_crew_hours_p90'])} less SGW's "
             f"own {_n(plan['own_crew_hours'])} ({a['crew_hours_source']}), over {plan['restoration_days']:.0f} days.")
    L.append(f"   - {scale_line(plan)}")
    for z in plan["zones"]:
        site = z.get("assigned_site_name") or "no site below the wind threshold (Storm Director to decide)"
        moved = f" (displaced from {z['mapped_site_name']})" if z.get("displaced") else ""
        L.append(f"   - {z['zone_name']}: stage at {site}{moved}.")
    last = (actions_a or [None])[0]
    if last:
        L.append(f"   - Storm Director action: {last['action']}" + (f", reason: {last['reason']}" if last.get("reason") else "") + ".")
    else:
        L.append("   - Storm Director action: not yet signed.")
    L += ["", "3. Substations (Decision B)"]
    if b is None:
        L.append("   - Decision B has not run for this advisory (no P-Surge grid and before the watch window).")
    else:
        rc = b["recommendation"]
        rows = {r["asset_id"]: r for r in b["estimates"]["substations"]}
        nm = lambda r: rows[r["asset_id"]]["name"] or r["asset_id"]  # noqa: E731
        de = [r for r in rc["substations"] if r["recommendation"] == "DE-ENERGIZE"]
        held = [r for r in rc["substations"] if r["recommendation"] == "WATCH" and r["above_threshold"]]
        judg = [r for r in rc["substations"] if r["recommendation"] == "JUDGMENT"]
        if judg:
            L.append(f"   - Probability source: {rc['prob_source']}: no P-Surge snapshot for this advisory, so each site "
                     "has a STATIC tier from its FEMA zone and BFE and goes to P2's judgment; switchgear height ESTIMATED. "
                     "Flood sensor values: blank (not available).")
            L.append(f"   - Routed to P2 judgment ({len(judg)}): " + ", ".join(
                f"{nm(r)} (STATIC tier {rows[r['asset_id']]['static_tier']}, P2 to decide)" for r in judg) + ".")
        else:
            L.append(f"   - Probability source: {rc['prob_source']}; switchgear height ESTIMATED; threshold in force "
                     f"{rc['threshold_in_force']:.2f}. Flood sensor values: blank (not available).")
        if de or held or not judg:
            L.append(f"   - DE-ENERGIZE recommended ({len(de)}): " +
                     (", ".join(f"{nm(r)} ({rows[r['asset_id']]['prob']:.2f})" for r in de) or "none") + ".")
            L.append(f"   - Held on WATCH for a critical load with unknown backup ({len(held)}): " +
                     (", ".join(nm(r) for r in held) or "none") + ".")
            L.append(f"   - Other screening-set substations on WATCH: {rc['counts'].get('WATCH', 0) - len(held)}.")
    deps = [d for d in a["estimates"]["dependents"] if d["type"] == "pumping" and d["inherited_score"] is not None][:5]
    L += ["", "4. Pumping stations by inherited exposure (generator list, PLACEHOLDER FEED)"]
    L += [f"   - {d['name']} ({d['asset_id']}) <- {d['feed_name'] or d['feed_asset_id']}: score {d['inherited_score']:.3f}" for d in deps] or ["   - none"]
    L += ["", f"Asset scores are an {EXPOSURE_LABEL}; they are not validated per asset.",
          "Drafted from the decision record by the fixed template (FR36). Edited and sent by P5."]
    return "\n".join(L)
