"""C2 — crew-hours and staging recommendation (PRD C2). A rule, no ML.

(a) crew-hours per zone at P50 and P90 = customers out at that quantile / restoration rate x hours per worker-day
(b) mutual aid request = (sum of P90 crew-hours - SGW's own available crew-hours), in whole crews of crew_size,
    sized for the restoration window: ceil(shortfall / (crew_size x hours_per_worker_day x restoration_days))
(c) each zone stages at its mapped site unless that site's 64 kt probability exceeds site_wind_threshold; then at
    the nearest listed site at or below the threshold, flagged displaced (crews move in after the wind window)
(d) the plan lists, per zone, P50/P90 crew-hours, the assigned site and whether (c) displaced it
restoration_days is a PLACEHOLDER added to the PRD rule so the request is in crews, not crew-days.
"""
from __future__ import annotations

import math

CALIBRATION_NOTE = ("Ian, FPL: about 2 million customers restored by about 21,000 workers in about 9 days, roughly 95 per "
                    "worker (FPL newsroom; S&P Global, Sept 2022)")


def scale_line(plan: dict) -> str:
    """One-line sense check of the request against Ian (shown on P1 and in the briefing)."""
    if not plan.get("customers_per_worker"):
        return "For scale: no mutual aid is requested at this advisory; after Ian, FPL managed about 95 customers per worker over 9 days."
    return (f"For scale: this request works out to {plan['customers_per_worker']:.0f} customers per worker over the "
            f"{plan['restoration_days']:.0f}-day placeholder window; after Ian, FPL managed about 95 per worker over 9 days.")


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def crew_hours(customers_out: float, cfg: dict) -> float:
    return customers_out / float(cfg["restoration_rate"]) * float(cfg["hours_per_worker_day"])


def assign_site(zone_fips: str, site_p64: dict[str, float], cfg: dict) -> dict:
    sites = {s["site_id"]: s for s in cfg["staging_sites"]}
    thr = float(cfg["site_wind_threshold"])
    mapped = next((s for s in cfg["staging_sites"] if s.get("zone_fips") == zone_fips), None)
    if mapped is None:
        return {"mapped_site": None, "assigned_site": None, "displaced": False, "note": "no site mapped to this zone"}
    mp = site_p64[mapped["site_id"]]
    base = {"mapped_site": mapped["site_id"], "mapped_site_name": mapped["name"], "mapped_site_p64": mp}
    if mp <= thr:
        return {**base, "assigned_site": mapped["site_id"], "assigned_site_name": mapped["name"],
                "assigned_site_p64": mp, "displaced": False, "distance_km": 0.0,
                "note": f"mapped site 64 kt probability {mp:.2f} <= {thr:.2f}"}
    candidates = sorted(
        ((haversine_km(mapped["lon"], mapped["lat"], s["lon"], s["lat"]), sid) for sid, s in sites.items()
         if sid != mapped["site_id"] and site_p64[sid] <= thr))
    if not candidates:
        return {**base, "assigned_site": None, "assigned_site_name": None, "assigned_site_p64": None, "displaced": True,
                "distance_km": None, "note": f"every listed site above {thr:.2f}: P1 to decide on judgment"}
    d, sid = candidates[0]
    return {**base, "assigned_site": sid, "assigned_site_name": sites[sid]["name"], "assigned_site_p64": site_p64[sid],
            "displaced": True, "distance_km": round(d, 1),
            "note": f"mapped site 64 kt probability {mp:.2f} > {thr:.2f}; nearest site at or below threshold, "
                    "move in after the wind window closes"}


def staging_plan(zones: list[dict], site_p64: dict[str, float], own_crew_hours: float, cfg: dict) -> dict:
    """zones: [{zone_fips, zone_name, out_p50, out_p90}] -> the C2 plan (plain dict, goes in the record)."""
    rows = []
    for z in sorted(zones, key=lambda z: z["zone_fips"]):
        rows.append({
            "zone_fips": z["zone_fips"], "zone_name": z["zone_name"],
            "customers_out_p50": z["out_p50"], "customers_out_p90": z["out_p90"],
            "crew_hours_p50": round(crew_hours(z["out_p50"], cfg), 1), "crew_hours_p90": round(crew_hours(z["out_p90"], cfg), 1),
            **assign_site(z["zone_fips"], site_p64, cfg),
        })
    tot50 = sum(r["crew_hours_p50"] for r in rows)
    tot90 = sum(r["crew_hours_p90"] for r in rows)
    per_crew = float(cfg["crew_size"]) * float(cfg["hours_per_worker_day"]) * float(cfg["restoration_days"])
    shortfall = max(0.0, tot90 - float(own_crew_hours))
    crews = math.ceil(shortfall / per_crew) if shortfall > 0 else 0
    crews_p50 = math.ceil(max(0.0, tot50 - float(own_crew_hours)) / per_crew)
    workers = crews * int(cfg["crew_size"])
    p90_out = sum(r["customers_out_p90"] for r in rows)
    return {
        "zones": rows, "total_crew_hours_p50": round(tot50, 1), "total_crew_hours_p90": round(tot90, 1),
        "own_crew_hours": float(own_crew_hours), "shortfall_crew_hours_p90": round(shortfall, 1),
        "mutual_aid_crews": crews, "mutual_aid_workers": workers,
        "customers_per_worker": round(p90_out / workers, 1) if workers else None,
        "calibration_note": CALIBRATION_NOTE,
        "mutual_aid_crews_if_p50": crews_p50, "crew_size": int(cfg["crew_size"]),
        "restoration_days": float(cfg["restoration_days"]), "restoration_rate": float(cfg["restoration_rate"]),
        "site_wind_threshold": float(cfg["site_wind_threshold"]), "site_p64": site_p64,
        "formula": ("crews = ceil((sum P90 crew-hours - own crew-hours) / (crew_size x hours_per_worker_day x "
                    "restoration_days)); crew-hours = customers out / restoration_rate x hours_per_worker_day"),
    }
