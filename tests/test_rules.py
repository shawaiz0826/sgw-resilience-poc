"""Rule tests for C2, C4, C5 on small hand-built inputs (no data files)."""
import math

import pandas as pd
import pytest

from src.config import load_config
from src.decisions import c5_inundation as c5
from src.decisions.c2_staging import assign_site, crew_hours, staging_plan
from src.decisions.c4_dependency import dependencies

CFG = load_config()
LOW = {"FTM": 0.1, "PGD": 0.1, "SRQ": 0.1, "SEB": 0.1}


def test_crew_hours_formula():
    # 10 customers per worker-day, 8 h per worker-day: 1,000 customers out -> 100 worker-days -> 800 crew-hours
    assert crew_hours(1000, CFG) == pytest.approx(1000 / CFG["restoration_rate"] * CFG["hours_per_worker_day"])


def test_mutual_aid_whole_crews_over_window():
    zones = [{"zone_fips": "12071", "zone_name": "Lee", "out_p50": 10_000, "out_p90": 50_000}]
    p = staging_plan(zones, LOW, own_crew_hours=0, cfg=CFG)
    per_crew = CFG["crew_size"] * CFG["hours_per_worker_day"] * CFG["restoration_days"]
    assert p["mutual_aid_crews"] == math.ceil(crew_hours(50_000, CFG) / per_crew)
    assert staging_plan(zones, LOW, own_crew_hours=10**9, cfg=CFG)["mutual_aid_crews"] == 0


def test_site_kept_below_threshold():
    s = assign_site("12071", LOW, CFG)
    assert s["assigned_site"] == "FTM" and not s["displaced"]


def test_site_displaced_to_nearest_below_threshold():
    s = assign_site("12071", {"FTM": 0.8, "PGD": 0.2, "SRQ": 0.1, "SEB": 0.1}, CFG)
    assert s["displaced"] and s["assigned_site"] == "PGD"  # Punta Gorda is nearer to Fort Myers than Sebring


def test_all_sites_above_threshold_left_to_p1():
    s = assign_site("12071", {k: 0.9 for k in LOW}, CFG)
    assert s["displaced"] and s["assigned_site"] is None and "P1" in s["note"]


def test_psurge_threshold_from_offset():
    assert c5.psurge_threshold_ft(1.0) == 3
    assert c5.psurge_threshold_ft(0.3048) == 1


def test_threshold_raised_when_estimated():
    assert c5.threshold_in_force(CFG, "ESTIMATED") == pytest.approx(CFG["decisionB_threshold"] + CFG["decisionB_margin"])
    assert c5.threshold_in_force(CFG, "RECORDED") == CFG["decisionB_threshold"]


def _registry():
    return pd.DataFrame([
        {"asset_id": "SUB-1", "type": "substation", "name": "a", "lon": -81.9, "lat": 26.6, "fema_zone": "AE",
         "fema_zone_subtype": None, "ground_elev_m": 1.0, "bfe_ft": 10.0, "bfe_m_navd88": 3.048, "feed_asset_id": None,
         "feed_distance_km": None, "county_fips": "12071", "source": "t", "source_vintage": "t"},
        {"asset_id": "SUB-2", "type": "substation", "name": "b", "lon": -81.8, "lat": 26.6, "fema_zone": "AE",
         "fema_zone_subtype": None, "ground_elev_m": 5.0, "bfe_ft": 10.0, "bfe_m_navd88": 3.048, "feed_asset_id": None,
         "feed_distance_km": None, "county_fips": "12071", "source": "t", "source_vintage": "t"},
        {"asset_id": "HSP-1", "type": "hospital", "name": "h", "lon": -81.9, "lat": 26.61, "fema_zone": "X",
         "fema_zone_subtype": None, "ground_elev_m": 3.0, "bfe_ft": None, "bfe_m_navd88": None, "feed_asset_id": "SUB-1",
         "feed_distance_km": 1.0, "county_fips": "12071", "source": "t", "source_vintage": "t"},
        {"asset_id": "PMP-1", "type": "pumping", "name": "p", "lon": -81.0, "lat": 26.0, "fema_zone": "X",
         "fema_zone_subtype": None, "ground_elev_m": 3.0, "bfe_ft": None, "bfe_m_navd88": None, "feed_asset_id": "SUB-2",
         "feed_distance_km": 25.0, "county_fips": "12071", "source": "t", "source_vintage": "t"},
    ])


def test_static_routes_to_judgment_with_tier():
    reg = _registry()
    f = c5.recommend(c5.inundation(reg, None, "x", pd.Timestamp("2024-01-01", tz="UTC"), CFG), reg, CFG).set_index("asset_id")
    assert f.loc["SUB-1", "static_tier"] == "HIGH"    # AE, BFE 3.05 m above switchgear 1.0 + 1.0 m
    assert f.loc["SUB-2", "static_tier"] == "MEDIUM"  # plain AE: BFE 3.05 m below switchgear 5.0 + 1.0 m
    assert set(f["recommendation"]) == {"JUDGMENT"} and set(f["prob_source"]) == {"STATIC"}
    assert f["threshold"].isna().all() and f["above_threshold"].isna().all()  # STATIC is never thresholded
    assert all("P2 to decide" in r for r in f["reason"])
    assert set(f["height_source"]) == {"ESTIMATED"}


def test_critical_load_forces_watch():
    reg = _registry()
    ps = pd.DataFrame([{"storm_id": "x", "advisory_time": pd.Timestamp("2024-01-01", tz="UTC"), "asset_id": a,
                        "threshold_ft": 3, "prob": 0.99} for a in ("SUB-1", "SUB-2")])
    f = c5.recommend(c5.inundation(reg, ps, "x", pd.Timestamp("2024-01-01", tz="UTC"), CFG), reg, CFG).set_index("asset_id")
    assert f.loc["SUB-1", "recommendation"] == "WATCH" and bool(f.loc["SUB-1", "escalation"])
    assert "ESCALATE: confirm backup for hospital:HSP-1 before de-energizing" in f.loc["SUB-1", "reason"]
    assert not bool(f.loc["SUB-2", "escalation"])
    assert f.loc["SUB-2", "recommendation"] == "DE-ENERGIZE"  # PMP-1's placeholder feed is 25 km: not a load on SUB-2
    assert f["sensor_value"].isna().all()


def test_no_plausible_feed_beyond_max_km():
    reg = _registry()
    expo = pd.DataFrame([{"asset_id": "SUB-1", "type": "substation", "score": 0.5, "cell_prob": 0.4, "rank": 1},
                         {"asset_id": "SUB-2", "type": "substation", "score": 0.2, "cell_prob": 0.2, "rank": 2}])
    d = dependencies(reg, expo, CFG).set_index("asset_id")
    assert d.loc["HSP-1", "feed_label"] == "PLACEHOLDER FEED" and d.loc["HSP-1", "inherited_score"] == 0.5
    assert d.loc["PMP-1", "feed_kind"] == "NONE" and pd.isna(d.loc["PMP-1", "inherited_score"])
