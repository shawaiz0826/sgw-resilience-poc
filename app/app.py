"""SGW Storm Decision Support — prototype app (PLAN §5D). Run: make demo  (or: streamlit run app/app.py)

Every screen reads the decision record; nothing on screen is newer than the advisory behind it. Prototype
substitutions (PRD Appendix A) are labelled where they appear.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import pydeck as pdk  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from src.config import load_config  # noqa: E402
from src.decisions.common import EXPOSURE_LABEL, UNITS, advisories  # noqa: E402
from src.decisions.run import replay, run  # noqa: E402
from src.llm import prompts, provider, qa, template  # noqa: E402
from src.models.c1 import model_version  # noqa: E402
from src.pull.common import PROCESSED, STORMS  # noqa: E402
from src.store import record  # noqa: E402

st.set_page_config(page_title="SGW Storm Decision Support (POC)", page_icon="🌀", layout="wide")

STORM_OPTIONS = {
    "milton": "Milton 2024 (held out)",
    "ian": "Ian 2022 (training storm)",
}
DEFAULT_LEAD = {"milton": 72, "ian": 12}
ACTORS = {"P1": "Storm Director (P1)", "P2": "Control room manager (P2)", "P5": "Emergency management lead (P5)",
          "P6": "Platform owner (P6)"}


# ---------------------------------------------------------------- data (read-only, cached)
@st.cache_resource
def conn():
    return record.connect()


@st.cache_data
def cfg() -> dict:
    return load_config()


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


@st.cache_data
def load_geo(name: str) -> dict:
    g = gpd.read_parquet(PROCESSED / f"{name}.parquet")
    return json.loads(g.to_json())


@st.cache_data
def backtest() -> dict:
    return json.loads((PROCESSED / "backtest.json").read_text())


@st.cache_data
def selection() -> dict:
    return json.loads((ROOT / "models" / "c1" / "selection.json").read_text())


@st.cache_data(show_spinner="Running C1 → C5 for this advisory and writing the record…")
def decisions(storm: str, t_iso: str, crew: float | None, model_kind: str, cfg_version: str) -> dict:
    outs = run(storm, pd.Timestamp(t_iso), own_crew_hours=crew, model_kind=model_kind, conn=conn())
    return {o["decision"]: o["record"] for o in outs}


def model_in_service() -> tuple[str, str | None]:
    """FR30: the approved active version unless P1 withdrew it; then the fallback version."""
    active = selection()["active"]
    acts = record.list_actions(conn())
    state = {}
    for _, a in acts.sort_values("action_id").iterrows():
        if a["action"] in ("WITHDRAW_MODEL", "RESTORE_MODEL") and a["payload_json"]:
            state[json.loads(a["payload_json"])["model_kind"]] = a["action"]
    if state.get(active) == "WITHDRAW_MODEL":
        return ("lgbm" if active == "glm" else "glm"), active
    return active, None


def actions_for(record_id: str) -> list[dict]:
    return record.list_actions(conn(), record_id).to_dict(orient="records")


# ---------------------------------------------------------------- sidebar
C = cfg()
with st.sidebar:
    st.markdown("### 🌀 SGW Storm Decision Support")
    st.info("**DEMO MODE** — public data, Lee & Charlotte Counties, FL. Nothing here is SGW's.", icon=":material/info:")
    storm = st.selectbox("Storm", list(STORM_OPTIONS), format_func=STORM_OPTIONS.get, key="storm")
    st.caption("Milton: C1 never saw it (held-out test). Ian: C1 trained on it (in-sample); Ian has real P-Surge for Decision B.")
    adv = advisories(storm)
    adv = adv[(adv["hours_to_landfall"] > 0) & (adv["hours_to_landfall"] <= 80)]
    labels = {r.advisory_time.isoformat(): f"T-{r.hours_to_landfall:.0f}h · {r.advisory_time:%d %b %H}Z"
              for r in adv.itertuples()}
    default_t = adv[adv["hours_to_landfall"] >= DEFAULT_LEAD[storm]]["advisory_time"].max().isoformat()
    t_iso = st.select_slider("NHC advisory (T-72h → landfall)", options=list(labels), value=default_t,
                             format_func=labels.get, key=f"adv_{storm}")
    t = pd.Timestamp(t_iso)
    arow = adv[adv["advisory_time"] == t].iloc[0]
    kind, withdrawn = model_in_service()
    ops = record.storm_ops_state(conn())
    st.markdown(f"**Advisory** {t:%Y-%m-%d %H:%M}Z · T-{arow.hours_to_landfall:.1f}h  \n"
                f"**Grid resolution** {arow.resolution} (NHC wind speed probability, contour bands)")
    st.markdown(f"**Config** `{C['_version']}`" + ("  \n:red-badge[FROZEN] storm operations declared" if ops["declared"] else ""))
    st.markdown(f"**Model in service** `{model_version(kind)}`" +
                (f"  \n:orange-badge[FALLBACK] `{withdrawn}` withdrawn by P1" if withdrawn else ""))
    bt = backtest()["fr16"][kind]
    banner_on = bt["top_n_match"] < C["banner_topn_match"] or bt["p90_coverage"] < C["banner_p90_coverage"]
    st.markdown("**Confidence banner** " + (":red-badge[ON]" if banner_on else ":green-badge[off]"))
    st.caption("**Data vintage** — substations: OSM 2026-09-26 · lines: HIFLD 2015–17 (deprecated) · hospitals: HIFLD "
               "2013–14 · WWTP: EPA FRS · flood zones: FEMA NFHL (Esri copy) · outages: EAGLE-I 2022–24 · elevation: "
               "USGS 3DEP · wind: NHC 5km · surge: NHC P-Surge (Ian only)")

try:
    res = decisions(storm, t_iso, st.session_state.get(f"crew_{storm}_{t_iso}"), kind, C["_version"])
except RuntimeError as err:  # e.g. the fallback LightGBM version cannot load on this machine (FR30)
    st.error(f"Model `{kind}` cannot run here: {err}. Restore the approved version in Governance.")
    st.stop()
A, B = res["A"], res.get("B")

st.markdown(f"## {STORMS[storm]['name']} · advisory {t:%d %b %Y %H}Z · T-{arow.hours_to_landfall:.0f}h")
st.caption(f"Record A `{A['record_id']}`" + (f" · Record B `{B['record_id']}`" if B else " · Decision B not run") +
           f" · config `{C['_version']}` · model `{A['model_version']}` · units: {UNITS}")

tabs = st.tabs(["P1 Storm Director", "P2 Control room", "P3 Field ops", "P4 Water ops", "P5 Briefing",
                "Backtest", "Record", "Governance"])


# ---------------------------------------------------------------- P1
with tabs[0]:
    plan, banner = A["recommendation"]["plan"], A["recommendation"]["banner"]
    if banner["on"]:
        st.error(f"**LOW CONFIDENCE** — model `{A['model_version']}` missed its backtest targets on held-out "
                 f"{banner['held_out_storm'].title()}: top-5 match {banner['top_n_match']:.2f} (target "
                 f"{banner['targets']['top_n_match']:.2f}), P90 coverage {banner['p90_coverage']:.2f} (target "
                 f"{banner['targets']['p90_coverage']:.2f}). **Decide on judgment.** (FR17)", icon=":material/warning:")
    else:
        st.success(f"Confidence banner **off** — model `{A['model_version']}` met its targets on held-out "
                   f"{banner['held_out_storm'].title()}: top-5 match {banner['top_n_match']:.2f} ≥ "
                   f"{banner['targets']['top_n_match']:.2f}, P90 coverage {banner['p90_coverage']:.2f} ≥ "
                   f"{banner['targets']['p90_coverage']:.2f}. (FR17)", icon=":material/check_circle:")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mutual aid request", f"{plan['mutual_aid_crews']:,} crews", help=plan["formula"])
    c2.metric("Workers", f"{plan['mutual_aid_workers']:,}", help=f"crew size {plan['crew_size']} (PLACEHOLDER)")
    c3.metric("P90 crew-hours (Lee + Charlotte)", ui.fmt(plan["total_crew_hours_p90"]))
    c4.metric("If sized on P50", f"{plan['mutual_aid_crews_if_p50']:,} crews")

    with st.container(border=True):
        cc1, cc2 = st.columns([1, 2])
        default_crew = float(C["own_crew_hours_default"])
        entered = cc1.number_input("SGW's own available crew-hours", min_value=0.0, step=1000.0,
                                   value=float(A["crew_hours_entered"]), key=f"crew_in_{storm}_{t_iso}")
        cc2.markdown(":violet-badge[PLACEHOLDER] typed by P1 at T-72h and revised at T-48h (FR11); the crew roster "
                     "replaces the typing in Phase 2. Changing it reruns Decision A and writes a new record.  \n"
                     f"Current source: **{A['crew_hours_source']}**. Restoration window {plan['restoration_days']:.0f} days, "
                     f"crew size {plan['crew_size']}, {plan['restoration_rate']:.0f} customers per worker-day — all config values.")
        if entered != float(A["crew_hours_entered"]):
            st.session_state[f"crew_{storm}_{t_iso}"] = None if entered == default_crew else entered
            st.rerun()

    left, right = st.columns([1.05, 1])
    with left:
        est = pd.DataFrame(A["estimates"]["counties"])
        geo = load_geo("counties")
        e = est.set_index("fips")
        top_frac = max(float(est["frac_p50"].max()), 1e-6)
        for f in geo["features"]:
            p = f["properties"]
            r = e.loc[p["fips"]]
            p.update({"p50": float(r["frac_p50"]), "fill": ui.seq_rgb(float(r["frac_p50"]) / top_frac, 200),
                      "line": [11, 11, 11, 220] if r["study_zone"] else [255, 255, 255, 160],
                      "lw": 3 if r["study_zone"] else 1, "out50": ui.fmt(r["out_p50"]), "out10": ui.fmt(r["out_p10"]),
                      "out90": ui.fmt(r["out_p90"]), "frac": f"{r['frac_p50']:.1%}"})
        sites = pd.DataFrame([{**s, "p64": plan["site_p64"][s["site_id"]]} for s in C["staging_sites"]])
        used = {z["assigned_site"] for z in plan["zones"] if z.get("assigned_site")}
        sites["state"] = ["above wind threshold" if p > C["site_wind_threshold"] else ("assigned" if s in used else "available")
                          for s, p in zip(sites["site_id"], sites["p64"])]
        sites["color"] = [ui.rgb(ui.STATUS["critical"]) if st_ == "above wind threshold" else
                          (ui.rgb(ui.STATUS["good"]) if st_ == "assigned" else ui.rgb(ui.INK["muted"])) for st_ in sites["state"]]
        sites["label"] = [f"{n} | p64 {p:.2f}" for n, p in zip(sites["name"], sites["p64"])]
        layers = [
            pdk.Layer("GeoJsonLayer", geo, pickable=True, stroked=True, filled=True, get_fill_color="properties.fill",
                      get_line_color="properties.line", get_line_width="properties.lw", line_width_units="'pixels'"),
            pdk.Layer("ScatterplotLayer", sites, get_position=["lon", "lat"], get_fill_color="color", get_radius=9,
                      radius_units="'pixels'", stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2, pickable=True),
            pdk.Layer("TextLayer", sites, get_position=["lon", "lat"], get_text="label", get_size=12, get_color=[11, 11, 11],
                      get_pixel_offset=[12, 0], get_text_anchor="'start'", background=True,
                      font_family="'Helvetica Neue, Helvetica, Arial, sans-serif'",
                      get_background_color=[255, 255, 255, 220]),
        ]
        st.pydeck_chart(ui.deck(layers, ui.FL_VIEW, {"html": "<b>{county}</b><br/>P50 {out50} out ({frac})<br/>"
                                                             "P10 {out10} · P90 {out90}<br/>{name} {state}",
                                                     "style": ui.TOOLTIP_STYLE}, height=470))
        st.caption(f"County colour = P50 fraction of customers out, scaled to the highest county at this advisory "
                   f"({top_frac:.1%}); light = low, dark = high. Black outline = Lee and Charlotte (the zones). "
                   "Staging sites: 🟢 assigned · ⚪ available · 🔴 above the 64 kt threshold. Hover a county for P10/P50/P90.")
    with right:
        st.markdown("**Staging plan by zone** (C2)")
        zp = pd.DataFrame(plan["zones"])
        zp["assigned"] = [f"{'⚠️ DISPLACED → ' if d else ''}{s or 'none below threshold: P1 decides'}"
                          for d, s in zip(zp["displaced"], zp["assigned_site_name"])]
        zp["mapped"] = [f"{n} ({p:.2f})" for n, p in zip(zp["mapped_site_name"], zp["mapped_site_p64"])]
        st.dataframe(zp[["zone_name", "customers_out_p50", "customers_out_p90", "crew_hours_p90", "mapped", "assigned"]],
                     hide_index=True, width="stretch",
                     column_config={"zone_name": "Zone", "customers_out_p50": st.column_config.NumberColumn("P50 out", format="%,d"),
                                    "customers_out_p90": st.column_config.NumberColumn("P90 out", format="%,d"),
                                    "crew_hours_p50": st.column_config.NumberColumn("P50 crew-h", format="%,d"),
                                    "crew_hours_p90": st.column_config.NumberColumn("P90 crew-h", format="%,d"),
                                    "mapped": "Mapped site (p64)", "assigned": "Assigned site"})
        for z in plan["zones"]:
            st.caption(f"{z['zone_name']}: {z['note']}")
        st.caption(f":violet-badge[PLACEHOLDER] staging site list (the real list lives in SGW's storm plan, A19). "
                   f"Site displaced if its 64 kt probability > {C['site_wind_threshold']:.2f}.")
        st.markdown("**County estimates** (C1, P10 / P50 / P90 customers out) — zones and the statewide top 8")
        top = est.sort_values("out_p50", ascending=False)
        show = pd.concat([top[top["study_zone"]], top[~top["study_zone"]].head(8)])
        st.dataframe(show[["county", "customers", "p64", "p34", "out_p10", "out_p50", "out_p90"]], hide_index=True, width="stretch",
                     column_config={"county": "County", "customers": st.column_config.NumberColumn("Customers", format="%,d"),
                                    "p64": st.column_config.NumberColumn("p64", format="%.2f"),
                                    "p34": st.column_config.NumberColumn("p34", format="%.2f"),
                                    "out_p10": st.column_config.NumberColumn("P10", format="%,d"),
                                    "out_p50": st.column_config.NumberColumn("P50", format="%,d"),
                                    "out_p90": st.column_config.NumberColumn("P90", format="%,d")}, height=330)

    st.markdown("#### Sign-off (FR23)")
    with st.container(border=True):
        choice = st.radio("Storm Director action", ["Approve", "Edit", "Override"], horizontal=True, key=f"p1act_{A['record_id']}")
        with st.form(f"p1form_{A['record_id']}", clear_on_submit=True):
            actor = st.text_input("Signed by", ACTORS["P1"])
            payload = {}
            if choice == "Edit":
                payload["mutual_aid_crews"] = st.number_input("Crews to request", min_value=0, value=int(plan["mutual_aid_crews"]), step=50)
                payload["staging_note"] = st.text_input("Staging change (optional)")
            if choice == "Override":
                payload["plan_as_decided"] = st.text_area("Plan as decided")
            reason = st.text_area("Reason" + (" (required)" if choice != "Approve" else " (optional)"))
            if st.form_submit_button(f"{choice} and write to the record", type="primary"):
                try:
                    record.add_action(conn(), choice.upper(), actor, A["record_id"], reason or None, payload or None)
                    st.success(f"{choice} logged against `{A['record_id']}`.")
                except ValueError as err:
                    st.error(str(err))
        hist = record.list_actions(conn(), A["record_id"])
        if len(hist):
            st.dataframe(hist[["created_at", "action", "actor", "reason", "payload_json"]], hide_index=True, width="stretch")
        else:
            st.caption("No action yet on this record: status PROPOSED.")

    with st.expander("How this was computed (inputs and rule)"):
        st.markdown(f"- Advisory snapshot files: `{', '.join(A['inputs']['snapshot_files'])}` (resolution {A['resolution']})\n"
                    f"- C1 features per county: p64, p34 (area-weighted, cumulative 0-120 h), customers (EAGLE-I MCC), coastal flag\n"
                    f"- C1 model: `{A['model_version']}`; estimates are not editable, P1 acts through C2\n"
                    f"- C2: {plan['formula']}\n- Units: {UNITS}")


# ---------------------------------------------------------------- P2
with tabs[1]:
    st.markdown("#### Decision B — de-energize a surge-zone substation (T-12h to T-6h)")
    if B is None:
        st.info(f"Decision B does not run for this advisory: no P-Surge grid is loaded and it is before T-"
                f"{C['decisionB_starts_hours']}h (the watch window). P2 sees nothing from C5. Move the advisory slider "
                "closer to landfall, or pick Ian (real P-Surge).", icon=":material/schedule:")
    else:
        rc = B["recommendation"]
        est_b = pd.DataFrame(B["estimates"]["substations"])
        rec_b = pd.DataFrame(rc["substations"])
        sb = est_b.merge(rec_b, on="asset_id")
        src = rc["prob_source"]
        st.markdown(("Probability :blue-badge[PSURGE] P-Surge as issued, P(surge > "
                     f"{B['inputs']['psurge_threshold_ft']} ft above ground)" if src == "PSURGE" else
                     f"Probability :orange-badge[{src}] FEMA zone + BFE fallback") +
                    f" · Height :orange-badge[ESTIMATED] ground (3DEP) + {C['height_offset_m']} m placeholder offset"
                    f" · Threshold in force **{rc['threshold_in_force']:.2f}** = {rc['threshold']:.2f} + margin "
                    f"{rc['margin']:.2f} (raised because height is ESTIMATED)"
                    " · Flood sensor :gray-badge[BLANK] no historian in the prototype (FR21)")
        held = sb[(sb["recommendation"] == "WATCH") & sb["above_threshold"]]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Screening set", len(sb), help="Substations in FEMA zones " + ", ".join(C["screening_zones"]))
        m2.metric("🔴 DE-ENERGIZE", int((sb["recommendation"] == "DE-ENERGIZE").sum()))
        m3.metric("🟡 WATCH: critical load", len(held), help="Above threshold, but a hospital or pumping station on the "
                                                             "PLACEHOLDER FEED has unknown backup (A18)")
        m4.metric("🟡 WATCH: below threshold", int((sb["recommendation"] == "WATCH").sum()) - len(held))
        mc1 = mc2 = st.container()
        with mc1:
            show_zones = st.toggle("FEMA flood zones", value=True)
            show_hwm = st.toggle("USGS high-water marks (Ian, observed)", value=(storm == "ian"), disabled=(storm != "ian"))
            layers = []
            if show_zones:
                zgeo = load_geo("flood_sfha")
                for f in zgeo["features"]:
                    z = f["properties"]["FLD_ZONE"]
                    f["properties"]["fill"] = ui.rgb(ui.SEQ[5] if z == "VE" else ui.SEQ[2], 90)
                layers.append(pdk.Layer("GeoJsonLayer", zgeo, filled=True, stroked=False, get_fill_color="properties.fill"))
            if show_hwm and storm == "ian":
                h = load_parquet("hwm_ian").copy()
                h["txt"] = [f"HWM {w:.2f} m NAVD88" for w in h["water_elev_m_navd88"]]
                layers.append(pdk.Layer("ScatterplotLayer", h, get_position=["lon", "lat"], get_radius=4, radius_units="'pixels'",
                                        get_fill_color=ui.rgb(ui.SEQ[11], 220), pickable=True))
            sb["color"] = [ui.rgb(ui.STATUS["critical"]) if r == "DE-ENERGIZE" else ui.rgb(ui.STATUS["warning"]) for r in sb["recommendation"]]
            sb["txt"] = [f"{n or a} · {r} · p {p:.2f}" for n, a, r, p in zip(sb["name"], sb["asset_id"], sb["recommendation"], sb["prob"])]
            layers.append(pdk.Layer("ScatterplotLayer", sb, get_position=["lon", "lat"], get_radius=8, radius_units="'pixels'",
                                    get_fill_color="color", stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2,
                                    pickable=True))
            st.pydeck_chart(ui.deck(layers, ui.STUDY_VIEW, {"html": "{txt}", "style": ui.TOOLTIP_STYLE}, height=440))
            st.caption("🔴 DE-ENERGIZE · 🟡 WATCH · blue fill = FEMA special flood hazard area (darker = VE) · dark dots = "
                       "Ian high-water marks. Hover for values.")
        with mc2:
            sb["rec"] = sb["recommendation"].map(ui.REC_ICON)
            st.dataframe(sb.sort_values("prob", ascending=False)[["asset_id", "name", "fema_zone", "ground_elev_m",
                                                                  "switchgear_m_navd88", "prob", "prob_source", "threshold",
                                                                  "critical_load_check", "rec", "reason"]],
                         hide_index=True, width="stretch", height=440,
                         column_config={"asset_id": "ID", "name": "Substation", "fema_zone": "Zone",
                                        "ground_elev_m": st.column_config.NumberColumn("Ground m", format="%.2f"),
                                        "switchgear_m_navd88": st.column_config.NumberColumn("Switchgear m (ESTIMATED)", format="%.2f"),
                                        "prob": st.column_config.ProgressColumn("P(inundation)", min_value=0.0, max_value=1.0, format="%.2f"),
                                        "prob_source": "Source", "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
                                        "critical_load_check": "Critical load", "rec": "Recommendation", "reason": "Why"})
        st.markdown("#### Sign-off (FR24) — the control room switches in its own systems; the platform never touches SCADA")
        with st.form(f"p2form_{B['record_id']}", clear_on_submit=True):
            opts = sb["asset_id"].tolist()
            picked = st.multiselect("Substations", opts, format_func=lambda a: f"{a} · {sb.set_index('asset_id').loc[a, 'name'] or ''} · "
                                                                          f"{sb.set_index('asset_id').loc[a, 'recommendation']}")
            actor = st.text_input("Signed by", ACTORS["P2"])
            reason = st.text_input("Reason (required to decline)")
            s1, s2 = st.columns(2)
            do_sign = s1.form_submit_button("Sign off recommendation", type="primary")
            do_decline = s2.form_submit_button("Decline")
            if (do_sign or do_decline) and picked:
                try:
                    for a in picked:
                        record.add_action(conn(), "SIGN_OFF" if do_sign else "DECLINE", actor, B["record_id"], reason or None,
                                          {"asset_id": a, "recommendation": sb.set_index("asset_id").loc[a, "recommendation"]})
                    st.success(f"{'Signed off' if do_sign else 'Declined'} {len(picked)} substation(s) on `{B['record_id']}`.")
                except ValueError as err:
                    st.error(str(err))
        hist = record.list_actions(conn(), B["record_id"])
        if len(hist):
            st.dataframe(hist[["created_at", "action", "actor", "reason", "payload_json"]], hide_index=True, width="stretch")
        if storm == "ian":
            f22 = backtest()["fr22"]
            if pd.Timestamp(f22["advisory_time"]) == t:
                st.markdown("#### FR22 — how this advisory's flags compare with what Ian did (USGS high-water marks)")
                rows = []
                for tk, lab in f22["truth_defs"].items():
                    for v in ("psurge_screening_set", "static_screening_set"):
                        r = f22["results"][tk][v]
                        rows.append({"truth": lab, "variant": "P-Surge" if v.startswith("psurge") else "STATIC fallback",
                                     "substations observed": r["with_hwm_observation"], "observed flooded": r["observed_flooded"],
                                     "flag precision": r["inundation_flag"]["precision"], "flag recall": r["inundation_flag"]["recall"]})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("HWMs are sparse (median substation ~4 km from one), so read each row with its sample size. "
                           "PRD M4 target: precision 0.7, recall 0.8.")


# ---------------------------------------------------------------- P3
with tabs[2]:
    st.markdown(f"#### Field operations — :gray-badge[{EXPOSURE_LABEL}] (read-only, FR32)")
    st.caption("Each asset takes C1's P50 for its 0.1° cell × static weights (FEMA zone, ground elevation, voltage class). "
               "Validated only at county level until Phase 2 retrains on SGW's asset outcomes. Public attributes only; "
               "maintenance attributes enter in production (PRD Appendix A).")
    last = actions_for(A["record_id"])
    st.markdown("**Staging plan as P1 left it:** " + (f"{last[0]['action']} by {last[0]['actor']}" +
                (f" — reason: {last[0]['reason']}" if last[0]["reason"] else "") if last else "not yet signed (PROPOSED)") +
                " · " + "; ".join(f"{z['zone_name']} → {z.get('assigned_site_name') or 'P1 decides'}" for z in plan["zones"]))
    assets = pd.DataFrame(A["estimates"]["assets"])
    types = st.multiselect("Asset types", sorted(assets["type"].unique()), default=sorted(assets["type"].unique()))
    av = assets[assets["type"].isin(types)].copy()
    mx = max(av["score"].max(), 1e-9) if len(av) else 1
    av["fill"] = [ui.seq_rgb(s / mx, 230) for s in av["score"]]
    av["txt"] = [f"#{r} {n or a} ({t_}) · score {s:.3f}" for r, n, a, t_, s in zip(av["rank"], av["name"], av["asset_id"], av["type"], av["score"])]
    p3l, p3r = st.columns([1, 1.2])
    with p3l:
        st.pydeck_chart(ui.deck([pdk.Layer("ScatterplotLayer", av, get_position=["lon", "lat"], get_radius=6, radius_units="'pixels'",
                                           get_fill_color="fill", stroked=True, get_line_color=[255, 255, 255],
                                           line_width_min_pixels=1, pickable=True)],
                                ui.STUDY_VIEW, {"html": "{txt}", "style": ui.TOOLTIP_STYLE}, height=440))
        st.caption("Colour = exposure score (light low, dark high), relative to the highest score shown.")
    with p3r:
        st.dataframe(av[["rank", "asset_id", "type", "name", "score", "cell_prob", "fema_zone", "w_zone", "ground_elev_m",
                         "w_elev", "voltage_class", "w_voltage", "source_vintage"]], hide_index=True, width="stretch", height=440,
                     column_config={"rank": "Rank", "cell_prob": st.column_config.NumberColumn("Cell P50", format="%.3f"),
                                    "score": st.column_config.NumberColumn("Score", format="%.3f"),
                                    "ground_elev_m": st.column_config.NumberColumn("Ground m", format="%.1f"),
                                    "source_vintage": "Vintage"})


# ---------------------------------------------------------------- P4
with tabs[3]:
    st.markdown("#### Water operations — pumping stations and plants inherit their feeding substation's exposure (C4)")
    st.markdown(":violet-badge[PLACEHOLDER FEED] feed = nearest substation by straight line (the recorded feed comes from "
                f"Phase 0; production never assigns by distance, R10). Beyond {C['placeholder_feed_max_km']:g} km: "
                "*no plausible feed*. Backup feeds unknown (U4). Read-only (FR32).")
    dep = pd.DataFrame(A["estimates"]["dependents"])
    dep["feed_name"] = dep["feed_name"].where(dep["feed_name"].fillna("").str.len() > 0, dep["feed_asset_id"])
    dtypes = st.multiselect("Dependent types", ["pumping", "plant", "hospital"], default=["pumping", "plant"])
    dv = dep[dep["type"].isin(dtypes)].copy()
    st.dataframe(dv[["asset_id", "type", "name", "feed_label", "feed_name", "feed_distance_km", "inherited_score",
                     "critical_load", "backup_status", "source_vintage"]], hide_index=True, width="stretch",
                 column_config={"feed_label": "Feed", "feed_name": "Feeding substation",
                                "feed_distance_km": st.column_config.NumberColumn("km", format="%.1f"),
                                "inherited_score": st.column_config.NumberColumn("Inherited score", format="%.3f")})
    reg = load_parquet("registry").set_index("asset_id")
    links = dv[dv["feed_asset_id"].notna()].copy()
    links["flon"] = links["feed_asset_id"].map(reg["lon"])
    links["flat"] = links["feed_asset_id"].map(reg["lat"])
    mxs = max(dv["inherited_score"].max(), 1e-9) if dv["inherited_score"].notna().any() else 1
    dv["fill"] = [ui.seq_rgb((s or 0) / mxs, 230) if pd.notna(s) else ui.rgb(ui.STATUS["serious"]) for s in dv["inherited_score"]]
    dv["txt"] = [f"{n} ({t_}) ← {f or 'no plausible feed'}" for n, t_, f in zip(dv["name"], dv["type"], dv["feed_name"])]
    st.pydeck_chart(ui.deck([
        pdk.Layer("LineLayer", links, get_source_position=["lon", "lat"], get_target_position=["flon", "flat"],
                  get_color=ui.rgb(ui.INK["muted"], 180), get_width=1.5),
        pdk.Layer("ScatterplotLayer", dv, get_position=["lon", "lat"], get_radius=7, radius_units="'pixels'", get_fill_color="fill",
                  stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2, pickable=True)],
        ui.STUDY_VIEW, {"html": "{txt}", "style": ui.TOOLTIP_STYLE}, height=420))
    st.caption("Grey line = PLACEHOLDER FEED to the nearest substation. Colour = inherited score; orange-red = no plausible feed.")


# ---------------------------------------------------------------- P5
with tabs[4]:
    pname = provider.provider_name()
    st.markdown(f"#### Situational briefing (C6) — mode: `{provider.describe()}`")
    st.caption("C6 reads the record only after the run has written it, copies every number from it and shows the record "
               "IDs used; it cannot recommend, change a threshold or write to any log (FR33-FR35). Switch with one "
               "setting: `LLM_PROVIDER` in `.env` (template · anthropic · ollama · groq · gemini).")
    ids = [A["record_id"]] + ([B["record_id"]] if B else [])
    draft_key = f"draft_{'_'.join(ids)}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = template.briefing(A, B, actions_for(A["record_id"]))
    b1, b2 = st.columns([1, 3])
    if b1.button("Redraft from template (FR36)"):
        st.session_state[draft_key] = template.briefing(A, B, actions_for(A["record_id"]))
    if pname != "template" and b2.button(f"Draft with LLM ({provider.describe()})"):
        ctx = prompts.record_context(A, B, actions_for(A["record_id"]))
        try:
            out = provider.complete(prompts.SYSTEM, f"DECISION RECORD:\n{ctx}\n\nTASK:\n{prompts.BRIEFING_TASK}")
            st.session_state[draft_key] = out
            bad = prompts.unsupported_numbers(out, ctx)
            st.session_state[draft_key + "_check"] = bad
        except provider.LLMError as err:
            st.error(f"{err}. Falling back to the template (FR36).")
    bad = st.session_state.get(draft_key + "_check")
    if bad:
        st.warning(f"FR35 check: numbers in the draft not found in the record: {', '.join(bad)}. Edit before sending.")
    elif bad == []:
        st.success("FR35 check: every number in the draft appears in the record.")
    text = st.text_area("Draft (P5 edits, then sends outside the system)", key=draft_key, height=430)
    st.caption("Record IDs used: " + ", ".join(f"`{i}`" for i in ids))
    if st.button("Mark as sent (logs draft hash and record IDs)"):
        record.add_action(conn(), "BRIEFING_SENT", ACTORS["P5"], A["record_id"], None,
                          {"record_ids": ids, "draft_sha256": hashlib.sha256(text.encode()).hexdigest()[:16]})
        st.success("Logged.")

    st.markdown("#### Ask the record (FR34)")
    qc1, qc2, qc3 = st.columns(3)
    preset = None
    if qc1.button("Why is Gladiolus on watch?"):
        preset = "Why is Gladiolus on watch?"
    if qc2.button("How many mutual aid crews, and why?"):
        preset = "How many mutual aid crews, and why?"
    if qc3.button("Refusal test: What will the weather be tomorrow?"):
        preset = "What will the weather be tomorrow?"
    question = st.text_input("Question", value=preset or "", key="q_input" if preset is None else f"q_{preset}")
    if question:
        reg_df = load_parquet("registry")
        ans, rows, used = qa.answer(question, A, B, reg_df)
        if pname != "template" and ans != qa.NOT_IN_RECORD:
            ctx = prompts.record_context(A, B, actions_for(A["record_id"]))
            try:
                llm = provider.complete(prompts.SYSTEM, f"DECISION RECORD:\n{ctx}\n\nQUESTION: {question}")
                bad_q = prompts.unsupported_numbers(llm, ctx)
                st.markdown(llm)
                if bad_q:
                    st.warning(f"FR35 check: numbers not in the record: {', '.join(bad_q)}")
            except provider.LLMError as err:
                st.error(str(err))
        (st.info if ans == qa.NOT_IN_RECORD else st.markdown)(ans)
        if rows is not None:
            st.caption("Source rows (" + ", ".join(f"`{u}`" for u in used) + ")")
            st.dataframe(rows, hide_index=True, width="stretch")


# ---------------------------------------------------------------- Backtest
with tabs[5]:
    bt_all = backtest()
    f16 = bt_all["fr16"]
    tab = json.loads((PROCESSED / "backtest_tabpfn.json").read_text()) if (PROCESSED / "backtest_tabpfn.json").exists() else None
    st.markdown(f"#### FR16 — C1 on Milton (held out), advisory {f16['advisory_time'][:16]}Z (T-{f16['hours_to_landfall']:.1f}h), 67 counties")
    rows = []
    for k, lab in [("mae_customers", "MAE, customers out per county"), ("spearman", "Spearman, county order"),
                   ("top_n_match", "Top-5 match (target ≥ 0.70)"), ("p90_coverage", "P90 coverage (target ≥ 0.85)"),
                   ("p10_p90_coverage", "P10-P90 coverage")]:
        f = (lambda v: f"{v:,.0f}") if k == "mae_customers" else (lambda v: f"{v:.3f}")
        rows.append({"Metric": lab, "GLM (active)": f(f16["glm"][k]), "LightGBM": f(f16["lgbm"][k]),
                     "TabPFN v2 (benchmark)": f(tab[k]) if tab else "not run"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Selection (FR30): {f16['selection']['reason']}. Active: `{f16['selection']['model_version']}`. "
               + ("TabPFN column: **Built with PriorLabs-TabPFN** (v2 regressor, benchmark only: no monotone constraints)." if tab else ""))
    mc = load_parquet("backtest_milton_counties")
    if tab:
        tc = load_parquet("backtest_tabpfn_counties").merge(mc[mc["model"] == "glm"][["fips", "county", "customers", "obs_out"]], on="fips")
        for q in ("p10", "p50", "p90"):
            tc[f"pred_out_{q}"] = tc[q] * tc["customers"]
        mc = pd.concat([mc, tc.assign(model="tabpfn")], ignore_index=True)
    pick = st.radio("Model", [m for m in ("glm", "lgbm", "tabpfn") if m in set(mc["model"])], horizontal=True,
                    format_func=ui.SERIES_LABEL.get)
    d = mc[mc["model"] == pick].copy()
    g1, g2 = st.columns([1.2, 1])
    with g1:
        st.altair_chart(ui.pred_vs_obs(d, pick), width="stretch")
        t5 = pd.DataFrame({"Observed top 5": d.nlargest(5, "obs_out")["county"].tolist(),
                           "Predicted top 5 (P50)": d.nlargest(5, "pred_out_p50")["county"].tolist()})
        st.dataframe(t5, hide_index=True, width="stretch")
    with g2:
        lt = load_parquet("backtest_leadtime")
        if (PROCESSED / "backtest_tabpfn_leadtime.parquet").exists():
            lt = pd.concat([lt, load_parquet("backtest_tabpfn_leadtime")], ignore_index=True)
        st.altair_chart(ui.leadtime(lt, "top_n_match", "Top-5 county match by lead time", C["banner_topn_match"]), width="stretch")
        st.altair_chart(ui.leadtime(lt, "p90_coverage", "P90 coverage by lead time", C["banner_p90_coverage"]), width="stretch")
    st.caption("Dashed vertical line = T-72h, the Decision A advisory. Dashed horizontal line = FR17 target. "
               "Full tables: docs/backtest_report.md.")
    st.markdown("#### FR22 — Decision B on Ian at T-12h (USGS high-water marks)")
    f22 = bt_all["fr22"]
    rows = []
    short = {"max_500m": "any HWM ≤ 500 m (PLAN)", "idw_1km": "IDW ≤ 1 km", "idw_3km": "IDW ≤ 3 km"}
    for tk, lab in f22["truth_defs"].items():
        lab = short.get(tk, lab)
        for v, vl in [("psurge_screening_set", "P-Surge, screening set"), ("static_screening_set", "STATIC, screening set"),
                      ("psurge_all_substations", "P-Surge, all substations")]:
            r = f22["results"][tk][v]
            rows.append({"Truth": lab, "Variant": vl, "Observed": r["with_hwm_observation"], "Flooded": r["observed_flooded"],
                         "Flag precision": r["inundation_flag"]["precision"], "Flag recall": r["inundation_flag"]["recall"],
                         "DE-ENERGIZE precision": r["de_energize"]["precision"], "DE-ENERGIZE recall": r["de_energize"]["recall"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ---------------------------------------------------------------- Record
with tabs[6]:
    st.markdown("#### Decision record (FR25) — append-only; every row replays exactly")
    recs = record.list_records(conn())
    st.dataframe(recs, hide_index=True, width="stretch", height=260)
    if len(recs):
        rid = st.selectbox("Record", recs["record_id"].tolist(),
                           index=recs["record_id"].tolist().index(A["record_id"]) if A["record_id"] in set(recs["record_id"]) else 0)
        r = record.get_record(conn(), rid)
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"**Decision** {r['decision']} · {r['storm_id']}  \n**Advisory** {r['advisory_time'][:16]}Z")
        k2.markdown(f"**Config** `{r['config_version']}`  \n**Model** `{r['model_version']}`")
        k3.markdown(f"**Status** {r['status']}  \n**Output hash** `{r['output_hash']}`")
        k4.markdown(f"**Height** {r['height_source'] or '—'} · **Prob** {r['prob_source'] or '—'}  \n"
                    f"**Sensor** {('BLANK' if r['sensor_value'] is None else r['sensor_value']) if r['decision'] == 'B' else '—'} · **Crew-h** "
                    f"{ui.fmt(r['crew_hours_entered'])}")
        if st.button("Rerun this record", type="primary"):
            rr = replay(rid, conn=conn())
            if rr["identical"]:
                st.success(f"Identical output: original `{rr['original_hash']}` = replay `{rr['replay_hash']}` "
                           f"(config `{rr['config_version']}`, model `{rr['model_version']}`).", icon=":material/verified:")
            else:
                st.error(f"Replay differs or could not run: {rr}")
        st.download_button("Export this record (JSON, FR27 degraded mode)", json.dumps(
            {k: r[k] for k in r if not k.endswith("_json")}, indent=2, default=str), file_name=f"{rid}.json")
        with st.expander("Inputs"):
            st.json(r["inputs"], expanded=False)
        with st.expander("Recommendation"):
            st.json(r["recommendation"], expanded=False)
        with st.expander("Actions on this record"):
            st.dataframe(record.list_actions(conn(), rid), hide_index=True, width="stretch")


# ---------------------------------------------------------------- Governance
with tabs[7]:
    st.markdown("#### Governance (PRD Section 8) — prototype: roles are labels, no SSO (A20)")
    gl, gr = st.columns(2)
    with gl:
        st.markdown("**Storm operations (FR29)** — declaration freezes configuration and starts the 99.9% window")
        st.markdown(f"State: **{'DECLARED' if ops['declared'] else 'not declared'}**" +
                    (f" by {ops.get('actor')} at {ops.get('created_at')}" if ops.get("actor") else ""))
        with st.form("ops"):
            why = st.text_input("Reason (required to end)")
            o1, o2 = st.columns(2)
            if o1.form_submit_button("Declare storm operations (P1)"):
                record.add_action(conn(), "DECLARE_STORM_OPS", ACTORS["P1"], None, why or "NHC watch touching the territory",
                                  {"config_version": C["_version"]})
                st.rerun()
            if o2.form_submit_button("End storm operations (P1)"):
                try:
                    record.add_action(conn(), "END_STORM_OPS", ACTORS["P1"], None, why)
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))
        st.markdown("**Model version (FR30)** — P6 approves between storms; during a storm only P1 may withdraw, "
                    "with a reason; the system falls back to the previous approved version.")
        st.markdown(f"Approved active: `{selection()['model_version']}` · in service: `{model_version(kind)}`")
        with st.form("model"):
            why = st.text_input("Reason (required)")
            w1, w2 = st.columns(2)
            if w1.form_submit_button(f"Withdraw `{selection()['active']}` (P1)"):
                try:
                    record.add_action(conn(), "WITHDRAW_MODEL", ACTORS["P1"], None, why, {"model_kind": selection()["active"]})
                    st.cache_data.clear()
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))
            if w2.form_submit_button("Restore after post-storm review (P6)"):
                try:
                    record.add_action(conn(), "RESTORE_MODEL", ACTORS["P6"], None, why or "post-storm review",
                                      {"model_kind": selection()["active"]})
                    st.cache_data.clear()
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))
    with gr:
        st.markdown(f"**Configuration `{C['_version']}` (FR28)** — every value; any edit is a new version")
        flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in C.items() if not k.startswith("_")}
        st.dataframe(pd.DataFrame({"setting": list(flat), "value": [str(v) for v in flat.values()]}), hide_index=True,
                     width="stretch", height=520)
    st.markdown("**Audit log** (all actions, newest first)")
    st.dataframe(record.list_actions(conn()), hide_index=True, width="stretch", height=240)
