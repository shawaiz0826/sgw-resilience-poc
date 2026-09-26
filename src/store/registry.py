"""Asset registry for the prototype (PLAN §5A; PRD Appendix A substitutions).

No entity resolution (that is Phase 0): each asset keeps its source identifier. Every row carries its
source and vintage, which the app stamps on screen. Writes (all small, committed):
  processed/registry.parquet     one row per asset, attributes C3/C4/C5 use
  processed/lines.parquet        HIFLD transmission line geometry (map)
  processed/counties.parquet     simplified Florida county polygons (map, C1 aggregation)
  processed/flood_sfha.parquet   FEMA special flood hazard areas, dissolved + simplified (map)
  processed/hwm_ian.parquet      USGS high-water marks, Lee + Charlotte, metres NAVD88 (FR22)
"""
from __future__ import annotations

import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from src.pull.common import PROCESSED, RAW, STUDY_FIPS
from src.pull.elevation import elevations

A = RAW / "assets"
UTM17 = 26917  # NAD83 / UTM 17N, metres, for distances in SW Florida
FT = 0.3048
PLACEHOLDER_FEED_MAX_KM = 10.0  # mirrors config/v1.yaml placeholder_feed_max_km; config wins at decision time
SFHA_ZONES = {"VE", "V", "AE", "A", "AO", "AH", "A99"}


def _pull_log() -> dict:
    return json.loads((RAW / "pull_log.json").read_text())


def _osm_points(name: str) -> tuple[gpd.GeoDataFrame, str]:
    d = json.loads((A / f"{name}.json").read_text())
    vintage = d.get("osm3s", {}).get("timestamp_osm_base", "")[:10]
    rows = []
    for e in d["elements"]:
        lon, lat = (e["lon"], e["lat"]) if e["type"] == "node" else (e["center"]["lon"], e["center"]["lat"])
        rows.append({"source_id": f"osm:{e['type']}/{e['id']}", "tags": e.get("tags", {}), "lon": lon, "lat": lat})
    df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326), vintage


def _max_kv(v) -> float | None:
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(v or ""))]
    return max(nums) / 1000 if nums else None


def voltage_class(kv) -> str:
    if kv is None or (isinstance(kv, float) and np.isnan(kv)):
        return "unknown"
    if kv >= 345:
        return "EHV (>=345 kV)"
    if kv >= 115:
        return "HV (115-230 kV)"
    if kv >= 69:
        return "sub-T (69-114 kV)"
    return "distribution (<69 kV)"


def substations() -> gpd.GeoDataFrame:
    g, vintage = _osm_points("osm_substations")
    g["type"] = "substation"
    g["name"] = g.tags.map(lambda t: t.get("name") or t.get("ref") or "")
    g["operator"] = g.tags.map(lambda t: t.get("operator", ""))
    g["osm_substation"] = g.tags.map(lambda t: t.get("substation", ""))
    g["voltage_kv"] = g.tags.map(lambda t: _max_kv(t.get("voltage")))
    g["source"], g["source_vintage"] = "OpenStreetMap power=substation", f"OSM {vintage}"
    return g


def pumping() -> gpd.GeoDataFrame:
    g, vintage = _osm_points("osm_pumping")
    g["type"] = "pumping"
    g["name"] = g.tags.map(lambda t: t.get("name", "") or "Pumping station")
    g["operator"] = g.tags.map(lambda t: t.get("operator", ""))
    g["source"], g["source_vintage"] = "OpenStreetMap man_made=pumping_station", f"OSM {vintage}"
    return g


def wastewater() -> gpd.GeoDataFrame:
    epa = gpd.read_file(A / "epa_frs_wwtp.geojson")
    epa = epa.assign(source_id="epa_frs:" + epa["REGISTRY_ID"].astype(str), name=epa["CWP_NAME"],
                     operator="", lon=epa.geometry.x, lat=epa.geometry.y,
                     source="EPA FRS WWTP (via FEMA Critical Infrastructure service)", source_vintage="EPA FRS (service current)")
    osm, vintage = _osm_points("osm_wastewater")
    osm["name"] = osm.tags.map(lambda t: t.get("name", "") or "Wastewater plant")
    osm["operator"] = osm.tags.map(lambda t: t.get("operator", ""))
    osm["source"], osm["source_vintage"] = "OpenStreetMap man_made=wastewater_plant", f"OSM {vintage}"
    # Drop OSM plants within 500 m of an EPA plant: same facility, EPA is the record of reference.
    d = gpd.sjoin_nearest(osm.to_crs(UTM17), epa.to_crs(UTM17)[["geometry"]], distance_col="d")
    dup = d.loc[d["d"] < 500].index.unique()
    osm = osm.drop(index=dup)
    g = pd.concat([epa, osm], ignore_index=True)
    g["type"] = "plant"
    return gpd.GeoDataFrame(g, geometry="geometry", crs=4326)


def hospitals() -> gpd.GeoDataFrame:
    h = gpd.read_file(A / "hifld_hospitals.geojson")
    h = h[h["STATUS"].str.upper() == "OPEN"].copy()
    h = h.assign(source_id="hifld:" + h["ID"].astype(str), name=h["NAME"], operator=h["OWNER"],
                 lon=h.geometry.x, lat=h.geometry.y, type="hospital",
                 source="HIFLD Hospitals (via FEMA Critical Infrastructure service)",
                 source_vintage="HIFLD, validated " + pd.to_datetime(h["VAL_DATE"], unit="ms").dt.strftime("%Y-%m"))
    return h


def lines() -> gpd.GeoDataFrame:
    l = gpd.read_file(A / "hifld_lines.geojson")
    l = l.assign(source_id="hifld:" + l["ID"].astype(str), name=l["SUB_1"].fillna("") + " - " + l["SUB_2"].fillna(""),
                 operator=l["OWNER"], voltage_kv=l["VOLTAGE"].where(l["VOLTAGE"] > 0),
                 source="HIFLD Electric Power Transmission Lines (deprecated, ArcGIS Online)",
                 source_vintage="HIFLD, validated " + pd.to_datetime(l["VAL_DATE"], unit="ms").dt.strftime("%Y-%m"))
    return l


def flood_zones() -> gpd.GeoDataFrame:
    cache = RAW / "fema" / "nfhl_flood_zones.parquet"
    if not cache.exists():
        gpd.read_file(RAW / "fema" / "nfhl_flood_zones.geojson").to_parquet(cache)
    z = gpd.read_parquet(cache)
    # R2: sentinel BFE (-9999) is "no value", not a height.
    z["bfe_ft"] = z["STATIC_BFE"].where(z["STATIC_BFE"] > -9000)
    return z


def attach_flood(assets: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    j = gpd.sjoin(assets[["asset_id", "geometry"]], zones[["FLD_ZONE", "ZONE_SUBTY", "SFHA_TF", "bfe_ft", "V_DATUM",
                                                              "LEN_UNIT", "DFIRM_ID", "geometry"]], how="left", predicate="within")
    # A point on a boundary can hit two polygons: keep the most hazardous (highest BFE, then SFHA).
    j["_rank"] = j["SFHA_TF"].eq("T").astype(int) * 1e6 + j["bfe_ft"].fillna(-1)
    j = j.sort_values("_rank", ascending=False).drop_duplicates("asset_id")
    j = j.set_index("asset_id")
    out = assets.set_index("asset_id")
    # Esri's NFHL "reduced set" carries SFHA and shaded X only. Lee and Charlotte are fully mapped, so a
    # point outside every polygon is unshaded Zone X; say so rather than presenting it as a FEMA record.
    inferred = j["FLD_ZONE"].isna()
    out["fema_zone"] = j["FLD_ZONE"].fillna("X")
    out["fema_zone_subtype"] = j["ZONE_SUBTY"].where(~inferred, "Minimal hazard (inferred: outside NFHL reduced set)")
    out["sfha"] = j["SFHA_TF"].eq("T")
    out["bfe_ft"] = j["bfe_ft"]
    out["bfe_datum"] = j["V_DATUM"].where(j["bfe_ft"].notna())
    out["bfe_unit"] = j["LEN_UNIT"].where(j["bfe_ft"].notna())
    out["fema_panel"] = j["DFIRM_ID"]
    # R3: one unit and one datum. Only NAVD88 feet converts; anything else stays NaN and C5 refuses it.
    navd = out["bfe_datum"].astype(str).str.upper().str.contains("NAVD")
    feet = out["bfe_unit"].astype(str).str.lower().str.startswith("f")
    out["bfe_m_navd88"] = np.where(navd & feet, out["bfe_ft"] * FT, np.nan)
    return out.reset_index()


def attach_feeds(assets: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """PLACEHOLDER FEED: nearest substation by straight line (PRD Appendix A; forbidden in production, R10)."""
    subs = assets[assets["type"] == "substation"].to_crs(UTM17)
    dep = assets[assets["type"].isin(["pumping", "plant", "hospital"])].to_crs(UTM17)
    near = gpd.sjoin_nearest(dep[["asset_id", "geometry"]], subs[["asset_id", "geometry"]].rename(columns={"asset_id": "feed"}),
                             distance_col="dist_m").drop_duplicates("asset_id").set_index("asset_id")
    assets = assets.set_index("asset_id")
    km = near["dist_m"] / 1000
    ok = km <= PLACEHOLDER_FEED_MAX_KM
    assets["feed_asset_id"] = near["feed"].where(ok)
    assets["feed_distance_km"] = km.round(2)
    assets["feed_kind"] = pd.Series(np.where(ok, "PLACEHOLDER", "NONE"), index=near.index)
    return assets.reset_index()


def main() -> None:
    counties = gpd.read_file(RAW / "census" / "fl_counties.geojson")
    study = counties[counties["fips"].isin(STUDY_FIPS)]
    parts = [substations(), pumping(), wastewater(), hospitals()]
    cols = ["source_id", "type", "name", "operator", "lon", "lat", "source", "source_vintage", "geometry"]
    pts = pd.concat([p.reindex(columns=cols + ["voltage_kv", "osm_substation"]) for p in parts], ignore_index=True)
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=4326)
    ln = lines()
    mid = ln.geometry.to_crs(UTM17).interpolate(0.5, normalized=True).to_crs(4326)
    ln_pts = gpd.GeoDataFrame(ln[["source_id", "name", "operator", "voltage_kv", "source", "source_vintage"]].assign(
        type="line", lon=mid.x, lat=mid.y), geometry=mid, crs=4326)
    assets = gpd.GeoDataFrame(pd.concat([pts, ln_pts], ignore_index=True), geometry="geometry", crs=4326)
    # Clip to Lee + Charlotte (PLAN §4) and tag county.
    assets = gpd.sjoin(assets, study[["fips", "geometry"]], how="inner", predicate="within").drop(columns="index_right")
    assets = assets.rename(columns={"fips": "county_fips"}).drop_duplicates("source_id").reset_index(drop=True)
    prefix = {"substation": "SUB", "line": "LIN", "plant": "WWT", "pumping": "PMP", "hospital": "HSP"}
    assets = assets.sort_values(["type", "source_id"]).reset_index(drop=True)
    assets["asset_id"] = [f"{prefix[t]}-{i:04d}" for t, i in zip(assets["type"], assets.groupby("type").cumcount() + 1)]
    assets["voltage_class"] = assets["voltage_kv"].map(voltage_class)
    assets = attach_flood(assets, flood_zones())
    assets["ground_elev_m"] = elevations(list(zip(assets["lon"], assets["lat"])))
    assets["elev_source"] = np.where(assets["ground_elev_m"].notna(), "USGS 3DEP via EPQS (m NAVD88)", "UNKNOWN")
    assets = attach_feeds(gpd.GeoDataFrame(assets, geometry="geometry", crs=4326))
    keep = ["asset_id", "type", "name", "operator", "source_id", "source", "source_vintage", "lon", "lat", "county_fips",
            "voltage_kv", "voltage_class", "osm_substation", "fema_zone", "fema_zone_subtype", "sfha", "bfe_ft", "bfe_datum",
            "bfe_unit", "bfe_m_navd88", "fema_panel", "ground_elev_m", "elev_source", "feed_asset_id", "feed_kind",
            "feed_distance_km"]
    reg = pd.DataFrame(assets[keep]).sort_values("asset_id").reset_index(drop=True)
    reg.to_parquet(PROCESSED / "registry.parquet", index=False)

    # Map layers.
    ln_study = gpd.sjoin(ln, study[["geometry"]], how="inner", predicate="intersects").drop_duplicates("source_id")
    ln_study[["source_id", "name", "operator", "voltage_kv", "source_vintage", "geometry"]].to_parquet(PROCESSED / "lines.parquet")
    c = counties.copy()
    c["geometry"] = c.geometry.simplify(0.005, preserve_topology=True)
    c.to_parquet(PROCESSED / "counties.parquet")
    z = flood_zones()
    z = z[z["FLD_ZONE"].isin(SFHA_ZONES)][["FLD_ZONE", "geometry"]].copy()
    z["geometry"] = z.geometry.make_valid()
    z = gpd.clip(z, study.union_all()).dissolve("FLD_ZONE").reset_index()
    z["geometry"] = z.geometry.simplify(0.0005, preserve_topology=True)
    z.to_parquet(PROCESSED / "flood_sfha.parquet")
    hwm_study()
    print(f"[registry] {len(reg)} assets: {reg['type'].value_counts().to_dict()}")


def hwm_study() -> None:
    h = pd.DataFrame(json.loads((RAW / "usgs" / "ian_hwms.json").read_text()))
    h = h[h["verticalDatumName"].eq("NAVD88") & h["elev_ft"].notna()].copy()
    g = gpd.GeoDataFrame(h, geometry=gpd.points_from_xy(h["longitude_dd"], h["latitude_dd"]), crs=4326)
    counties = gpd.read_file(RAW / "census" / "fl_counties.geojson")
    g = gpd.sjoin(g, counties[counties["fips"].isin(STUDY_FIPS)][["fips", "geometry"]], predicate="within")
    out = pd.DataFrame({
        "hwm_id": g["hwm_id"], "lon": g["longitude_dd"], "lat": g["latitude_dd"], "county_fips": g["fips"],
        "water_elev_m_navd88": (g["elev_ft"].astype(float) * FT).round(3),
        "height_above_gnd_m": (pd.to_numeric(g["height_above_gnd"], errors="coerce") * FT).round(3),
        "quality": g["hwmQualityName"], "hwm_type": g["hwmTypeName"], "environment": g["hwm_environment"],
    })
    out.to_parquet(PROCESSED / "hwm_ian.parquet", index=False)
    print(f"[registry] {len(out)} Ian HWMs in Lee + Charlotte")


if __name__ == "__main__":
    main()
