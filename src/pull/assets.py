"""D4, D5, D5b, D12 — public asset layers for the study bbox.

Sources actually used (PLAN §4 primary was the NASA NCCS HIFLD mirror, unreachable on 2026-09-26;
HIFLD's own Electric Substations layer is no longer published — fallback per PLAN §6):
  substations   OSM power=substation (Overpass)                         -> raw/assets/osm_substations.json
  lines         HIFLD Electric_Power_Transmission_Lines (ArcGIS Online)   -> raw/assets/hifld_lines.geojson
  wastewater    EPA FRS WWTPs via FEMA Critical Infrastructure service   -> raw/assets/epa_frs_wwtp.geojson
                + OSM man_made=wastewater_plant supplement               -> raw/assets/osm_wastewater.json
  pumping       OSM man_made=pumping_station                             -> raw/assets/osm_pumping.json
  hospitals     HIFLD Hospitals via FEMA Critical Infrastructure service -> raw/assets/hifld_hospitals.geojson
"""
from __future__ import annotations

import json
import time

from .common import BBOX, RAW, arcgis_query, overpass, write_pull_log

OUT = RAW / "assets"
HIFLD_LINES = "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Power_Transmission_Lines/FeatureServer/0"
FEMA_CI = "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/Critical_Infrastructure_Map_Service/FeatureServer"
FEMA_CI_WWTP = f"{FEMA_CI}/8"       # EPA FRS Wastewater Treatment Plants
FEMA_CI_HOSPITALS = f"{FEMA_CI}/10"  # HIFLD Hospitals

# Overpass bbox order is (south, west, north, east)
_OB = f"{BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]}"
OSM_QUERIES = {
    "osm_substations": f'[out:json][timeout:90];nwr["power"="substation"]({_OB});out center tags;',
    "osm_pumping": f'[out:json][timeout:90];nwr["man_made"="pumping_station"]({_OB});out center tags;',
    "osm_wastewater": f'[out:json][timeout:90];nwr["man_made"="wastewater_plant"]({_OB});out center tags;',
}
ARCGIS_LAYERS = {
    "hifld_lines": HIFLD_LINES,
    "epa_frs_wwtp": FEMA_CI_WWTP,
    "hifld_hospitals": FEMA_CI_HOSPITALS,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, q in OSM_QUERIES.items():
        dest = OUT / f"{name}.json"
        if dest.exists():
            results[name] = {"skipped": "exists"}
            continue
        data, ep = overpass(q)
        time.sleep(5)  # be polite to shared Overpass instances
        dest.write_text(json.dumps(data))
        results[name] = {"count": len(data.get("elements", [])), "endpoint": ep,
                         "osm_base": data.get("osm3s", {}).get("timestamp_osm_base")}
        print(f"[assets] {name}: {results[name]}", flush=True)
    for name, url in ARCGIS_LAYERS.items():
        dest = OUT / f"{name}.geojson"
        if dest.exists():
            results[name] = {"skipped": "exists"}
            continue
        fc = arcgis_query(url)
        dest.write_text(json.dumps(fc))
        results[name] = {"count": len(fc["features"]), "url": url}
        print(f"[assets] {name}: {results[name]['count']} features", flush=True)
    write_pull_log("D4_D5_D12", {"results": results})


if __name__ == "__main__":
    main()
