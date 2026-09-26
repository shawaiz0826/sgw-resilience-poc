"""D6 — FEMA NFHL flood hazard zones (S_FLD_HAZ_AR) for the study bbox.

Primary endpoint hazards.fema.gov reset every connection on 2026-09-26. We use Esri's Living Atlas
hosted copy of the same NFHL layer ("USA Flood Hazard Reduced Set"), which carries FLD_ZONE,
ZONE_SUBTY, SFHA_TF, STATIC_BFE, V_DATUM, LEN_UNIT and DFIRM_ID — the fields C5 needs.
The FEMA endpoint is tried first so a machine that can reach it uses the authoritative source.
"""
from __future__ import annotations

import json

import requests

from .common import RAW, arcgis_query, write_pull_log

FEMA_NFHL = "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28"
ESRI_NFHL = "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer/0"
FIELDS = "DFIRM_ID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,V_DATUM,LEN_UNIT,DEPTH"


def main() -> None:
    dest = RAW / "fema" / "nfhl_flood_zones.geojson"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print("[fema] exists")
        return
    used = None
    for url in (FEMA_NFHL, ESRI_NFHL):
        try:
            fc = arcgis_query(url, out_fields=FIELDS, page=500)
            used = url
            break
        except (requests.RequestException, RuntimeError) as e:
            print(f"[fema] {url} failed: {e.__class__.__name__}", flush=True)
    if used is None:
        raise RuntimeError("no NFHL endpoint reachable")
    dest.write_text(json.dumps(fc))
    write_pull_log("D6", {"source": used, "features": len(fc["features"])})
    print(f"[fema] {len(fc['features'])} flood zone polygons from {used}")


if __name__ == "__main__":
    main()
