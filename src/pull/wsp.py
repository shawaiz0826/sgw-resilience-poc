"""D2 — NHC wind speed probability archive (34/50/64 kt, cumulative 0-120 h), one zip per 6-h cycle.

The archive page (gis/archive_wsp.php) lists per cycle `YYYYMMDDHH_wsp_120hr5km.zip` (0.05 deg contour
bands) and `..._120hrhalfDeg.zip`. PLAN §4 named `tenthDeg`, which the archive only carries for early
years; for 2022-2024 the high-resolution product is `5km`. We take 5km and fall back to halfDeg, and
record the resolution per snapshot (R4). Products are basin-wide (not storm-specific): the advisory
store keys them by storm and cycle time.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import requests

from .common import RAW, STORMS, download, get, write_pull_log

BASE = "https://www.nhc.noaa.gov/gis/"
INDEX = BASE + "archive_wsp.php"
OUT = RAW / "wsp"
RESOLUTIONS = ["5km", "halfDeg"]  # preference order


def _cycles(start: str, end: str):
    t = datetime.fromisoformat(start.replace("Z", "+00:00"))
    stop = datetime.fromisoformat(end.replace("Z", "+00:00"))
    while t <= stop:
        yield t
        t += timedelta(hours=6)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        html = get(INDEX, timeout=120).text
    except requests.RequestException as e:
        if not any(OUT.glob("*/*.zip")):
            raise
        print(f"[wsp] NHC unreachable ({e.__class__.__name__}); keeping cached snapshots in {OUT}")
        return
    links = set(re.findall(r'href="(forecast/archive/\d{10}_wsp_120hr\w+\.zip)"', html))
    results = {}
    for storm_id, s in STORMS.items():
        got, missing = [], []
        for t in _cycles(s["wsp_start"], s["wsp_end"]):
            stamp = t.strftime("%Y%m%d%H")
            for res in RESOLUTIONS:
                rel = f"forecast/archive/{stamp}_wsp_120hr{res}.zip"
                if rel in links:
                    download(BASE + rel, OUT / storm_id / f"{stamp}_wsp_120hr{res}.zip")
                    got.append({"advisory_time": t.strftime("%Y-%m-%dT%H:%MZ"), "resolution": res})
                    break
            else:
                missing.append(stamp)
        results[storm_id] = {"cycles": len(got), "missing": missing,
                             "resolutions": sorted({g["resolution"] for g in got})}
        print(f"[wsp] {storm_id}: {results[storm_id]}", flush=True)
    write_pull_log("D2", {"source": INDEX, "results": results})


if __name__ == "__main__":
    main()
