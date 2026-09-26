"""D3 — NHC P-Surge for Ian (AL092022), from the GIS archive (gis/archive_psurge_results.php).

Each file `al092022_psurgeN_YYYYMMDDHH.zip` is a contour shapefile of the cumulative (to +78 h)
probability that storm surge exceeds N feet ABOVE GROUND LEVEL (GRIB2 metadata: "Prob of Hurricane
Storm Surge > 0.91 m", "height level above ground"). C5 compares against switchgear height above
ground, so the configured offset maps straight onto a threshold N. We pull N = 1..6 ft for every
cycle so the offset stays a configurable value (FR28) without a re-pull.
"""
from __future__ import annotations

import re

import requests

from .common import RAW, download, get, write_pull_log

PAGE = "https://www.nhc.noaa.gov/gis/archive_psurge_results.php?id=al09&year=2022&name=Hurricane%20IAN"
BASE = "https://www.nhc.noaa.gov/gis/"
THRESHOLDS_FT = range(1, 7)
OUT = RAW / "psurge" / "ian"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        html = get(PAGE, timeout=120).text
    except requests.RequestException as e:
        have = sorted(p.name for p in OUT.glob("*.zip"))
        if not have:
            raise
        # Offline or NHC unreachable: keep the snapshots already on disk and record what they cover.
        got = sorted({int(re.match(r"al092022_psurge(\d+)_", n).group(1)) for n in have})
        cycles = sorted({re.search(r"_(\d{10})\.zip", n).group(1) for n in have})
        write_pull_log("D3", {"source": PAGE, "cycles": cycles, "thresholds_ft": list(THRESHOLDS_FT),
                              "thresholds_on_disk": got, "files": len(have), "failed": ["index unreachable"],
                              "reference": "above ground level, cumulative 78 h", "note": f"cached ({e.__class__.__name__})"})
        print(f"[psurge] NHC unreachable ({e.__class__.__name__}); using {len(have)} cached files, thresholds {got} ft")
        return
    links = re.findall(r'href="(storm_surge/al092022_psurge(\d+)_(\d{10})\.zip)"', html)
    wanted = [(rel, int(n), t) for rel, n, t in links if int(n) in THRESHOLDS_FT]
    cycles = sorted({t for _, _, t in wanted})
    failed = []
    for rel, n, t in wanted:
        try:
            download(BASE + rel, OUT / rel.split("/")[-1])
        except requests.RequestException:
            failed.append(rel.split("/")[-1])
    write_pull_log("D3", {"source": PAGE, "cycles": cycles, "thresholds_ft": list(THRESHOLDS_FT),
                          "files": len(wanted) - len(failed), "failed": failed,
                          "reference": "above ground level, cumulative 78 h"})
    print(f"[psurge] ian: {len(cycles)} cycles, {len(wanted) - len(failed)}/{len(wanted)} files "
          f"({cycles[0]} .. {cycles[-1]}); failed: {len(failed)}")


if __name__ == "__main__":
    main()
