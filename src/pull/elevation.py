"""D9 — ground elevation (USGS 3DEP, metres NAVD88) via the EPQS point service, cached per point.

One call per asset location (a few hundred). EPQS returns -1000000 where it has no data; that
sentinel becomes None (R2), and the registry marks the elevation UNKNOWN.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import requests

from .common import RAW, session

EPQS = "https://epqs.nationalmap.gov/v1/json"
CACHE = RAW / "elevation" / "epqs_cache.json"
SENTINEL = -1_000_000


def _key(lon: float, lat: float) -> str:
    return f"{lon:.6f},{lat:.6f}"


def _one(lon: float, lat: float, sess) -> float | None:
    for _ in range(3):
        try:
            r = sess.get(EPQS, params={"x": lon, "y": lat, "units": "Meters", "wkid": 4326}, timeout=30)
            r.raise_for_status()
            v = float(r.json()["value"])
            return None if v <= SENTINEL + 1 else round(v, 3)
        except (requests.RequestException, ValueError, KeyError):
            continue
    return None


def elevations(points: list[tuple[float, float]]) -> list[float | None]:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = sorted({_key(lon, lat) for lon, lat in points} - set(cache))
    if todo:
        sess = session()
        with ThreadPoolExecutor(8) as ex:
            vals = list(ex.map(lambda k: _one(*map(float, k.split(",")), sess), todo))
        cache.update(dict(zip(todo, vals)))
        CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True))
    return [cache.get(_key(lon, lat)) for lon, lat in points]
