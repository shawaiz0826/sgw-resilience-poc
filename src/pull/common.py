"""Shared paths, study area and HTTP helpers for the pull scripts."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"          # created by the pull scripts when they write; the demo never needs it
PROCESSED = ROOT / "data" / "processed"

# Lee (12071) + Charlotte (12015), Florida. PLAN §4.
STUDY_FIPS = ["12071", "12015"]
STUDY_COUNTIES = {"12071": "Lee", "12015": "Charlotte"}
BBOX = (-82.45, 26.30, -81.45, 27.10)  # lon_min, lat_min, lon_max, lat_max
FL_STATEFP = "12"

# Storms in the advisory store. Landfall times from NHC Tropical Cyclone Reports (UTC).
STORMS = {
    "ian": {
        "name": "Ian", "year": 2022, "atcf": "AL092022",
        "landfall": "2022-09-28T19:05:00Z",  # Cayo Costa, Lee County
        "wsp_start": "2022-09-23T00:00:00Z", "wsp_end": "2022-09-29T12:00:00Z",
        "outage_window": ("2022-09-26", "2022-10-10"),
        "role": "train",
    },
    "idalia": {
        "name": "Idalia", "year": 2023, "atcf": "AL102023",
        "landfall": "2023-08-30T11:45:00Z",  # Keaton Beach, Taylor County
        "wsp_start": "2023-08-27T00:00:00Z", "wsp_end": "2023-08-31T00:00:00Z",
        "outage_window": ("2023-08-28", "2023-09-08"),
        "role": "train",
    },
    "milton": {
        "name": "Milton", "year": 2024, "atcf": "AL142024",
        "landfall": "2024-10-10T00:30:00Z",  # Siesta Key, Sarasota County
        "wsp_start": "2024-10-05T00:00:00Z", "wsp_end": "2024-10-10T12:00:00Z",
        "outage_window": ("2024-10-08", "2024-10-22"),
        "role": "holdout",
    },
}

UA = {"User-Agent": "sgw-resilience-poc/0.1 (research prototype; public data)"}


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA)
    return s


def get(url: str, *, params=None, timeout=60, retries=3, stream=False, sess=None, **kw):
    """GET with retries and backoff. Raises on final failure."""
    sess = sess or session()
    last = None
    for i in range(retries):
        try:
            r = sess.get(url, params=params, timeout=timeout, stream=stream, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last


def download(url: str, dest: Path, *, timeout=120, sess=None) -> Path:
    """Download to dest unless it already exists (raw files are never overwritten)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = get(url, timeout=timeout, stream=True, sess=sess)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_pull_log(dataset_id: str, info: dict) -> None:
    """Record what each pull actually got; inventory.py reads these."""
    RAW.mkdir(parents=True, exist_ok=True)
    log = RAW / "pull_log.json"
    data = json.loads(log.read_text()) if log.exists() else {}
    data[dataset_id] = {**info, "pulled_at": utcnow()}
    log.write_text(json.dumps(data, indent=2))


def arcgis_query(layer_url: str, *, bbox=BBOX, where="1=1", out_fields="*", page=1000, sess=None) -> dict:
    """Page through an ArcGIS REST layer inside bbox; return one GeoJSON FeatureCollection."""
    feats, offset = [], 0
    while True:
        params = {
            "where": where, "outFields": out_fields, "f": "geojson", "outSR": 4326,
            "geometry": ",".join(map(str, bbox)), "geometryType": "esriGeometryEnvelope",
            "inSR": 4326, "spatialRel": "esriSpatialRelIntersects",
            "resultOffset": offset, "resultRecordCount": page,
        }
        d = get(f"{layer_url}/query", params=params, timeout=120, sess=sess).json()
        if "error" in d:
            raise RuntimeError(f"{layer_url}: {d['error']}")
        batch = d.get("features", [])
        feats += batch
        if len(batch) < page and not d.get("properties", {}).get("exceededTransferLimit"):
            break
        offset += len(batch)
    return {"type": "FeatureCollection", "features": feats}


OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def overpass(query: str, rounds: int = 4) -> tuple[dict, str]:
    """Run an Overpass QL query, trying mirrors in turn with backoff (instances rate-limit and overload)."""
    errors = []
    for i in range(rounds):
        for ep in OVERPASS:
            try:
                r = session().post(ep, data={"data": query}, timeout=120)
                r.raise_for_status()
                return r.json(), ep
            except (requests.RequestException, ValueError) as e:
                code = getattr(getattr(e, "response", None), "status_code", "")
                errors.append(f"{ep}: {e.__class__.__name__} {code}".strip())
        time.sleep(15 * (i + 1))
    raise RuntimeError("all Overpass endpoints failed: " + "; ".join(errors[-3:]))
