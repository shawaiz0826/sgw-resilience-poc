"""D8 — USGS STN high-water marks for Hurricane Ian (event 325), Florida."""
from __future__ import annotations

import json

from .common import RAW, get, write_pull_log

STN = "https://stn.wim.usgs.gov/STNServices"


def ian_event_id() -> int:
    events = get(f"{STN}/Events.json", timeout=120).json()
    ids = [e["event_id"] for e in events if e.get("event_name", "").strip().lower() == "2022 ian"]
    if not ids:
        raise RuntimeError("Ian event not found in STN Events.json")
    return ids[0]


def main() -> None:
    dest = RAW / "usgs" / "ian_hwms.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print("[hwm] exists")
        return
    eid = ian_event_id()
    # FilteredHWMs returns the flat HWM table with lat/lon, elevation and datum.
    hwms = get(f"{STN}/HWMs/FilteredHWMs.json", params={"Event": eid, "States": "FL"}, timeout=300).json()
    dest.write_text(json.dumps(hwms))
    write_pull_log("D8", {"source": f"{STN}/HWMs/FilteredHWMs.json?Event={eid}&States=FL", "event_id": eid, "count": len(hwms)})
    print(f"[hwm] event {eid}: {len(hwms)} Florida high-water marks")


if __name__ == "__main__":
    main()
