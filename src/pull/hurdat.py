"""D1 — HURDAT2 (Atlantic). Used to confirm landfall times for the three storms, nothing else."""
from __future__ import annotations

import re

import requests

from .common import RAW, STORMS, download, get, write_pull_log

INDEX = "https://www.nhc.noaa.gov/data/"


def landfalls(path) -> dict[str, list[str]]:
    """Return {ATCF id: ['YYYY-MM-DDTHH:MMZ', ...]} for 'L' (landfall) records of our storms."""
    want = {s["atcf"] for s in STORMS.values()}
    out, cur = {}, None
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if re.match(r"^AL\d{6}$", parts[0]):
            cur = parts[0] if parts[0] in want else None
            continue
        if cur and len(parts) > 3 and parts[2] == "L":
            d, hm = parts[0], parts[1]
            out.setdefault(cur, []).append(f"{d[:4]}-{d[4:6]}-{d[6:]}T{hm[:2]}:{hm[2:]}Z")
    return out


def main() -> None:
    try:
        html = get(INDEX, timeout=120).text
    except requests.RequestException as e:
        cached = sorted((RAW / "hurdat").glob("hurdat2-*.txt"))
        if not cached:
            raise
        print(f"[hurdat] NHC unreachable ({e.__class__.__name__}); landfalls from cached {cached[-1].name}: {landfalls(cached[-1])}")
        return
    links = re.findall(r'href="([^"]*hurdat2-1851-\d{4}-[^"]*\.txt)"', html)
    if not links:
        raise RuntimeError("HURDAT2 Atlantic link not found on NHC data page")
    rel = sorted(links)[-1]
    url = rel if rel.startswith("http") else "https://www.nhc.noaa.gov" + (rel if rel.startswith("/") else "/data/" + rel)
    dest = download(url, RAW / "hurdat" / url.split("/")[-1])
    lf = landfalls(dest)
    write_pull_log("D1", {"source": url, "landfalls": lf})
    for k, v in lf.items():
        print(f"[hurdat] {k} landfalls: {v}")


if __name__ == "__main__":
    main()
