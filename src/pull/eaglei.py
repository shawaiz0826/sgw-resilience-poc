"""D7 — EAGLE-I county outages (ORNL), via the figshare mirror of the Sci Data release.

figshare article 10.6084/m9.figshare.24237376 holds one CSV per year (2014-2025), ~1.2-1.4 GB each.
We download each needed year (resumable), keep only Florida rows inside the storm outage windows,
then delete the full file, so the raw copy on disk is a few MB. Output: data/raw/eaglei/fl_<storm>.csv.gz
and data/raw/eaglei/mcc.csv (modeled customers per county).
"""
from __future__ import annotations

import csv
import gzip
import time
from pathlib import Path

import requests

from .common import RAW, STORMS, get, session, write_pull_log

FIGSHARE_ARTICLE = "https://api.figshare.com/v2/articles/24237376"
YEAR_FILE = {2022: "eaglei_outages_2022.csv", 2023: "eaglei_outages_2023.csv", 2024: "eaglei_outages_2024.csv"}
OUT = RAW / "eaglei"


def _file_urls() -> dict[str, str]:
    meta = get(FIGSHARE_ARTICLE, timeout=60).json()
    return {f["name"]: f["download_url"] for f in meta["files"]}


def _download_resumable(url: str, dest: Path, attempts: int = 20) -> None:
    """Figshare/S3 connections drop on multi-GB files; resume with HTTP Range until complete."""
    sess = session()
    for _ in range(attempts):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with sess.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code == 416:  # already complete
                    return
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored Range; start over
                    dest.unlink()
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                with open(dest, "ab") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if dest.stat().st_size >= total:
                return
        except requests.RequestException as e:
            print(f"[eaglei]   connection dropped at {dest.stat().st_size if dest.exists() else 0:,} bytes ({e.__class__.__name__}); resuming", flush=True)
            time.sleep(3)
    raise RuntimeError(f"could not complete download of {url}")


def pull_year(url: str, storm_id: str) -> dict:
    s = STORMS[storm_id]
    lo, hi = s["outage_window"]
    dest = OUT / f"fl_{storm_id}.csv.gz"
    if dest.exists():
        return {"file": dest.name, "skipped": "exists"}
    big = OUT / f"_full_{s['year']}.csv"
    _download_resumable(url, big)
    kept = 0
    tmp = dest.with_suffix(".part")
    with open(big, encoding="utf-8-sig", newline="") as src, gzip.open(tmp, "wt", newline="") as g:
        reader = csv.reader(src)
        header = next(reader)
        # 2023 names the count column "sum"; normalise to customers_out.
        norm = ["customers_out" if h == "sum" else h for h in header]
        i_state, i_time = norm.index("state"), norm.index("run_start_time")
        w = csv.writer(g)
        w.writerow(norm)
        for row in reader:
            if row[i_state] == "Florida" and lo <= row[i_time][:10] <= hi:
                w.writerow(row)
                kept += 1
    tmp.rename(dest)
    big.unlink()  # keep disk use low; the filtered copy is the raw of record
    return {"file": dest.name, "rows": kept, "columns": norm, "window": [lo, hi]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    urls = _file_urls()
    results = {}
    mcc = OUT / "mcc.csv"
    if not mcc.exists():
        mcc.write_bytes(get(urls["MCC.csv"]).content)
    results["mcc"] = {"file": "mcc.csv"}
    for storm_id, s in STORMS.items():
        name = YEAR_FILE[s["year"]]
        print(f"[eaglei] {storm_id}: {name} (Florida rows in storm window only)", flush=True)
        results[storm_id] = pull_year(urls[name], storm_id)
        print(f"[eaglei] {storm_id}: {results[storm_id]}", flush=True)
    write_pull_log("D7", {"source": "figshare 10.6084/m9.figshare.24237376 (ORNL EAGLE-I)", "results": results})


if __name__ == "__main__":
    main()
