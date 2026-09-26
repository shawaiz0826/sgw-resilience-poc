# Data

Twelve public datasets, pulled by `make data` (network required) into `data/raw/` (not committed) and reduced to
`data/processed/` (committed, 1.5 MB), which is all the demo reads. `src/pull/manifest.yaml` is the
machine-readable version of this table; `data/processed/INVENTORY.md` lists rows, vintages and gaps.

Study area: Lee (FIPS 12071) and Charlotte (12015) Counties, Florida; bounding box lon -82.45 to -81.45, lat
26.30 to 27.10. C1 uses all 67 Florida counties.

| ID | Dataset | Source actually used | Vintage | License | How pulled | In use |
|---|---|---|---|---|---|---|
| D0 | County boundaries | US Census cartographic boundary file `cb_2023_us_county_500k` | 2023 | Public domain | `counties.py`, zip download | 67 FL counties |
| D1 | HURDAT2 best tracks | NOAA NHC `hurdat2-1851-2025` | 1851-2025 | Public domain | `hurdat.py`; landfall times only | 3 storms |
| D2 | Wind speed probabilities (34/64 kt, 0-120 h) | NOAA NHC GIS archive, `YYYYMMDDHH_wsp_120hr5km.zip` | per 6-h advisory | Public domain | `wsp.py`, parse archive page | 67 advisories (Ian 27, Idalia 17, Milton 23), 5 km |
| D3 | P-Surge, Ian (P(surge > N ft above ground), N = 1-4) | NOAA NHC GIS P-Surge archive, `al092022_psurgeN_*.zip` | 2022-09-26 00Z to 09-30 18Z | Public domain | `psurge.py` | 20 advisories x 4 thresholds, sampled at 64 substations |
| D4 | Substations | OpenStreetMap `power=substation` via Overpass (fallback: HIFLD substations no longer published) | OSM 2026-09-26 | ODbL 1.0, (c) OpenStreetMap contributors | `assets.py` | 64 |
| D4 | Transmission lines | HIFLD Electric Power Transmission Lines (ArcGIS Online, deprecated) | validated 2015-2017 | Public domain (US Government) | `assets.py`, ArcGIS REST | 85 segments |
| D5 | Wastewater treatment plants | EPA FRS WWTP (via FEMA Critical Infrastructure service) + OSM `man_made=wastewater_plant` | EPA current; OSM 2026-07-24 | Public domain; ODbL | `assets.py` | 29 |
| D5b | Pumping stations | OSM `man_made=pumping_station` | OSM 2026-09-26 | ODbL | `assets.py` | 11 |
| D6 | Flood hazard zones, BFE | FEMA NFHL via Esri Living Atlas "USA Flood Hazard Reduced Set" (hazards.fema.gov unreachable) | current effective | FEMA data, public domain | `fema.py`, ArcGIS REST; FEMA endpoint tried first | zone and BFE at 198 assets; SFHA polygons for the map |
| D7 | County outages, 15-min | ORNL EAGLE-I via figshare `10.6084/m9.figshare.24237376` (2022, 2023, 2024 + MCC) | 2022-2024 | CC BY 4.0 (Brelsford et al., Scientific Data, 2024) | `eaglei.py`, resumable download, Florida rows in the storm windows kept | 67 counties x 3 storms |
| D8 | High-water marks, Ian | USGS STN event 325 | Sept-Oct 2022 | Public domain | `hwm.py` | 259 in the study area (NAVD88) |
| — | Imagery truth for Decision B (Ian) | Manual inspection of NOAA NGS post-Ian imagery, 29 Sep–3 Oct 2022 (`data/processed/imagery_truth_ian.csv`, created by `src/models/imagery_truth.py`) | 2022 | Public domain imagery | 34 substations (screening set + P-Surge flags outside it); `flooded` = Y / N / UNCLEAR, filled by hand, blank until labelled | 0 labelled |
| D9 | Ground elevation | USGS 3DEP via the EPQS point service | current | Public domain | `elevation.py`, one cached call per asset | 198 of 198 |
| D12 | Hospitals | HIFLD Hospitals (via FEMA Critical Infrastructure service) | validated 2013-2014 | Public domain | `assets.py` | 9 open |

Not loaded: D10 (census blocks, a Phase 2 fallback) and D11 (NLCD tree canopy, the optional C1 covariate; the
coastal flag was chosen instead).

## Processed tables

| File | What |
|---|---|
| `advisories.parquet` | Advisory store index: storm, advisory time, product (wsp34, wsp64, psurge_Nft), resolution, raw file |
| `advisory_county.parquet` | p34, p64 per county per advisory (area-weighted band midpoints) |
| `advisory_cells.parquet` | p34, p64 at 0.1-degree cell centres over SW Florida (C3, staging-site wind check) |
| `psurge_sites.parquet` | P-Surge probability at each substation, per Ian advisory and threshold |
| `outages_county.parquet` | Peak customers out per county per storm, net of baseline; MCC customers; fraction |
| `outage_series_study.parquet` | Hourly customers out for Lee and Charlotte |
| `registry.parquet` | 198 assets: type, source, vintage, location, voltage, FEMA zone, BFE (NAVD88 m), ground elevation, placeholder feed |
| `lines.parquet`, `counties.parquet`, `flood_sfha.parquet`, `hwm_ian.parquet` | Map layers and FR22 truth |
| `backtest*.json/.parquet`, `fr22_ian.parquet` | Backtest outputs read by the app |

## Units and datums

Outage counts are customers (meters), never people (R7). Heights are metres NAVD88; FEMA BFE in feet is converted
only when its datum is NAVD88 (R3). P-Surge heights are above ground. EAGLE-I timestamps are UTC.

## Basemap

Map tiles: CARTO Positron (c) CARTO, (c) OpenStreetMap contributors; loaded by the browser, no token.
