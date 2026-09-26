# Limitations

Draft — completed in Phase E. Everything here is also visible on screen where it applies.

## Fallbacks taken (PLAN §6), with dates

| When | Trigger | Action | Label on screen / in docs |
|---|---|---|---|
| 2026-09-26 | NASA NCCS HIFLD mirror did not respond; HIFLD Electric Substations no longer published | Substations from OpenStreetMap `power=substation`; transmission lines still HIFLD (ArcGIS Online, deprecated) | Vintage "OSM <date>"; feeds are "nearest OSM substation", PLACEHOLDER FEED |
| 2026-09-26 | hazards.fema.gov reset every connection | FEMA NFHL flood zones from Esri Living Atlas "USA Flood Hazard Reduced Set" (same FEMA polygons) | Zone X outside the reduced set labelled "inferred" |
| 2026-09-26 | NHC archive has no `tenthDeg` wind grids for 2022-2024 | `5km` contour-band product, probability = band midpoint | Resolution "5km" in sidebar and every record |
| 2026-09-26 | ORNL EAGLE-I pages point to Globus (login) | Same EAGLE-I release from figshare (CC BY 4.0); MCC as customers-per-county denominator | DATA.md |
| 2026-09-26 | Local DNS intermittently failed for www.nhc.noaa.gov | Ian P-Surge thresholds 5-6 ft not retrieved (1-4 ft present for all 20 cycles) | Offset limited to <= 1.22 m while PSURGE is the source |

## Data surprises (from data/processed/INVENTORY.md)

See the "Surprises" section of `data/processed/INVENTORY.md`; copied here in Phase E.
