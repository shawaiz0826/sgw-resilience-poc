# Limitations

This prototype shows that the PRD's decision chain works end to end on public data for Lee and Charlotte
Counties, Florida. It is evidence for the design, not a Phase 0 pilot, and nothing in it is SGW's.
Every substitution below is also labelled on screen where it appears.

## 1. Prototype substitutions (PRD Appendix A)

| Production (PRD) | This prototype | On screen |
|---|---|---|
| Phase 0 registry, one ID per asset (FR1) | OpenStreetMap substations, HIFLD transmission lines, EPA FRS + OSM wastewater plants, OSM pumping stations, HIFLD hospitals, each under its own source ID; no entity resolution | Source and vintage per asset; sidebar vintage line |
| Maintenance attributes in C3's weight (FR2) | Public attributes only: FEMA zone, ground elevation, voltage class | "exposure ranking, county-validated" |
| Switchgear height from the registry (FR18) | ESTIMATED = USGS 3DEP ground elevation + 1.0 m placeholder offset, for every substation | ESTIMATED badge; threshold raised by the margin |
| Recorded feeding substation (FR3, R10) | Nearest OSM substation by straight line; none beyond 10 km | PLACEHOLDER FEED; "no plausible feed" |
| Customers served per substation (FR4) | None; customers per county from EAGLE-I | Decision B ranks on critical loads only |
| Critical loads and backup status (A18) | HIFLD hospitals and OSM pumping stations on the placeholder feed; backup unknown, treated as none | "critical load, backup unknown" |
| SGW's own crew-hours (FR11) | Placeholder default (20,000), editable by P1 | PLACEHOLDER badge |
| Flood sensor values (FR8, FR21) | None; the field is blank and marked blank | Sensor: BLANK |
| Staging site list (A19) | Four placeholder sites (Fort Myers, Punta Gorda, Sarasota, Sebring) | PLACEHOLDER badge |
| C6 in a private deployment (A7) | Template by default (no LLM); optional hosted API, or Ollama as the self-hosted switch | Mode shown on the briefing tab |
| SSO, MFA, RBAC (A20) | None: roles are labels typed into the sign-off forms | Governance tab note |
| P-Surge for every storm (FR6) | Real P-Surge for Ian only; from T-48h Milton gets STATIC tiers routed to P2's judgment | "STATIC (P-Surge grid not loaded)", JUDGMENT |

## 2. Out of scope (PLAN §1)

Phase 0 entity resolution, sentinel and validity rules (R1, R2); maintenance attributes and their staleness
check (R8, staleness_months is in config but unused); customers per substation; flood sensors and the SCADA
historian; the crew roster; the NLCD tree-canopy covariate; hospitals' backup status; access routing; anomaly
detection; computer-vision damage assessment; every hazard but hurricane; operating districts as zones; the
Phase 2 customer-weighted county-to-asset step (FR38). Non-functional items are not built: availability targets
and degraded-mode operations beyond the JSON export (FR27), encryption with SGW keys, identity provider
integration, the 6-hourly scheduler (the prototype runs a decision when an advisory is selected), and automated
post-storm re-scoring (FR31: the Milton backtest is the same computation, run once by hand).

## 3. Model limits (C1)

- **County-validated only.** C1 is selected on leave-one-storm-out between the two training storms and validated
  at county level on one held-out storm (Milton), reported once. The per-asset exposure ranking (C3) has no
  observed outcome behind it (PRD U1).
- **Two training storms is the model's main limit.** Ian and Idalia, 1,742 county-advisory rows. Between them,
  skill is low (Spearman 0.09–0.13 LightGBM, 0.29–0.64 GLM). Milton's result is stronger because its track
  resembled Ian's; a storm on a different coast would look like the leave-one-storm-out numbers. Ian is
  in-sample: Decision A on Ian is not a test.
- **The active model is the GLM baseline**, not LightGBM, by the leave-one-storm-out rule (MODELS.md). Its
  intervals are wide (Lee at Milton T-72h: P10 5.5k, P50 37k, P90 393k; observed 273k) and its P50 sits below the
  observed values for the hardest-hit counties. P1 acts on P90 through C2.
- **LightGBM deviation.** LightGBM's built-in quantile objective rejects monotone constraints; the prototype uses
  the pinball loss as a custom objective so the constraint holds (MODELS.md).
- **Denominator.** Customers per county are EAGLE-I's modeled counts (MCC). Where observed outages exceed MCC,
  the fraction is capped at 1 (2 counties for Ian, 4 for Idalia, 2 for Milton).
- **EAGLE-I coverage.** EAGLE-I scrapes utility outage maps; its Lee County peak for Ian is about 73% of MCC
  customers, below utility-reported figures. Observed fractions are a floor. Units are customers (meters), not
  people (R7).
- **Wind inputs.** NHC 5 km products are contour bands; probabilities are band midpoints. Area-weighted county
  means smooth over within-county gradients.

## 4. Decision B limits (C5)

- **Switchgear height is ESTIMATED everywhere** (ground + 1.0 m), so the raised threshold (0.65) is always in
  force. The 1.0 m offset maps to P-Surge's 3 ft threshold (0.91 m, the nearest whole foot at or below it).
- **P-Surge is above ground at the SLOSH grid**, compared here with a 3DEP ground elevation at the site; the two
  ground surfaces differ locally.
- **Ian only.** P-Surge thresholds 1-4 ft were retrieved for all 20 Ian cycles; 5-6 ft were not (local DNS
  failures), so offsets above 1.22 m cannot use P-Surge in this build.
- **STATIC is a tier, not a probability.** Without a P-Surge snapshot (Milton in this prototype), each
  screening-set substation gets a STATIC tier from its FEMA zone and BFE: HIGH (VE/V, or AE with the NAVD88 BFE
  above the ESTIMATED switchgear), MEDIUM (AE, A, AO, AH), LOW otherwise. STATIC rows are never compared with the
  threshold; they route to P2's judgment (JUDGMENT), so FR22 reports how many route there rather than a precision
  or recall. The old static probabilities stay in the record as tier anchors.
- **Validation is thin.** USGS high-water marks are sparse: the median substation is about 4 km from the nearest
  one. The PLAN rule (any HWM within 500 m above switchgear) covers 2 substations; 1 km and 3 km
  inverse-distance-weighted water surfaces cover 4 and 12 in the screening set. P-Surge recall is 1.0 at every
  radius; precision falls to 0.2 at 3 km.
- **The FEMA screening set misses surge outside mapped zones.** Ian's P-Surge gave >= 0.8 probability of more
  than 3 ft above ground at several Zone X substations (for example Piney Road, West Cape, Calusa). A FEMA-zone
  screening set (PRD C5) does not look at them. Phase 0 should screen on P-Surge coverage as well as FEMA zone.

## 5. Decision A limits (C2)

- `restoration_days` (7) is added to the PRD rule so the request is in crews rather than crew-days; it, the crew
  size (4) and SGW's own crew-hours (20,000) are placeholders.
- A staging site's wind check uses the cumulative 0-120 h 64 kt probability of the 0.1-degree cell containing
  it, not a timed wind window. During Milton no mapped site exceeded 0.5, so no site was displaced; during Ian
  both were, and at T-13h every listed site was above 0.5 and the plan hands the choice to P1.

## 6. Asset data limits (C3, C4)

- OSM substations mix transmission and distribution and carry OSM's voltage tags; 64 in the study area, 1
  without voltage. HIFLD transmission lines are 2015-2017 vintage from a deprecated repository (R6).
- OSM maps only 11 pumping (lift) stations in Lee and Charlotte; the P4 list is illustrative, not complete.
- HIFLD hospitals were last validated in 2013-2014. EPA FRS and OSM wastewater plants are merged, dropping OSM
  plants within 500 m of an EPA plant.
- FEMA zones come from Esri's hosted copy of NFHL (reduced set), which omits unshaded Zone X; assets outside it
  are labelled "X, inferred". BFE is used only when its datum is NAVD88 (R3).

## 7. C6 limits

- Template mode (the default) answers questions by keyword retrieval over the record; it does not understand
  paraphrases. The LLM modes are implemented but were not exercised with live keys in this build.
- The FR35 number check flags any number in an LLM draft that does not appear in the record context. It is a
  guard, not a proof: a number that happens to appear elsewhere in the record passes.

## 8. Fallbacks taken (PLAN §6)

| When | Trigger | Action | Label |
|---|---|---|---|
| 2026-09-26 | NASA NCCS HIFLD mirror did not respond; HIFLD Electric Substations no longer published | Substations from OpenStreetMap; lines still HIFLD | Vintage "OSM <date>"; PLACEHOLDER FEED to nearest OSM substation |
| 2026-09-26 | hazards.fema.gov reset every connection | FEMA NFHL from Esri Living Atlas | Zone X "inferred" outside the reduced set |
| 2026-09-26 | No `tenthDeg` wind grids for 2022-2024 | NHC `5km` contour-band product | Resolution "5km" in sidebar and records |
| 2026-09-26 | ORNL EAGLE-I pages point to Globus (login) | Same release from figshare (CC BY 4.0); MCC denominator | DATA.md |
| 2026-09-26 | Local DNS intermittently failed for www.nhc.noaa.gov | Ian P-Surge 5-6 ft thresholds not retrieved; NHC pulls fall back to cached raw files | Offset <= 1.22 m while PSURGE is the source |
| 2026-09-26 | P-Surge found for Ian | Real P-Surge used (optional item 3 of PLAN §1) instead of STATIC | PSURGE badge |

## 9. Data surprises (from data/processed/INVENTORY.md)

1. NHC's archive has no `tenthDeg` wind-probability files for 2022-2024; the high-resolution product is `5km`
   (0.05 deg contour bands). All 67 advisories use 5km; probabilities are band midpoints.
2. HIFLD Electric Substations is no longer published anywhere reachable, and the NASA NCCS HIFLD mirror did not
   respond. Substations come from OpenStreetMap (pre-authorised fallback); feeds are nearest OSM substation.
3. hazards.fema.gov reset every connection. Flood zones come from Esri's Living Atlas copy of NFHL (reduced
   set), which omits unshaded Zone X; assets outside it are labelled 'X, inferred'.
4. EAGLE-I came from the figshare release (no login), not the ORNL Globus pages. The 2024 `total_customers`
   field overstates several counties (Miami-Dade 1.9x the modeled count), so MCC is the denominator.
5. Real P-Surge exists for Ian, so Decision B on Ian uses PSURGE probabilities, not the STATIC fallback. STATIC
   remains the path for any advisory without a P-Surge snapshot (e.g. Milton in this prototype).
6. OSM maps only a handful of pumping (lift) stations in Lee and Charlotte; the P4 list is illustrative, not
   complete.
7. Ian's P-Surge gave >= 0.8 probability of > 3 ft above ground at several Zone X substations. A FEMA-zone
   screening set (PRD C5) would not look at them.
8. EAGLE-I shows Lee County at about 73% of MCC customers out at Ian's peak, below utility-reported figures;
   EAGLE-I coverage of a county's utilities is incomplete, so fractions are a floor.

## 10. Running it

- The map basemap (Carto) needs internet; the data and the decisions do not.
- On macOS, the LightGBM fallback model needs a system OpenMP (`brew install libomp`). The app starts without it
  (the GLM is active); withdrawing the GLM on such a machine shows an error instead. Docker has no such issue.
- TabPFN (the benchmark column) needs `requirements-optional.txt` (torch); its results are committed.
