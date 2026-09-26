# Models

C1 is the only fitted model in the prototype (PRD C1). C2 to C5 are rules and lookups; C6 is a template by
default and an LLM only when one is configured. Full numbers: [backtest_report.md](backtest_report.md).

## C1 — county outage model

| Item | Value |
|---|---|
| Output | Fraction of customers (meters) out per county at P10, P50, P90; times customers gives customers out |
| Features (max 4) | `p64`, `p34`: area-weighted mean of NHC's cumulative 0-120 h wind speed probability over the county; `customers`: EAGLE-I modeled customers per county (MCC); `coastal`: county touches the Gulf or Atlantic (hand-coded, 35 of 67) |
| Target | Storm peak customers out per county (15-min EAGLE-I, landfall -24 h to +96 h, net of the first-day median) / customers, capped at 1 |
| Training rows | One per (county, storm, advisory) for Ian + Idalia, advisories issued between T-78h and landfall: 1,742 rows, 67 counties, 26 advisories |
| Held out | Milton, entirely (no training, no tuning) |
| Pairing rule (R5) | Every advisory is paired with its storm's peak; nothing is interpolated across the 15-min / 6-h cadences |

### Primary: LightGBM, pinball loss, monotone

Three boosters (alpha = 0.1, 0.5, 0.9), `num_leaves=7`, `min_data_in_leaf=20`, 200 rounds, learning rate 0.05,
L2 1.0, deterministic single-threaded, seed 0. Monotone increasing in `p64` and `p34`
(`monotone_constraints=[1,1,0,0]`, method `advanced`).

**Deviation from PLAN §5B, and why.** LightGBM's built-in `objective="quantile"` refuses `monotone_constraints`
(`Cannot use monotone_constraints in quantile objective`): its leaf-value renewal step would break them. We pass
the same pinball loss as a custom objective (gradient `-alpha` / `1-alpha`, unit hessian, starting score at the
training quantile), which LightGBM does constrain. Monotonicity is checked by sweeping `p64` and `p34` from 0 to 1
on 300 training rows for every quantile: no prediction ever decreases. Quantiles are sorted per row so
P10 <= P50 <= P90.

### Baseline: GLM, fractional logit

statsmodels `GLM(Binomial)` on the fraction with terms `p64`, `p34`, `log(customers)`, `coastal`. Monotone by
construction: a `p64`/`p34` term with a negative coefficient is dropped and the model refitted (none was dropped).
P10/P50/P90 = logistic(linear predictor + empirical residual quantile on the logit scale).

### Selection rule (FR30), fixed before the Milton run

The GLM ships if it matches or beats LightGBM on at least two of: MAE of customers out (lower), Spearman rank
correlation of counties (higher), top-5 match (higher). Result on Milton T-72h:

| | LightGBM | GLM | TabPFN v2 (benchmark) |
|---|---|---|---|
| MAE, customers out per county | 40,770 | 48,449 | 26,692 |
| Spearman | 0.81 | **0.90** | 0.85 |
| Top-5 match (target 0.70) | 0.60 | **1.00** | 0.60 |
| P90 coverage (target 0.85) | 0.81 | **0.87** | 0.79 |

**The GLM is the active model** (`glm-v1-fa24c9fa`), and it meets both FR17 targets, so the low-confidence banner
is off. LightGBM (`lgbm-v1-23dd02c3`) stays approved as the fallback; were P1 to withdraw the GLM (FR30), the
banner would come on, because LightGBM misses both targets. Leave-one-storm-out between the two training storms
pointed the same way (GLM Spearman 0.29 / 0.64 vs LightGBM 0.09 / 0.13): with two storms, the simpler model
transfers better.

Read the result with its limits: one held-out storm, two training storms, and wide GLM intervals (Lee at Milton
T-72h: P10 5.5k, P50 37k, P90 393k customers out; observed 273k). P1 acts on P90 through C2.

### Challenger: TabPFN v2 (optional, benchmark only) — Built with PriorLabs-TabPFN

- Weights: Hugging Face `Prior-Labs/TabPFN-v2-reg`, file `tabpfn-v2-regressor.ckpt`, revision `4972a65a`,
  sha256 `2ab5a07d5c41...` (checked at download; not committed). The `tabpfn` 9.0 package default (`auto`) would
  load a newer model; v2 is pinned explicitly as PLAN §3 specifies.
- In-context learning on the same 1,742 rows, no fitting; quantiles from its predictive distribution. CPU, about
  2.5 minutes for the whole Milton backtest.
- It has the best MAE but misses both banner targets and has **no monotone constraints**, so it cannot be the active
  C1 model (PRD C1 requires monotonicity in the wind inputs).
- License: Prior Labs License 1.1 (Apache 2.0 + attribution), copy in [licenses/TabPFN-LICENSE.txt](licenses/TabPFN-LICENSE.txt).
- Run: `pip install -r requirements-optional.txt && python -m src.models.tabpfn_challenger && make fit`.
- macOS note: torch must be imported before LightGBM in the same process (two OpenMP runtimes segfault otherwise).

## Models considered

| Model | Role | Status | Why |
|---|---|---|---|
| LightGBM pinball x3, monotone | C1 primary | Fitted, approved fallback | PRD C1 spec; lost the selection rule on Milton |
| Fractional-logit GLM | C1 baseline | Fitted, **active** | Won 2/3 on Milton; meets FR17 targets |
| TabPFN v2 regression | C1 challenger | Benchmark column | No monotone constraints (PRD C1) |
| Random forest on gust, wind duration, customer density (Guikema, Nateghi) | C1 alternative | Not built | Needs gust and duration fields and asset-level history; a Phase 2 candidate once SGW's outage history exists (FR37) |
| Chronos-Bolt, TimesFM, Moirai | Time-series foundation models | Rejected | The problem is cross-sectional (which counties, how many), not extrapolating a series; exogenous covariates are weak or absent |
| Prithvi-EO-2.0-300M-TL-Sen1Floods11 (IBM/NASA) | Flood extent from satellite imagery | Named only, Phase 3 | Post-event; not downloaded |
| xBD / xView2 fine-tuned damage model | Damage assessment on NOAA NGS imagery | Named only, Phase 3 (FR40) | Post-event; not downloaded |
| Any instruct LLM via provider interface | C6 briefing and questions | Optional | Template by default (FR36); outside the decision path |

## Versions and reproducibility

Model versions are content hashes of the saved files (`models/c1/<kind>/`); refitting with the same data gives
the same versions (checked). `models/c1/selection.json` records the active version and the rule's reasoning.
