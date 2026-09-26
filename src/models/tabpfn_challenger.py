"""Optional C1 challenger: TabPFN v2 regression (Built with PriorLabs-TabPFN).

In-context learning, no training: the same 1,742 Ian + Idalia rows are the context, Milton is predicted.
It reports full-distribution quantiles (P10/P50/P90), but it has no monotone constraints, so it can never be
the active C1 model (PRD C1); it is a benchmark column in the backtest report only.

Weights: Hugging Face Prior-Labs/TabPFN-v2-reg, file tabpfn-v2-regressor.ckpt, pinned to a repo revision
and checked by sha256. The package default ("auto") would pick a newer TabPFN; we pin v2 as PLAN §3 specifies.
License: Prior Labs License 1.1 (Apache 2.0 + attribution), copy in docs/licenses/TabPFN-LICENSE.txt.
Needs requirements-optional.txt (torch). Not required by make demo.
"""
from __future__ import annotations

import hashlib
import json
import os

# torch must load its OpenMP runtime before LightGBM's (imported via backtest -> c1): the reverse order
# segfaults on macOS (two libomp copies in one process).
import torch  # noqa: F401  isort: skip
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from src.config import ROOT, load_config
from src.models.backtest import county_metrics
from src.models.features import FEATURES, advisory_at_lead, county_frame, train_storms, training_rows
from src.pull.common import PROCESSED, RAW

REPO = "Prior-Labs/TabPFN-v2-reg"
REVISION = "4972a65a1b30806315c6f92499959ffbfc69a673"
FILENAME = "tabpfn-v2-regressor.ckpt"


def checkpoint() -> tuple[str, str]:
    path = hf_hub_download(REPO, FILENAME, revision=REVISION, cache_dir=str(RAW / "hf_cache"))
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    lic = ROOT / "docs" / "licenses" / "TabPFN-LICENSE.txt"
    if not lic.exists():
        lic.parent.mkdir(parents=True, exist_ok=True)
        lic.write_text(open(hf_hub_download(REPO, "LICENSE.txt", revision=REVISION, cache_dir=str(RAW / "hf_cache"))).read())
    return path, sha


def predict_quantiles(model, df: pd.DataFrame) -> pd.DataFrame:
    q = model.predict(df[FEATURES].astype(float), output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
    v = np.sort(np.clip(np.column_stack(q), 0.0, 1.0), axis=1)
    return pd.DataFrame(v, columns=["p10", "p50", "p90"], index=df.index)


def main() -> None:
    # 1,742 context rows is inside TabPFN v2's pretraining limits; the package only guards CPU speed above 1,000.
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    from tabpfn import TabPFNRegressor

    cfg = load_config()
    path, sha = checkpoint()
    tr = training_rows(train_storms())
    model = TabPFNRegressor(model_path=path, device="cpu", random_state=0, n_estimators=8)
    model.fit(tr[FEATURES].astype(float), tr["frac_out"].astype(float))
    frame = county_frame(["milton"])
    t72 = advisory_at_lead("milton", cfg["decisionA_lead_hours"])
    at = frame[frame["advisory_time"] == t72].reset_index(drop=True)
    p = predict_quantiles(model, at)
    df = pd.concat([at, p], axis=1)
    res = {"model": f"tabpfn-v2-reg@{REVISION[:8]}", "checkpoint": FILENAME, "sha256": sha, "context_rows": len(tr),
           "advisory_time": str(t72), "attribution": "Built with PriorLabs-TabPFN", **county_metrics(df, int(cfg["top_n"]))}
    curve = []
    for t, g in frame[(frame["hours_to_landfall"] > 0) & (frame["hours_to_landfall"] <= 120)].groupby("advisory_time"):
        g = g.reset_index(drop=True)
        m = county_metrics(pd.concat([g, predict_quantiles(model, g)], axis=1), int(cfg["top_n"]))
        curve.append({"model": "tabpfn", "advisory_time": t, "hours_to_landfall": float(g["hours_to_landfall"].iloc[0]),
                      **{k: m[k] for k in ("mae_customers", "spearman", "top_n_match", "p90_coverage")}})
    (PROCESSED / "backtest_tabpfn.json").write_text(json.dumps(res, indent=2, default=str))
    df["pred_out_p50"] = df["p50"] * df["customers"]
    df[["fips", "p10", "p50", "p90", "pred_out_p50"]].to_parquet(PROCESSED / "backtest_tabpfn_counties.parquet", index=False)
    pd.DataFrame(curve).to_parquet(PROCESSED / "backtest_tabpfn_leadtime.parquet", index=False)
    print(json.dumps({k: res[k] for k in ("model", "sha256", "mae_customers", "spearman", "top_n_match", "p90_coverage")}, indent=2))


if __name__ == "__main__":
    main()
