"""C1 county outage model (PRD C1): P10/P50/P90 of the fraction of customers out per county.

Two implementations behind one interface:
  LGBMQuantile  primary: LightGBM x3 on the pinball (quantile) loss, monotone increasing in p64 and p34
  GLMQuantile   baseline: fractional-logit GLM (statsmodels), monotone by construction (a p64/p34 term
                with a negative coefficient is dropped and the model refitted); P10/P50/P90 from
                empirical residual quantiles on the logit scale
Both save to models/c1/<kind>/ as small text/JSON files; the model version is content-addressed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit

from src.config import ROOT
from src.models.features import FEATURES, MONOTONE

MODEL_DIR = ROOT / "models" / "c1"
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
EPS = 1e-3

# LightGBM's built-in "quantile" objective rejects monotone_constraints (its leaf-value renewal step would
# break them). We pass the same pinball loss as a custom objective, which LightGBM does constrain.
LGBM_PARAMS = {
    "num_leaves": 7, "min_data_in_leaf": 20, "learning_rate": 0.05,
    "monotone_constraints": MONOTONE, "monotone_constraints_method": "advanced",
    "feature_fraction": 1.0, "bagging_fraction": 1.0, "lambda_l2": 1.0,
    "deterministic": True, "force_row_wise": True, "num_threads": 1, "seed": 0, "verbose": -1,
}
LGBM_ROUNDS = 200


def _version(kind: str, folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(folder.glob("*")):
        if f.name != "card.json":
            h.update(f.read_bytes())
    return f"{kind}-v1-{h.hexdigest()[:8]}"


def _non_crossing(p: pd.DataFrame) -> pd.DataFrame:
    """Sort each row's quantiles so P10 <= P50 <= P90, and bound them to [0, 1]."""
    v = np.sort(np.clip(p[["p10", "p50", "p90"]].to_numpy(), 0.0, 1.0), axis=1)
    return pd.DataFrame(v, columns=["p10", "p50", "p90"], index=p.index)


class LGBMQuantile:
    kind = "lgbm"

    def __init__(self, boosters: dict | None = None, base: dict | None = None):
        self.boosters = boosters or {}
        self.base = base or {}  # starting score per quantile (custom objectives do not boost from average)

    @staticmethod
    def _pinball(alpha: float):
        def objective(preds, data):
            y = data.get_label()
            grad = np.where(y > preds, -alpha, 1.0 - alpha)
            return grad, np.ones_like(preds)
        return objective

    def fit(self, df: pd.DataFrame) -> "LGBMQuantile":
        y = df["frac_out"].astype(float).to_numpy()
        for name, a in QUANTILES.items():
            self.base[name] = float(np.quantile(y, a))
            ds = lgb.Dataset(df[FEATURES].astype(float), label=y, init_score=np.full(len(y), self.base[name]),
                             free_raw_data=False)
            self.boosters[name] = lgb.train({**LGBM_PARAMS, "objective": self._pinball(a)}, ds,
                                            num_boost_round=LGBM_ROUNDS)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].astype(float)
        raw = pd.DataFrame({n: b.predict(X) + self.base[n] for n, b in self.boosters.items()}, index=df.index)
        return _non_crossing(raw)

    def save(self) -> str:
        folder = MODEL_DIR / self.kind
        folder.mkdir(parents=True, exist_ok=True)
        for n, b in self.boosters.items():
            b.save_model(str(folder / f"{n}.txt"))
        (folder / "base.json").write_text(json.dumps(self.base, indent=2, sort_keys=True))
        return _version(self.kind, folder)

    @classmethod
    def load(cls) -> "LGBMQuantile":
        folder = MODEL_DIR / cls.kind
        boosters = {n: lgb.Booster(model_file=str(folder / f"{n}.txt")) for n in QUANTILES}
        return cls(boosters, json.loads((folder / "base.json").read_text()))


class GLMQuantile:
    kind = "glm"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @staticmethod
    def _design(df: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X["const"] = 1.0
        for t in terms:
            X[t] = np.log(df["customers"].astype(float)) if t == "log_customers" else df[t].astype(float)
        return X

    def fit(self, df: pd.DataFrame) -> "GLMQuantile":
        terms = ["p64", "p34", "log_customers", "coastal"]
        dropped = []
        y = df["frac_out"].astype(float).clip(0, 1)
        while True:
            X = self._design(df, terms)
            res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            neg = [t for t in ("p64", "p34") if t in terms and res.params[t] < 0]
            if not neg:
                break
            worst = min(neg, key=lambda t: res.params[t])  # monotone constraint: drop, refit
            terms.remove(worst)
            dropped.append(worst)
        eta = X.to_numpy() @ res.params.to_numpy()
        resid = logit(y.clip(EPS, 1 - EPS).to_numpy()) - eta
        self.params = {
            "terms": terms, "dropped_for_monotonicity": dropped,
            "coef": {k: float(v) for k, v in res.params.items()},
            "resid_quantiles": {n: float(np.quantile(resid, a)) for n, a in QUANTILES.items()},
        }
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._design(df, self.params["terms"])
        b = np.array([self.params["coef"][c] for c in X.columns])
        eta = X.to_numpy() @ b
        raw = pd.DataFrame({n: expit(eta + q) for n, q in self.params["resid_quantiles"].items()}, index=df.index)
        return _non_crossing(raw)

    def save(self) -> str:
        folder = MODEL_DIR / self.kind
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "params.json").write_text(json.dumps(self.params, indent=2, sort_keys=True))
        return _version(self.kind, folder)

    @classmethod
    def load(cls) -> "GLMQuantile":
        return cls(json.loads((MODEL_DIR / cls.kind / "params.json").read_text()))


MODELS = {"lgbm": LGBMQuantile, "glm": GLMQuantile}


def load_model(kind: str):
    return MODELS[kind].load()


def model_version(kind: str) -> str:
    return _version(kind, MODEL_DIR / kind)


def active_model_kind() -> str:
    """The version FR30 put in service: written by the backtest selection rule."""
    sel = ROOT / "models" / "c1" / "selection.json"
    return json.loads(sel.read_text())["active"] if sel.exists() else "lgbm"
