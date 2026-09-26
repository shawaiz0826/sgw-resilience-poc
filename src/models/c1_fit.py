"""Fit C1 on the training storms (Ian + Idalia), Milton held out entirely (FR16).

Also reports leave-one-storm-out skill between the two training storms (group-aware CV by storm) for
information only; hyperparameters are fixed in c1.py and not tuned on it.
"""
from __future__ import annotations

import json

from scipy.stats import spearmanr

from src.models.c1 import MODEL_DIR, GLMQuantile, LGBMQuantile
from src.models.features import FEATURES, HORIZON_H, train_storms, training_rows


def _loso(df) -> dict:
    out = {}
    for held in df["storm_id"].unique():
        tr, te = df[df["storm_id"] != held], df[df["storm_id"] == held]
        for cls in (LGBMQuantile, GLMQuantile):
            p = cls().fit(tr).predict(te)
            rho = spearmanr(p["p50"], te["frac_out"]).statistic
            cov = float((te["frac_out"] <= p["p90"]).mean())
            out[f"{cls.kind}_trained_without_{held}"] = {"spearman_frac": round(float(rho), 3), "p90_coverage": round(cov, 3)}
    return out


def main() -> None:
    storms = train_storms()
    df = training_rows(storms)
    cards = {}
    for cls in (LGBMQuantile, GLMQuantile):
        m = cls().fit(df)
        version = m.save()
        card = {
            "model_version": version, "kind": cls.kind, "features": FEATURES, "target": "peak fraction of customers out",
            "training_storms": storms, "holdout_storms": ["milton"], "training_rows": len(df),
            "training_counties": int(df["fips"].nunique()), "horizon_hours_before_landfall": list(HORIZON_H),
            "training_advisories": int(df.groupby("storm_id")["advisory_time"].nunique().sum()),
        }
        if cls is GLMQuantile:
            card["glm"] = m.params
        (MODEL_DIR / cls.kind / "card.json").write_text(json.dumps(card, indent=2))
        cards[cls.kind] = card
        print(f"[c1_fit] {version}: {len(df)} rows from {storms}")
    cv = _loso(df)
    (MODEL_DIR / "loso_cv.json").write_text(json.dumps(cv, indent=2))
    print(f"[c1_fit] leave-one-storm-out (info only): {cv}")
    print(f"[c1_fit] GLM terms: {cards['glm']['glm']['terms']}, dropped: {cards['glm']['glm']['dropped_for_monotonicity']}")


if __name__ == "__main__":
    main()
