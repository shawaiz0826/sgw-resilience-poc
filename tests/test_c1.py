"""C1 guarantees: monotone in the wind inputs (PRD C1), non-crossing quantiles, versions stable."""
import numpy as np
import pandas as pd
import pytest

from src.models.c1 import load_model, model_version
from src.models.features import training_rows


@pytest.mark.parametrize("kind", ["glm", "lgbm"])
def test_monotone_and_non_crossing(kind):
    m = load_model(kind)
    rows = training_rows(["ian", "idalia"]).sample(40, random_state=1)
    grid = np.linspace(0, 1, 11)
    for feat in ("p64", "p34"):
        for _, r in rows.iterrows():
            X = pd.DataFrame([r] * len(grid))
            X[feat] = grid
            p = m.predict(X)
            assert (np.diff(p.to_numpy(), axis=0) >= -1e-12).all()
            assert (p["p10"] <= p["p50"]).all() and (p["p50"] <= p["p90"]).all()


def test_versions_are_content_hashes():
    assert model_version("glm").startswith("glm-v1-") and model_version("lgbm").startswith("lgbm-v1-")
