# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression checks for horizon-independent direct reduction features."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from sktime.forecasting.compose import DirectReductionForecaster
from sktime.transformations.impute import Imputer
from sktime.transformations.lag import ReducerTransform


@pytest.mark.parametrize("fh", [[-8, -3, -1], [1, 3, 5], [-3, 0, 2]])
@pytest.mark.parametrize("panel", [False, True])
@pytest.mark.parametrize("X_treatment", ["concurrent", "shifted"])
@pytest.mark.parametrize("windows_identical", [False, True])
def test_direct_reduction_reuses_deterministic_features(
    monkeypatch, fh, panel, X_treatment, windows_identical
):
    """Reuse feature work, not horizon regressors; match independent transforms."""
    y = pd.DataFrame({"y": np.arange(30.0) + np.sin(np.arange(30))})
    if panel:
        y = pd.concat({"a": y, "b": y * 2 + 3}, names=["instance"])
    params = dict(
        estimator=Ridge(alpha=0.1),
        window_length=4,
        pooling="global",
        X_treatment=X_treatment,
        windows_identical=windows_identical,
    )
    # A supplied transformer retains the independent per-horizon path, while
    # using exactly the same deterministic imputation as the default.
    reference = DirectReductionForecaster(
        **params, impute_method=Imputer(method="bfill")
    ).fit(y, fh=fh)
    expected = reference.predict()

    calls = {"fit": 0, "transform": 0}
    original_fit = ReducerTransform._fit
    original_transform = ReducerTransform._transform

    def counted_fit(self, X, y=None):
        calls["fit"] += 1
        return original_fit(self, X=X, y=y)

    def counted_transform(self, X, y=None):
        calls["transform"] += 1
        return original_transform(self, X=X, y=y)

    monkeypatch.setattr(ReducerTransform, "_fit", counted_fit)
    monkeypatch.setattr(ReducerTransform, "_transform", counted_transform)
    forecaster = DirectReductionForecaster(**params).fit(y, fh=fh)
    actual = forecaster.predict()
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
    n_instances = 2 if panel else 1
    assert calls == {"fit": n_instances, "transform": 2 * n_instances}
    if not (X_treatment == "shifted" and windows_identical):
        assert len(forecaster.estimators_) == len(fh)
        assert len({id(x) for x in forecaster.lagger_y_to_X_.values()}) == len(fh)


@pytest.mark.parametrize("panel", [False, True])
def test_direct_reduction_keeps_prediction_feature_index(panel):
    """Pandas-aware regressors receive horizon-shifted feature row labels."""
    from sklearn.base import BaseEstimator, RegressorMixin

    class IndexRegressor(RegressorMixin, BaseEstimator):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X.index.get_level_values(-1).to_numpy(dtype=float).reshape(-1, 1)

    y = pd.DataFrame({"y": np.arange(30.0)})
    if panel:
        y = pd.concat({"a": y, "b": y + 3}, names=["instance"])
    params = dict(estimator=IndexRegressor(), window_length=4, pooling="global")
    actual = DirectReductionForecaster(**params).fit_predict(y, fh=[-3, 0, 2])
    expected = DirectReductionForecaster(
        **params, impute_method=Imputer(method="bfill")
    ).fit_predict(y, fh=[-3, 0, 2])
    pd.testing.assert_frame_equal(actual, expected)
