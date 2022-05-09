# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from xgbse import XGBSEDebiasedBCE, XGBSEStackedWeibull
from xgbse.converters import convert_to_structured

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)

# synthcity relative
from ._base import SurvivalAnalysisPlugin


class XGBSurvivalAnalysis(SurvivalAnalysisPlugin):
    booster = ["gbtree", "gblinear", "dart"]

    def __init__(
        self,
        n_estimators: int = 100,
        colsample_bynode: float = 0.5,
        max_depth: int = 8,
        subsample: float = 0.5,
        learning_rate: float = 5e-2,
        min_child_weight: int = 50,
        tree_method: str = "hist",
        booster: int = 2,
        random_state: int = 0,
        objective: str = "aft",  # "aft", "cox"
        strategy: str = "weibull",  # "weibull", "debiased_bce"
        time_points: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        surv_params = {}
        if objective == "aft":
            surv_params = {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.0,
            }
        else:
            surv_params = {
                "objective": "survival:cox",
                "eval_metric": "cox-nloglik",
            }
        xgboost_params = {
            # survival
            **surv_params,
            **kwargs,
            # basic xgboost
            "n_estimators": n_estimators,
            "colsample_bynode": colsample_bynode,
            "max_depth": max_depth,
            "subsample": subsample,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "verbosity": 0,
            "tree_method": tree_method,
            "booster": XGBSurvivalAnalysis.booster[booster],
            "random_state": random_state,
            "n_jobs": 4,
        }
        lr_params = {
            "C": 1e-3,
            "max_iter": 10000,
        }

        if strategy == "debiased_bce":
            base_model = XGBSEDebiasedBCE(xgboost_params, lr_params)
        elif strategy == "weibull":
            base_model = XGBSEStackedWeibull(xgboost_params)
        else:
            raise ValueError(f"unknown strategy {strategy}")

        self.model = base_model
        self.time_points = time_points

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self, X: pd.DataFrame, T: pd.Series, Y: pd.Series
    ) -> "SurvivalAnalysisPlugin":
        "Training logic"

        y = convert_to_structured(T, Y)

        censored_times = T[Y == 0]
        obs_times = T[Y == 1]

        lower_bound = max(censored_times.min(), obs_times.min()) + 1
        if pd.isna(lower_bound):
            lower_bound = T.min()
        upper_bound = min(censored_times.max(), obs_times.max()) - 1
        if pd.isna(upper_bound):
            upper_bound = T.max()

        time_bins = np.linspace(lower_bound, upper_bound, self.time_points, dtype=int)

        self.model.fit(X, y, time_bins=time_bins)
        return self

    def _find_nearest(self, array: np.ndarray, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame, time_horizons: List) -> pd.DataFrame:
        "Predict risk"
        chunks = int(len(X) / 1024) + 1

        preds_ = []
        for chunk in np.array_split(X, chunks):
            local_preds_ = np.zeros([len(chunk), len(time_horizons)])
            surv = self.model.predict(chunk)
            surv = surv.loc[:, ~surv.columns.duplicated()]
            time_bins = surv.columns
            for t, eval_time in enumerate(time_horizons):
                nearest = self._find_nearest(time_bins, eval_time)
                local_preds_[:, t] = np.asarray(1 - surv[nearest])
            preds_.append(local_preds_)
        return pd.DataFrame(
            np.concatenate(preds_, axis=0), columns=time_horizons, index=X.index
        )

    @staticmethod
    def name() -> str:
        return "survival_xgboost"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="max_depth", low=2, high=6),
            IntegerDistribution(name="min_child_weight", low=0, high=50),
            CategoricalDistribution(name="objective", choices=["aft", "cox"]),
            CategoricalDistribution(
                name="strategy", choices=["weibull", "debiased_bce"]
            ),
        ]
