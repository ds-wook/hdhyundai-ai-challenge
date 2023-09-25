from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from omegaconf import DictConfig

from modeling.base import BaseModel


class XGBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> xgb.Booster:
        dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, y_valid, enable_categorical=True)

        model = xgb.train(
            dict(self.config.models.params),
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dvalid, "eval")],
            num_boost_round=self.config.models.num_boost_round,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            verbose_eval=self.config.models.verbose_eval,
            feval=self._evaluation,
        )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> CatBoostRegressor:
        train_set = Pool(X_train, y_train, cat_features=self.config.features.categorical_features)
        valid_set = Pool(X_valid, y_valid, cat_features=self.config.features.categorical_features)

        model = CatBoostRegressor(
            cat_features=self.config.features.categorical_features,
            random_state=self.config.models.seed,
            **self.config.models.params,
        )

        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.config.models.verbose_eval,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
        )

        return model


class LightGBMTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> lgb.Booster:
        train_set = lgb.Dataset(X_train, y_train, categorical_feature=self.config.features.categorical_features)
        valid_set = lgb.Dataset(X_valid, y_valid, categorical_feature=self.config.features.categorical_features)

        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(self.config.models.params),
            num_boost_round=self.config.models.num_boost_round,
            feval=self._evaluation,
            callbacks=[
                lgb.log_evaluation(self.config.models.verbose_eval),
                lgb.early_stopping(self.config.models.early_stopping_rounds),
            ],
        )

        return model
