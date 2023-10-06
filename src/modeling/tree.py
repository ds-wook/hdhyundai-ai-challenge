from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb.catboost as wandb_cb
import wandb.lightgbm as wandb_lgb
import wandb.xgboost as wandb_xgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from omegaconf import DictConfig

from modeling.base import BaseModel


class XGBoostTrainer(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> xgb.Booster:
        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        model = xgb.train(
            dict(self.cfg.models.params),
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dvalid, "eval")],
            num_boost_round=self.cfg.models.num_boost_round,
            early_stopping_rounds=self.cfg.models.early_stopping_rounds,
            verbose_eval=self.cfg.models.verbose_eval,
            callbacks=[wandb_xgb.WandbCallback()],
        )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> CatBoostRegressor:
        train_set = Pool(X_train, y_train, cat_features=self.cfg.store.categorical_features)
        valid_set = Pool(X_valid, y_valid, cat_features=self.cfg.store.categorical_features)

        model = CatBoostRegressor(
            cat_features=self.cfg.store.categorical_features,
            random_state=self.cfg.models.seed,
            **self.cfg.models.params,
        )

        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.cfg.models.verbose_eval,
            early_stopping_rounds=self.cfg.models.early_stopping_rounds,
            callbacks=[wandb_cb.WandbCallback()],
        )

        return model


class LightGBMTrainer(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> lgb.Booster:
        train_set = lgb.Dataset(X_train, y_train, categorical_feature=self.cfg.store.categorical_features)
        valid_set = lgb.Dataset(X_valid, y_valid, categorical_feature=self.cfg.store.categorical_features)

        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(self.cfg.models.params),
            num_boost_round=self.cfg.models.num_boost_round,
            callbacks=[
                lgb.log_evaluation(self.cfg.models.verbose_eval),
                lgb.early_stopping(self.cfg.models.early_stopping_rounds),
                wandb_lgb.wandb_callback(),
            ],
        )

        wandb_lgb.log_summary(model)

        return model
