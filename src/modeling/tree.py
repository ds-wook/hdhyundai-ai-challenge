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
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _huber_approx_obj(self, preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        d = preds - dtrain.get_label()
        h = 1
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt

        return grad, hess

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
            obj=self._huber_approx_obj,
            num_boost_round=self.cfg.models.num_boost_round,
            early_stopping_rounds=self.cfg.models.early_stopping_rounds,
            verbose_eval=self.cfg.models.verbose_eval,
            callbacks=[wandb_xgb.WandbCallback()],
        )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> CatBoostRegressor:
        train_set = Pool(X_train, y_train)
        valid_set = Pool(X_valid, y_valid)

        model = CatBoostRegressor(
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
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> lgb.Booster:
        train_set = lgb.Dataset(X_train, y_train)
        valid_set = lgb.Dataset(X_valid, y_valid)

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
