from __future__ import annotations

import gc
import pickle
from abc import ABCMeta, abstractclassmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, Any]


class BaseModel(metaclass=ABCMeta):
    def __init__(self, cfg: DictConfig) -> NoReturn:
        self.cfg = cfg

    @abstractclassmethod
    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> NoReturn:
        raise NotImplementedError

    def save_model(self, save_dir: Path) -> NoReturn:
        with open(save_dir, "wb") as output:
            pickle.dump(self.result, output, pickle.HIGHEST_PROTOCOL)

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> NoReturn:
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    def run_cv_training(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> NoReturn:
        oof_preds = np.zeros(X.shape[0])
        models = {}
        kfold = KFold(n_splits=self.cfg.data.n_splits, shuffle=True, random_state=self.cfg.data.seed)

        for fold, (train_idx, valid_idx) in enumerate(kfold.split(X=X), 1):
            with wandb.init(dir="never", project=self.cfg.experiment.project, name=f"{self.cfg.models.results}-{fold}"):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = self.fit(X_train, y_train, X_valid, y_valid)
                oof_preds[valid_idx] = (
                    model.predict(X_valid)
                    if isinstance(model, lgb.Booster)
                    else model.predict(xgb.DMatrix(X_valid))
                    if isinstance(model, xgb.Booster)
                    else model.predict(X_valid.to_numpy()).reshape(-1)
                    if isinstance(model, TabNetRegressor)
                    else model.predict(X_valid)
                )
                models[f"fold_{fold}"] = model

                wandb.log({"Fold Score": mean_absolute_error(y_valid, oof_preds[valid_idx])})

            del X_train, X_valid, y_train, y_valid
            gc.collect()

        print(f"CV Score: {mean_absolute_error(y, oof_preds)}")

        self.result = ModelResult(oof_preds=oof_preds, models=models)
