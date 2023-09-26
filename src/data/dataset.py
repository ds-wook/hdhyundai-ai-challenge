from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from engine.build import FeatureEngineer


def load_train_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.train}.csv")

    # feature engineering
    feature_engineering = FeatureEngineer(cfg, train)
    train = feature_engineering.get_train_pipeline()
    groups = train[cfg.data.groups]
    train = train.drop(columns=[*cfg.store.drop_features])
    train_y = train[cfg.data.target]
    train_x = train.drop(columns=[cfg.data.target])

    return train_x, train_y, groups


def load_test_dataset(cfg: DictConfig) -> pd.DataFrame:
    test = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.test}.csv")

    # feature engineering
    feature_engineering = FeatureEngineer(cfg, test)
    test = feature_engineering.get_test_pipeline()

    test_x = test.drop(columns=[*cfg.store.drop_features])

    return test_x
