from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from engine.build import FeatureEngineer


def load_train_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.train}.csv")

    train_y = train[cfg.data.target]
    train = train[cfg.store.selected_features]

    # feature engineering
    feature_engineering = FeatureEngineer(cfg, train)
    train = feature_engineering.get_train_pipeline()
    train_x = train.drop(columns=[*cfg.store.drop_features])

    return train_x, train_y


def load_test_dataset(cfg: DictConfig) -> pd.DataFrame:
    test = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.test}.csv")
    test = test[cfg.store.selected_features]

    # feature engineering
    feature_engineering = FeatureEngineer(cfg, test)
    test = feature_engineering.get_test_pipeline()

    test_x = test.drop(columns=[*cfg.store.drop_features])

    return test_x
