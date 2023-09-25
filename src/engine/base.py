from __future__ import annotations

import bisect
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class BaseFeatureEngineer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def _categorize_train_features(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical encoding
        Args:
            config: config
            train: dataframe
        Returns:
            dataframe
        """

        path = Path(get_original_cwd()) / self.cfg.data.encoder

        for cat_feature in tqdm(self.cfg.data.categorical_features, leave=False):
            le = LabelEncoder()
            train[cat_feature] = le.fit_transform(train[cat_feature])

            with open(path / f"{cat_feature}.pkl", "wb") as f:
                pickle.dump(le, f)

        return train

    def _categorize_test_features(self, test: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical encoding
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """

        path = Path(get_original_cwd()) / self.cfg.data.encoder

        for cat_feature in tqdm(self.cfg.data.categorical_features, leave=False):
            with open(path / f"{cat_feature}.pkl", "rb") as f:
                le = pickle.load(f)

            le_classes_set = set(le.classes_)
            test[cat_feature] = test[cat_feature].map(lambda s: "-1" if s not in le_classes_set else s)
            le_classes = le.classes_.tolist()
            bisect.insort_left(le_classes, "-1")
            le.classes_ = np.array(le_classes)
            test[cat_feature] = le.transform(test[cat_feature].astype(str))

        return test
