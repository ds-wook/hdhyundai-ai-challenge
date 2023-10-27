from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from engine.base import BaseFeatureEngineer


class FeatureEngineer(BaseFeatureEngineer):
    def __init__(self, cfg: DictConfig, df: pd.DataFrame):
        super().__init__(cfg)
        df = self._add_time_features(df)
        df = self._add_basic_features(df)
        self.df = df

    def get_train_pipeline(self):
        """
        Get train pipeline
        Returns:
            dataframe
        """
        self.df = self._one_hot_encoding(self.df)
        return self.df

    def get_test_pipeline(self):
        """
        Get test pipeline
        Returns:
            dataframe
        """
        self.df = self._one_hot_encoding(self.df)
        return self.df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time features
        Args:
            df: dataframe
        Returns:
            dataframe
        """
        df["ATA"] = pd.to_datetime(df["ATA"])
        df["year"] = df["ATA"].dt.year
        df["weekday"] = df["ATA"].dt.weekday

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic features
        Args:
            df: dataframe
        Returns:
            dataframe
        """
        # df["DIST_dev"] = (df["DIST"] - df["DIST"].mean()) ** 2/
        df["BN_LARGER"] = df["BN"].apply(lambda x: 1 if np.abs(x) > 5 else 0)

        return df
