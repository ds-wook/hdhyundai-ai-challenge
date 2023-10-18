from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from engine.base import BaseFeatureEngineer


class FeatureEngineer(BaseFeatureEngineer):
    def __init__(self, cfg: DictConfig, df: pd.DataFrame):
        super().__init__(cfg)
        df = self._add_time_features(df)
        df = self._add_trend_features(df)
        df = self._add_basic_features(df)
        self.df = df

    def get_train_pipeline(self):
        self.df = self._categorize_train_features(self.df)
        return self.df

    def get_test_pipeline(self):
        self.df = self._categorize_test_features(self.df)
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
        df["month"] = df["ATA"].dt.month
        df["day"] = df["ATA"].dt.day
        df["hour"] = df["ATA"].dt.hour
        df["minute"] = df["ATA"].dt.minute
        df["weekday"] = df["ATA"].dt.weekday

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend features
        Args:
            df: dataframe
        Returns:
            dataframe
        """
        trend_features = ["AIR_TEMPERATURE", "DIST"]

        for col in tqdm(trend_features, leave=False):
            df[f"{col}_diff1"] = df[col] - df.groupby("ARI_PO")[col].shift(1)

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["DIST"] = np.log1p(df["DIST"])

        return df
