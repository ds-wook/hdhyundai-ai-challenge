from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from engine.base import BaseFeatureEngineer


class FeatureEngineer(BaseFeatureEngineer):
    def __init__(self, cfg: DictConfig, df: pd.DataFrame):
        super().__init__(cfg)
        df = self._add_time_features(df)
        df = self._add_basic_features(df)
        df = self._add_trend_features(df)
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

    def _fill_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing features
        Args:
            df: dataframe
        Returns:
            dataframe
        """
        for col in tqdm(["AIR_TEMPERATURE", "U_WIND", "V_WIND"], leave=False):
            df[col] = df[col].fillna(df.groupby("ID")[col].transform("mean"))

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend features
        Args:
            df: dataframe
        Returns:
            dataframe
        """
        trend_features = ["DUBAI", "BRENT", "AIR_TEMPERATURE", "U_WIND", "V_WIND", "DIST", "PORT_SIZE"]

        for col in tqdm(trend_features, leave=False):
            df[f"{col}_diff1"] = df[col] - df.groupby("ID")[col].shift(1)

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        all_cols = [c for c in list(df.columns) if c not in ["ID", self.cfg.data.target]]
        num_features = [col for col in all_cols if col not in self.cfg.data.categorical_features]
        df_num_agg = df.groupby("ID")[num_features].agg(["mean", "std", "min", "max"])
        df_num_agg.columns = ["_".join(x) for x in df_num_agg.columns]
        df_num_agg.reset_index(inplace=True)

        df_cat_agg = df.groupby("ID")[[*self.cfg.data.categorical_features]].agg(["count", "nunique"])
        df_cat_agg.columns = ["_".join(x) for x in df_cat_agg.columns]
        df_cat_agg.reset_index(inplace=True)

        df = df.merge(df_num_agg, how="inner", on="ID")
        df = df.merge(df_cat_agg, how="inner", on="ID")

        return df
