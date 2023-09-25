from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from engine.base import BaseFeatureEngineer


class FeatureEngineer(BaseFeatureEngineer):
    def __init__(self, cfg: DictConfig, df: pd.DataFrame):
        super().__init__(cfg)
        self.df = df
