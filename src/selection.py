from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from lightgbm import LGBMRegressor
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from utils.plot import train_and_evaluate


@hydra.main(config_path="../config/", config_name="train", version_base="1.3.1")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        train_x, train_y = load_train_dataset(cfg)
        save_path = Path("output")

        # Model Tune for LGBM
        lgbm_feature_importances = train_and_evaluate(LGBMRegressor(), train_x, train_y)
        lgbm_feature_importances = lgbm_feature_importances.reset_index()
        lgbm_feature_importances.columns = ["feature", "importance"]
        lgbm_feature_importances = lgbm_feature_importances.sort_values(by="importance", ascending=False)
        lgbm_feature_importances.to_csv(save_path / "lgbm_feature_importances.csv", index=False)


if __name__ == "__main__":
    _main()
