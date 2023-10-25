from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from data.dataset import load_test_dataset, load_train_dataset
from engine.select import explaniable_selected_features


@hydra.main(config_path="../config/", config_name="train", version_base="1.3.1")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        train_x, train_y = load_train_dataset(cfg)
        test_x = load_test_dataset(cfg)
        save_path = Path("config/store/")

        # Model Tune for LGBM
        lgbm_feature_importances = explaniable_selected_features(train_x, train_y, test_x)
        boosting_shap_col = lgbm_feature_importances.column_name.values.tolist()

        basic_features = OmegaConf.load("config/store/features.yaml")
        basic_features["selected_features"] = boosting_shap_col
        basic_features["categorical_features"] = [
            col for col in boosting_shap_col if col in [*cfg.data.categorical_features]
        ]
        OmegaConf.save(basic_features, save_path / "features.yaml")


if __name__ == "__main__":
    _main()
