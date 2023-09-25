from __future__ import annotations

import pickle
from pathlib import Path

import hydra
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from tqdm import tqdm

from data.dataset import load_test_dataset


def inference_models(cfg: DictConfig, test_x: pd.DataFrame) -> np.ndarray:
    """
    Given a model, predict probabilities for each class.
    Args:
        model_results: ModelResult object
        test_x: test dataframe
    Returns:
        predict probabilities for each class
    """
    model_path = Path(cfg.models.path) / f"{cfg.models.results}.pkl"

    with open(model_path, "rb") as output:
        result = pickle.load(output)

    folds = len(result.models)
    ensemble_preds = []

    for model in tqdm(result.models.values(), total=folds, desc="Predicting models"):
        model_pred = (
            model.predict(test_x)
            if isinstance(model, lgb.Booster)
            else model.predict(xgb.DMatrix(test_x))
            if isinstance(model, xgb.Booster)
            else model.predict(test_x)[:, 1]
        )
        pred = np.where(model_pred < 0, 0, model_pred)
        ensemble_preds.append(pred)

    predictions = np.median(ensemble_preds, axis=0)

    return predictions


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    test_x = load_test_dataset(cfg)
    submit = pd.read_csv(Path(cfg.data.path) / cfg.data.submit)

    preds = inference_models(cfg, test_x)

    submit[cfg.data.target] = preds

    submit.to_csv(Path(cfg.output.path) / f"{cfg.models.results}.csv", index=False)


if __name__ == "__main__":
    _main()
