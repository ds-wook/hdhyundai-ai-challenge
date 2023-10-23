from __future__ import annotations

import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from tqdm import tqdm

from data.dataset import load_test_dataset, load_train_dataset


def postprocess(train_y: pd.Series, preds: np.ndarray) -> np.ndarray:
    """
    Postprocess data
        Parameter:
            train: train dataset
            preds: inference prediction
        Return:
            preds: median prediction
    """
    all_pressure = np.sort(train_y.unique())
    print("The first 25 unique pressures...")
    pressure_min = all_pressure[0].item()
    pressure_max = all_pressure[-1].item()
    pressure_step = (all_pressure[1] - all_pressure[0]).item()

    # ENSEMBLE FOLDS WITH MEDIAN AND ROUND PREDICTIONS
    preds = np.round((preds - pressure_min) / pressure_step) * pressure_step + pressure_min
    preds = np.clip(preds, pressure_min, pressure_max)

    return preds


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
        model_pred = model.predict(xgb.DMatrix(test_x)) if isinstance(model, xgb.Booster) else model.predict(test_x)
        # model_pred = np.where(model_pred < 0.01, 0, model_pred)
        ensemble_preds.append(model_pred)

    predictions = np.median(ensemble_preds, axis=0)

    return predictions


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    train_x, train_y, groups = load_train_dataset(cfg)
    test_x = load_test_dataset(cfg)
    test_x = test_x[cfg.store.selected_features]
    submit = pd.read_csv(Path(cfg.data.path) / cfg.data.submit)

    preds = inference_models(cfg, test_x)
    preds = postprocess(train_y, preds)
    submit[cfg.data.target] = preds

    submit.to_csv(Path(cfg.output.path) / f"{cfg.models.results}.csv", index=False)


if __name__ == "__main__":
    _main()
