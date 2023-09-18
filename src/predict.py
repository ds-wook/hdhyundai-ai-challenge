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
    preds = np.zeros((test_x.shape[0],))

    for model in tqdm(result.models.values(), total=folds):
        preds += (
            model.predict(test_x) / folds
            if isinstance(model, lgb.Booster)
            else model.predict(xgb.DMatrix(test_x)) / folds
            if isinstance(model, xgb.Booster)
            else model.predict(test_x)[:, 1] / folds
        )

    assert len(preds) == len(test_x)

    return preds


@hydra.main(config_path="../config/", config_name="predict")
def _main(cfg: DictConfig):
    test_x = load_test_dataset(cfg)
    folds = range(cfg.data.n_splits)
    submit = pd.read_csv(Path(cfg.data.path) / cfg.data.submit)

    for fold in tqdm(folds, leave=False):
        preds = inference_models(cfg, test_x)

    submit["prediction"] = preds

    submit.to_csv(Path(cfg.output.path) / f"{cfg.models.results}.csv", index=False)


if __name__ == "__main__":
    _main()
