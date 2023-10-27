from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../config/", config_name="ensemble", version_base="1.3.1")
def _main(cfg: DictConfig):
    submit = pd.read_csv(Path(get_original_cwd()) / cfg.data.path / cfg.data.submit)

    median_list = [
        pd.read_csv(Path(get_original_cwd()) / cfg.output.path / median_pred)[cfg.data.target].to_numpy()
        for median_pred in tqdm(cfg.median_preds)
    ]

    median_preds = np.median(np.vstack(median_list), axis=0)

    submit[cfg.data.target] = median_preds
    submit.to_csv(Path(get_original_cwd()) / cfg.output.path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
