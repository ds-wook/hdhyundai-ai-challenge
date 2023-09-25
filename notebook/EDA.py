# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%

train = pd.read_csv("../input/hdhyundai-ai-challenge/train.csv")
# %%
train.info()
# %%

train["CI_HOUR"]


# %%
def postprocess(cfg, train: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """
    Postprocess data
        Parameter:
            train: train dataset
            preds: inference prediction
        Return:
            preds: median prediction
    """
    all_pressure = np.sort(train[cfg.data.target].unique())
    print("The first 25 unique pressures...")
    pressure_min = all_pressure[0].item()
    pressure_max = all_pressure[-1].item()
    pressure_step = (all_pressure[1] - all_pressure[0]).item()

    # ENSEMBLE FOLDS WITH MEDIAN AND ROUND PREDICTIONS
    preds = np.round((preds - pressure_min) / pressure_step) * pressure_step + pressure_min
    preds = np.clip(preds, pressure_min, pressure_max)

    return preds


# %%
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor


def train_and_evaluate(model: LGBMRegressor, model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    print(f"Model Tune for {model_name}.")
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, len(X_train.columns)))
    plt.title(f"Feature Importances ({model_name})")
    plt.barh(range(X_train.shape[1]), feature_importances[sorted_idx], align="center")
    plt.yticks(range(X_train.shape[1]), X_train.columns[sorted_idx])
    plt.xlabel("Importance")
    plt.show()

    return feature_importances



# %%
