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

sns.histplot(data=np.log1p(train["CI_HOUR"]), kde=True)


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
train.groupby(["ID", "ATA"])[["AIR_TEMPERATURE", "U_WIND", "V_WIND"]].agg(["count", "mean", "std", "min", "max"])

# %%
train[["ID", "ATA"]].head()
# %%
train.loc[train["ID"] == "A111164"]
# %%
