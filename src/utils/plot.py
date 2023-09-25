from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor


def train_and_evaluate(model: LGBMRegressor, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    print(f"Model Tune for {model.__class__.__name__}.")
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, len(X_train.columns)))
    plt.title(f"Feature Importances ({model.__class__.__name__})")
    feature_importances_df = pd.DataFrame(index=X_train.columns, columns=["importance"], data=feature_importances)
    plt.barh(range(X_train.shape[1]), feature_importances[sorted_idx], align="center")
    plt.yticks(range(X_train.shape[1]), X_train.columns[sorted_idx])
    plt.xlabel("Importance")
    plt.show()

    return feature_importances_df
