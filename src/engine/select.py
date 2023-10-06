from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor


def explaniable_selected_features(train: pd.DataFrame, label: pd.Series, test: pd.DataFrame) -> pd.DataFrame:
    model = LGBMRegressor(random_state=42)
    print(f"{model.__class__.__name__} Train Start!")
    model.fit(train, label)
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(test)
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([test.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ["column_name", "shap_importance"]

    importance_df = importance_df.sort_values("shap_importance", ascending=False)
    importance_df = importance_df.query("shap_importance > 1")
    boosting_shap_col = importance_df.column_name.values.tolist()

    print(f"Total {len(train.columns)} Select {len(boosting_shap_col)}")
    print(f"Select Features: {boosting_shap_col}")
    shap.summary_plot(shap_values, test)

    return importance_df
