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
train.head()
# %%

train["ID"].unique().shape
# %%
train.groupby(["ID"])["DIST"].mean()
# %%
train[train["ID"] == "A111164"]
# %%
train[train["ARI_PO"] == "ZAG4"]
# %%
