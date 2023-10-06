# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%

train = pd.read_csv("../input/hdhyundai-ai-challenge/train.csv")
test = pd.read_csv("../input/hdhyundai-ai-challenge/test.csv")
# %%

train["ID"].unique().shape
# %%
test["ID"].unique().shape
# %%
