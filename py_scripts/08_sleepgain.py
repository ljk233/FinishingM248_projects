# ---
# jupyter:
#   jupytext:
#     formats: py_notebooks///ipynb,py_scripts///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3.8
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Wilcoxon
#
# ## Summary
#
# **description**
# : "wilcoxon sign-rank"

# %% [markdown]
# ## Results

# %% [markdown]
# ### Setup the notebook

# %%
# import packages and modules
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# %%
# set default seaborn theme
sns.set_theme()

# %%
# import custom modules not in root
sys.path[0] = "..\\"  # update path
from src import load  # noqa: E402

# %% [markdown]
# ### Import the data

# %%
# get data
sleep_gain: pd.DataFrame = load.Data.get("sleep_gain")
sleep_gain.head()

# %%
# check dtypes
sleep_gain.dtypes

# %%
# get samples
sleep_gain.sort_values(by=["id", "treatment"], inplace=True)
ltreat: np.array = sleep_gain.query(
    'treatment == "Ltreatment"')["sleep_gain"].to_numpy()
dtreat: np.array = sleep_gain.query(
    'treatment == "Dtreatment"')["sleep_gain"].to_numpy()
diff_treat: np.array = ltreat - dtreat

# %% [markdown]
# ### Visualise the sample

# %%
# boxplot of difference
ax = sns.boxplot(data=sleep_gain, x="sleep_gain", y="treatment")
ax.set(xlabel="Time", ylabel="Treatment")
plt.show()

# %%
# probability plots
f, axs = plt.subplots(ncols=2, sharey=True)
stats.probplot(x=ltreat, plot=axs[0])
stats.probplot(x=dtreat, plot=axs[1])
axs[0].set(title="Probability plot = Ltreatment")
axs[1].set(title="Probability plot = Dtreatment")
plt.show()

# %% [markdown]
# ### Analyse the data

# %%
# ltreat: check observations equal to 0
count: int = 0
for check in ltreat == 0:
    if check == True:
        count += 1
count

# %%
stats.wilcoxon(x=ltreat, alternative="greater")

# %%
# dtreat: check observations equal to 0
count: int = 0
for check in dtreat == 0:
    if check == True:
        count += 1
count

# %%
stats.wilcoxon(x=dtreat, alternative="greater")

# %%
# get difference
diff = ltreat - dtreat

# %%
stats.wilcoxon(x=diff_treat)
