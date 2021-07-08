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
# # Laurel and Hardy films

# %% [markdown]
# ## Results

# %% [markdown]
# ### Setup the notebook

# %%
# import packages and modules
from scipy import stats
import pandas as pd
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
laurel_hardy: pd.DataFrame = load.Data.get("laurel_hardy")
laurel_hardy.head()

# %%
# check dtypes
laurel_hardy.dtypes

# %%
# get samples as series
silent: pd.Series = laurel_hardy.query('type == "Silent"')["time"]
sound: pd.Series = laurel_hardy.query('type == "Sound"')["time"]

# %% [markdown]
# ### Visualise the sample

# %%
# boxplot of difference
ax = sns.boxplot(x=laurel_hardy["time"], y=laurel_hardy["type"])
plt.show()

# %%
# probability plots
f, axs = plt.subplots(ncols=2, sharey=True)
stats.probplot(x=silent, plot=axs[0])
axs[0].set(title="Probability plot = Silent")
stats.probplot(x=sound, plot=axs[1])
axs[1].set(title="Probability plot = Sound")
plt.show()

# %% [markdown]
# ### Analyse the data

# %%
stats.mannwhitneyu(
    x=silent, y=sound, use_continuity=False, method="asymptotic")
