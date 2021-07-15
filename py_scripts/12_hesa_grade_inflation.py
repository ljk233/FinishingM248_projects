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
# # Are the more firsts being awarded?
#
# ## Notes
#
# ### Question of interest
#
# Were there more firsts being awarded in 2018 compared to 2014?
#
# ### Method
#
# - Data modelled using an approximated normal model
#   - Checked with a frequency histogram and normal probability plot
# - Calculated mean percentage of first class honours and 95%
#   **z**-interval calculated for both the 2014 and 2018 academic years
# - Performed a two sample, one-tailed **z**-test of the hypothesis of
#   equal means

# %% [markdown]
# ## Results

# %% [markdown]
# ### Setup the notebook

# %%
# import packages and modules
from scipy import stats
import pandas as pd
import statsmodels.stats.weightstats as sm
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# %%
# set seaborn theme
sns.set_theme()

# %%
# import custom modules not in root
sys.path[0] = "..\\"  # update path
from src import load, describe, summarise  # noqa: E402

# %% [markdown]
# ### Import the data

# %%
# get data
data: pd.DataFrame = load.Data.get("hesa_grade_inflation")

# %%
# preview data
data.head()

# %%
# check dtypes
data.dtypes

# %%
# recast columns to int
data[["f_2014", "n_2014", "f_2018", "n_2018"]] = (
    data[["f_2014", "n_2014", "f_2018", "n_2018"]].apply(pd.to_numeric))
data.dtypes

# %%
# get both sample percentages
s1 = ((data["f_2014"] / data["n_2014"]) * 100).to_numpy()  # sample 1: AY2014
s2 = ((data["f_2018"] / data["n_2018"]) * 100).to_numpy()  # sample 1: AY2018

# %% [markdown]
# ### Visualise the data

# %%
# frequency histograms
f, axs = plt.subplots(ncols=2, sharey=True, figsize=(11, 4))
sns.histplot(x=s1, bins=10, ax=axs[0])
sns.histplot(x=s2, bins=10, ax=axs[1])
axs[0].set(title="Academic year = 2014")
axs[1].set(title="Academic year = 2018")
plt.show()

# %%
# probability plots
f, axs = plt.subplots(ncols=2, sharey=True, figsize=(11, 4))
stats.probplot(x=s1, plot=axs[0])
stats.probplot(x=s2, plot=axs[1])
axs[0].set(title="Probability plot = Etruscans")
axs[1].set(title="Probability plot = Italians")
plt.show()

# %% [markdown]
# ### Analyse the data

# %%
cm = sm.CompareMeans.from_data(data1=s1, data2=s2)

# %%
describe.ZSample(
    "AY2014", cm.d1.nobs, cm.d1.mean, cm.d1.zconfint_mean())

# %%
describe.ZSample(
    "AY2018", cm.d2.nobs, cm.d2.mean, cm.d2.zconfint_mean())

# %%
describe.ZDiff(
    cm.d1.mean - cm.d2.mean, cm.zconfint_diff())

# %%
# run the test
zstat, pval = cm.ztest_ind(alternative="smaller")
summarise.ZTest(zstat, pval)
