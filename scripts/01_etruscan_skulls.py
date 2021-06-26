# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,scripts///py:percent
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
# # Hypothesis test: Comparing skull sizes of Etruscans and Italians
#
# ## Summary
#
# **description**
# : "a two sample **t**-test of equal means"
#
# **data**
# : `data/skulls.csv`
#
# - **Etruscan** `float` : "breadth of etruscan skulls in mm"
# - **Italian** `float` : "breadth of italian skulls in mm"
#
# **summary results**
# :
#
# ```python
# res(
#     "diff": 11.3310,
#     "95%_confint": (9.454, 13.208),
#     "tstat": 11.925,
#     "pval": 0.000,
#     "dof": 152)
# ```
#
# **output**
# : <!--Add path to FinishingM248-->
#

# %% [markdown]
# ## Results

# %% [markdown]
# ### Setup the notebook

# %%
# chage working dir, testing update
import os
os.chdir("..\\")

# %%
# import packages and modules
from statsmodels.stats.weightstats import CompareMeans
from scipy.stats import probplot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# set default seaborn theme
sns.set_theme()

# %%
# import the data
df_skulls = pd.read_csv("data\\skulls.csv")

# %% [markdown]
# ### Declare local variables

# %%
# get columns as Series
etruscan = df_skulls["Etruscans"].dropna()
italian = df_skulls["Italians"].dropna()

# %%
# declare and initialise CompareMeans object for the hypothesis test
ttest = CompareMeans.from_data(data1=etruscan, data2=italian)

# %%
# construct new df for histogram by melting df_skulls from wide -> long
mskulls = df_skulls.melt(
    value_vars=["Etruscans", "Italians"],
    var_name="skull",
    value_name="size")
# drop NaN value, cast column to int
mskulls.dropna(inplace=True)
mskulls["size"] = mskulls["size"].astype("int")

# %% [markdown]
# ### Describe the data

# %%
# get descriptive table
df_skulls.describe().T

# %%
# 95% t-interval etruscan
ttest.d1.tconfint_mean()

# %%
# 95% t-interval italian
ttest.d2.tconfint_mean()

# %% [markdown]
# ### Visualise the data

# %%
# frequency histograms
g = sns.FacetGrid(mskulls, col="skull")
g.map_dataframe(sns.histplot, x="size", bins=10)
# save figs, drop to figures dir
os.chdir("figures")
plt.savefig("skulls_fig1")
os.chdir("..")
plt.show()

# %%
# Check normality of samples
# declare and initialise the fig and axes
f, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
# construct probability plots
probplot(x=etruscan, plot=ax1)
probplot(x=italian, plot=ax2)
# add discriptive title and labels
ax1.set(title="Probability plot = Etruscans")
ax2.set(title="Probability plot = Italians")
# save figs, drop to figures dir
os.chdir("figures")
plt.savefig("skulls_fig2")
os.chdir("..")
# output the probability plots
plt.show()

# %% [markdown]
# ### Run the hypothesis test

# %%
# Check assumption of common population variance
etruscan.var(), italian.var()

# %%
ttest.summary()

# %%
# get degrees of freedom
ttest.d1.nobs + ttest.d2.nobs - 2
