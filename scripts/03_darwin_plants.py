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
# # Hypothesis test: Comparing the heights of plants grown from
# cross-fertilised and self-fertilised seeds
#
# ## Summary
#
# **description**
# : "a one sample **t**-test of diff=0"
#
# **data**
# : `data/darwin.csv`
#
# - **Cross** `int` : "height of plants grown from cross-breeding
#                        (1/8 inch)"
# - **Self** `int` : "height of plants grown from cross-breeding"
#                       (1/8 inch)
#
# **summary results**
# :
#
# ```python
# res(
#     "sample_mean_diff": 20.93333,
#     "95%_confint": (0.03119, 41.83547),
#     "tstat": 2.147,
#     "pval": 0.050,
#     "dof": 14)
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
# chage working dir
import os
os.chdir("..\\")

# %%
# import packages and modules
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import probplot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# set default seaborn theme
sns.set_theme()

# %%
# import the data
df_darwin = pd.read_csv("data\\darwin.csv")

# %% [markdown]
# ### Declare local variables

# %%
# calculate the paired differences
diff = df_darwin["Cross"] - df_darwin["Self"]

# %%
# declare and initialise DescrStatsW objects for CI
ttest = DescrStatsW(data=diff)

# %%
# declare and initialise results dictionary
res = dict()

# %% [markdown]
# ### Describe the samples

# %%
# descriptive table
pd.DataFrame(diff, columns=["Difference"]).describe().T

# %%
#  get 95% t-interval of difference
ttest.tconfint_mean()

# %% [markdown]
# ### Visualise the sample

# %%
# boxplot of difference
ax = sns.boxplot(x=diff)
# save figs, drop to figures dir
os.chdir("figures")
plt.savefig("darwin_fig1")
os.chdir("..")
# output the plot
plt.show()

# %%
# check normality of samples
f, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
# construct probability plots
probplot(x=df_darwin["Cross"], plot=ax1)
probplot(x=df_darwin["Self"], plot=ax2)
# add descriptive title and labels
ax1.set(title="Probability plot = Cross")
ax2.set(title="Probability plot = Self")
# save figs, drop to figures dir
os.chdir("figures")
plt.savefig("darwin_fig2")
os.chdir("..")
# output the plot
plt.show()

# %% [markdown]
# ### Run the $t$-test

# %%
# populate dictionary
res["tstat"], res["pval"], res["dof"] = ttest.ttest_mean()
# print results
res
