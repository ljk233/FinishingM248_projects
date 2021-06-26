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
# # Hypothesis test: Pass rate of the practical driving test in the UK
#
# ## Summary
#
# **description**
# : "a one sample **z**-test of equal means"
#
# **data**
# : `data/practical_test.csv`
#
# - **Centre** `str` : "location of test centre"
# - **Male** `float` : "mean percentage pass rate of males at the
#                       centre from April 2014 to March 2015"
# - **Female** `float` : "mean percentage pass rate of females at the
#                         centre from April 2014 to March 2015"
# - **Total** `float` : "mean percentage pass rate of both males and
#                        females from April 2014 to March 2015"
#
# **summary results**
# :
#
# ```python
# res(
#     "sample_mean": 49.63038,
#     "95%_confint": (48.84034, 50.42042),
#     "zstat": 6.277,
#     "pval": 0.00000)
# ```
#
# **output**
# : <!--Add path to FinishingM248-->
#

# %% [markdown]
# ## Results
#
# ### Setup the notebook

# %%
# change working dir
import os
os.chdir("..\\")

# %%
# import packages
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
df_prac_test = pd.read_csv("data\\practical_test.csv")

# %% [markdown]
# ### Declare local variables

# %%
# get columns as Series
r_total = df_prac_test["Total"]

# %%
# declare and initialise DescrStatsW object for the hypothesis test
ztest = DescrStatsW(data=r_total)

# %%
# declare and initialise results dictionary
res = dict()

# %% [markdown]
# ### Describe the sample

# %%
# send summary table to df for pretty output
pd.DataFrame(r_total).describe().T

# %%
# 95% z-interval
ztest.zconfint_mean()

# %% [markdown]
# ### Visualise the sample

# %%
# plot the sample as a histogram
ax = sns.histplot(x=r_total, bins=12)
# save figs, drop to figures dir
os.chdir("figures")
plt.savefig("practest_fig1")
os.chdir("..")
# output the plot
plt.show()

# %%
# normal probability plot of the same
f, ax = plt.subplots()
# construct the probability plot
probplot(x=r_total, plot=ax)
# save figs, drop to figures dir
os.chdir("figures")
plt.savefig("practest_fig2")
os.chdir("..")
# output the plot
plt.show()

# %% [markdown]
# ### Run the $z$-test

# %%
# populate res
res["zstat"], res["pval"] = ztest.ztest_mean(value=47.1)
# print results
res
