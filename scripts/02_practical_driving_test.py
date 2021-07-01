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
# # Pass rate of the practical driving test in the UK
#
# ## Summary
#
# ### Data
#
# Centre `str` :
# location of test centre
#
# Male `float` :
# mean percentage pass rate of males at the centre from April 2014 to
# March 2015
#
# Female `float` :
# mean percentage pass rate of females at the centre from April 2014 to
# March 2015
#
# Total `float` :
# mean percentage pass rate of both males and females from April 2014
# to March 2015
#
# ### Method
#
# - Data modelled using an apporximate normal distribution
# - Frequency histogram and normal probability plot to check symmetry of
#   of data
# - Mean and 95% **z**-interval returned for samples
# - One sample, two-tailed **z**-test used to test the hypothesis that
#   the mean total pass rate of the UK driving practical test was equal
#   to 47.1%.
#
# ### Summary results
#
# ```python
# test_results{'zstat': 6.277, 'pval': 3.441}
# ```
#
# ### Output
#
# <!--Add path to FinishingM248-->
#
# ### Reference
#
# m248.b.act23
#
# -----

# %% [markdown]
# ## Results
#
# ### Setup the notebook

# %%
# import packages and modules
from scipy import stats
import statsmodels.stats.weightstats as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# %%
# set seaborn theme
sns.set_theme()

# %%
# change wkdir and import the data
os.chdir("..\\")
data = pd.read_csv("data\\practical_test.csv")

# %% [markdown]
# ### Preview and describe the data

# %%
data[["Total"]].head()

# %%
data[["Total"]].describe().T

# %% [markdown]
# ### Visualise the data

# %%
# frequency histogram
f, ax = plt.subplots()
ax = sns.histplot(data=data, x="Total", bins=12)
os.chdir("figures")
plt.savefig("practest_fig1")
os.chdir("..")
plt.show()

# %%
# probability plot of sample
f, ax = plt.subplots()
stats.probplot(x=data["Total"], plot=ax)
os.chdir("figures")
plt.savefig("practest_fig2")
os.chdir("..")
plt.show()

# %% [markdown]
# ### Run the hypothesis test

# %%
d1 = sm.DescrStatsW(data=data["Total"])

# %% [markdown]
# #### Get confidence interval

# %%
d1.zconfint_mean()

# %% [markdown]
# #### Run the test

# %%
mu0 = 47.1

# %%
zstat, pval = d1.ztest_mean(value=mu0)

# %%
zstat, pval
