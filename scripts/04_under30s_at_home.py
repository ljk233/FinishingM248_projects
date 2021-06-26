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
# # Hypothesis test: Comparing whether the number of Under-30s still living at home in Wales is greater than the national average
#
# ## Summary
#
# **description**
# : "a one sample test of a proportion"
#
# **data**
# : `None`
#
# **summary results**
# :
#
# ```python
# res(
#     "p_hat": 0.26772,
#     "95%_confint": (0.21327, 0.32217),
#     "zstat": 0.652,
#     "pval": 0.514)
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
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

# %% [markdown]
# ### Declare local variables

# %%
# count, size
x = 68
n = 254

# %%
# alterative proportion, national average
p0 = 0.25

# %%
# declare and initialise results dictionary
res = dict()

# %% [markdown]
# ### Describe the data

# %%
# estimate proportion
p_hat = x / n

# %%
# 95% ci of proportion
zinterval = proportion_confint(count=x, nobs=n)

# %%
# output estimates
p_hat, zinterval

# %% [markdown]
# ### Run the $t$-test

# %%
# populate results
res["zstat"], res["pval"] = (
    proportions_ztest(count=x, nobs=n, value=p0, prop_var=p0))
# print results
res
