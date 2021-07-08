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
# # Modelling the emission of alphaparticles as a Poisson distribution
#
# ## Summary
#
# ### Question of interest
#
# Can the number of alpha particles emitted from a radioactive source be
# modelled using a Poisson distribution?
#
# ### Data
#
# - Data shows the frequency of each number of emissions by a radioactive
#   source in a 7.5s time interval
# - Data fields
#   - **count** `str` : number of alpha-particle emissions
#   - **observed**7, `int` : observed frequency of each **count**
#
# ### Method
#
# - Data modelled using the Poisson distribution
#   - Sample mean number of observations calculated, used instead of the
#     unknown population mean
#   - Expected number of each emission count calculated
#   - All expected values > 5 checked
#   - Data was remodelled after combining groups `[11, 12, 13+] -> [13+]`
# - Model was checked using a side-by-side chart and checking for
#   equality between the sample mean and sample variance
# - Performed a **Chi-square** goodness-of-fit test: data modelled by a
#   Poisson distribution.
#
# ### Summary results
#
# - Data modelled using the Poisson distribution
#   - Sample mean number of observations calculated as $\overline{X}=$ 3.887
#   - Expected number of each emission count calculated
#   - Counts `[11, 12, 13+]` had expected values < 5
#   - Counts `[11, 12, 13+]` combined into a single new category `[11+]`,
#     and expected number recalculated
# - The side-by-side bar chart shows the observed and expected
#   frequencies have a similar distrubtion
# - Sample variance $s^{2}=$ 3.682, near enough the sample mean to not
#   cause concern
# - Performed **chi**-square goodness-of-fit
#   - One estimated property, sample mean
#   - `ResultSummary(chisq=10.417085, pval=0.404694, dof=9)`
#
# ### Discussion
#
# - Null hypothesis is not rejected with **p**-value > 0.1
# - There is little to no evidence against the hypothesis that that the
#   data has a Poisson distribution
#
# ### Reference
#
# No reference
#
# -----

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
alpha: pd.DataFrame = load.Data.get("alpha_particles")
alpha

# %%
# check dtypes
alpha.dtypes

# %%
# get samples as series
obs_x: pd.Series = alpha["observed"]

# %% [markdown]
# ### Model the data

# %%
# x, number of emissions
x: np.array = np.arange(start=0, stop=14)

# %%
# get weighted sample mean
mu: float = (x * obs_x).sum() / obs_x.sum()
mu

# %%
# model the data
pois_model: stats.poisson = stats.poisson(mu)
pmf_x: np.array = pois_model.pmf(x)
pmf_x[13] = 1 - pois_model.cdf(12)

# %%
exp_x: np.array = pmf_x * obs_x.sum()
alpha["expected"] = exp_x
alpha.query('expected < 5')

# %% [markdown]
# Not all expected values are greater than 5.
# Combine counts `11, 12, 13+ -> 11+`.

# %%
# remodel the data
x2: pd.Series = pd.Series(np.arange(start=0, stop=12))
x2[11] = ">=11"
obs_x2: pd.Series = obs_x[0:11]
obs_x2[11] = obs_x[11:].sum()
exp_x2: pd.Series = pd.Series(exp_x[0:11])
exp_x2[11] = exp_x[11:].sum()
alpha2: pd.DataFrame = pd.DataFrame(
    {'count': x2, 'observed': obs_x2, 'expected': exp_x2})
alpha2

# %% [markdown]
# All expected values now >= 5, so we can continue modelling.

# %% [markdown]
# ### Visualise the data

# %%
# side-by-side barchart
mdata = alpha2.melt(id_vars="count", value_vars=["observed", "expected"])
ax = sns.barplot(data=mdata, x="count", y="value", hue="variable")
ax.set(xlabel="Number of emissions", ylabel="Frequency")
plt.show()

# %% [markdown]
# ### Begin analysis

# %%
# Check sample mean = sample var
# simulate experimental results
res: list = list()
for i in range(0, 12):  # obs in the range of x
    for j in range(0, obs_x2[i]):  # obs should appear obs_x[i] times
        res.append(i)
res: pd.Series = pd.Series(res)
res.var()

# %%
chisq, pval = stats.chisquare(f_obs=obs_x2, f_exp=exp_x2, ddof=1)
summarise.ChisqGoodnessOfFit(chisq, pval, 9)
