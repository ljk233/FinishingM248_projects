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
# # Modelling the emission of alphaparticles as a Poisson distribution
#
# ## Summary
#
# ### Data
#
# count, `str` :
# number of alpha-particle emissions
#
# observed, `int` :
# observed frequency of each number of alpha-particles
#
# ### Method
#
# - Data modelled using the Poisson distribution
# - Sample mean number of observations calculated, used instead of the
#   unknown population mean
# - Expected number of each emission count calculated
# - Counts 11, 12, 13+ had expected values < 5, so they were combined
#   into a single new category (11+) and expected number recalculated
# - Observed and expected frequencies plotted as a side-by-side plot
# - Chi-square goodness-of-fit test used to test the hypothesis that the
#   emissions of alpha-particles can be modelled by a Poisson
#   distribution
#
# ### Summary results
#
# ```python
# test_results{'chisq': 10.417, 'pval': 0.405, "dof": 11}
# ```
#
# ### Output
#
# <!--Add path to FinishingM248-->
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
import os

# %%
# set seaborn theme
sns.set_theme()

# %%
# change wkdir and import the data
os.chdir("..")
data = pd.read_csv("data\\alpha_particles.csv")
obs_x = data["observed"]

# %% [markdown]
# ### Preview the data

# %%
data

# %% [markdown]
# ### Get expected number of observations

# %% [markdown]
# #### Declare `X`, number of emissions
#
# This will be used to generate the expected number of emissions.
# We cannot use the `count` column as it is not a column of `int`.

# %%
x = np.arange(start=0, stop=14)

# %% [markdown]
# #### Calculate weighted mean

# %%
mu = (x * obs_x).sum() / obs_x.sum()
mu

# %% [markdown]
# #### Construct the Poisson model
#
# Array `pmf_x` will not initially be correct, as the final entry will
# show **P(X=13)**, not **P(X>=13)**, so we replace it.

# %%
pois_model = stats.poisson(mu)
pmf_x = pois_model.pmf(x)
pmf_x[13] = 1 - pois_model.cdf(12)
# check
pmf_x.sum() == 1

# %% [markdown]
# #### Generate expected number of observations

# %%
exp_x = pmf_x * obs_x.sum()
data["expected"] = exp_x
# check
data.query('expected < 5')

# %% [markdown]
# Not all expected values are greater than 5. 
# Combine counts `11, 12, 13+ -> 11+`.

# %%
x2 = pd.Series(np.arange(start=0, stop=12))
x2[11] = ">=11"

# %%
obs_x2 = obs_x[0:11]
obs_x2[11] = obs_x[11:].sum()

# %%
exp_x2 = pd.Series(exp_x[0:11])
exp_x2[11] = exp_x[11:].sum()

# %%
data2 = pd.DataFrame(
    {'count': x2, 'observed': obs_x2, 'expected': exp_x2})

# %%
data2

# %% [markdown]
# ### Visualise the data

# %%
# side-by-side barchart
malpha = data2.melt(id_vars="count", value_vars=["observed", "expected"])
ax = sns.barplot(data=malpha, x="count", y="value", hue="variable")
ax.set(xlabel="Number of emissions", ylabel="Frequency")
os.chdir("figures")
plt.savefig("alpha_fig1")
os.chdir("..")
plt.show()

# %% [markdown]
# ### Run the hypothesis test

# %%
chisq, pval = stats.chisquare(f_obs=obs_x2, f_exp=exp_x2, ddof=1)

# %%
chisq, pval
