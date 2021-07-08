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

# %%
# noqa:  E501

# %% [markdown]
# # Was the number of Under-30s still living at home in Wales in 2014 greater than the national average?  # noqa:  E501
#
# ## Notes
#
# ### Question of interest
#
# Was the the proportion of under-30s still living at home with their
# family in Wales in 2014 greater than the national average of 25%?
#
# ### Data
#
# Data in the form of questionnaire results for the ONS in **year**:
# - Survey size $n=$ 254
# - Number living at home $x=$ 68
# - National average $p_{0}=$ 0.25
#
# ### Method
#
# - Data modelled by a normal approximation to the binomial
# - Point estimate $\widehat{p}_{W}$ and 95% **z**-interval of
#   $\widehat{p}_{W}$ calculated
# - Performed **test of a proportion**: proption greater than 0.25
#
# ### Results
#
# - $\widehat{p}_{W}$ and 95% **z**-interval of $\widehat{p}_{W}$
#   - `Proportion(p_hat=0.267717, zconfint_prop=(0.213265, 0.322168))`
# - Test of a proportion result
#   - `ResultSummary(zstat=0.652071, pval=0.257178)`
#
# ### Discussion
#
# - Null hypothesis is not rejected with **p**-value > 0.1
# - No evidence that the proportion of young people living at home with
#   their family in Wales is greater than the national average
#
# ### Reference
#
# -----

# %% [markdown]
# ## Results

# %% [markdown]
# ### Setup the notebook

# %%
# import packages and modules
import statsmodels.stats.proportion as sm
import sys

# %%
# import custom modules not in root
sys.path[0] = "..\\"  # update path
from src import describe, summarise  # noqa: E402

# %% [markdown]
# ### Declare the data

# %%
x = 68      # number living at home, wales
n = 254     # number surveyed, wales
p0 = 0.25   # hypothesised proportion, national average

# %% [markdown]
# ### Analyse the data

# %%
describe.Proportion(x/n, sm.proportion_confint(count=x, nobs=n))

# %%
# run the test
zstat, pval = sm.proportions_ztest(
                    count=x,
                    nobs=n,
                    value=p0,
                    prop_var=p0,
                    alternative="larger")
summarise.PropTest(zstat, pval)
