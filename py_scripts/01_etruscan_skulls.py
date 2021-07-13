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
# # Were the Etruscan people native to Italy?
#
# ## Notes
#
# ### Data
#
# - Data consists of 154 observations of male skull breadths
#   - Modern Italian males used as substitute for ancient Italian males
# - Data fields:
#   - **type** `str` : origin of skull, Etruscan or Italian
#   - **size** `float` : skull breadth (mm)
#
# ### Method
#
# - Given data is biological, reasonable to suppose that it can modelled
#   normally
#   - Normality of samples checked using a frequency histogram and normal
#     probability plot
# - Mean and 95% **t**-interval returned both samples
# - Mean difference and 95% **t**-interval of the mean difference
#   returned
# - Checked assumption of common population variance
# - Performed **t**-test: mean skull breadth of the Etruscan skulls is
#   equal to that of the Italian skulls
#
# ### Reference
#
# m248.b.act22
#
# -----

# %% [markdown]
# ## Full Results

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
skulls: pd.DataFrame = load.Data.get("skulls")

# %%
# preview data
skulls.head()

# %%
# check dtypes
skulls.dtypes

# %%
# get samples as series
etr: pd.Series = skulls.query('type == "Etruscan"')["size"]
ita: pd.Series = skulls.query('type == "Italian"')["size"]

# %% [markdown]
# ### Visualise the data

# %%
# frequency histograms
g = sns.FacetGrid(skulls, col="type")
g.map_dataframe(sns.histplot, x="size", bins=10)
g.set_axis_labels("Size", "Count")
plt.show()

# %%
# probability plots
f, axs = plt.subplots(ncols=2, sharey=True)
stats.probplot(x=etr, plot=axs[0])
stats.probplot(x=ita, plot=axs[1])
axs[0].set(title="Probability plot = Etruscans")
axs[1].set(title="Probability plot = Italians")
plt.show()

# %% [markdown]
# ### Analyse the data

# %%
# initialise CompareMeans object
ttest: sm.CompareMeans = sm.CompareMeans.from_data(data1=etr, data2=ita)

# %%
# check for common population variance. Expect < 3
max(ttest.d1.var / ttest.d2.var, ttest.d2.var / ttest.d1.var) < 3


# %%
describe.TSample(
    "Etruscans", ttest.d1.nobs, ttest.d1.mean, ttest.d1.tconfint_mean())


# %%
describe.TSample(
    "Italians", ttest.d2.nobs, ttest.d2.mean, ttest.d2.tconfint_mean())

# %%
describe.TDiff(
    ttest.d1.mean - ttest.d2.mean, ttest.tconfint_diff())

# %%
# run the test
tstat, pval, dof = ttest.ttest_ind()
summarise.TTest(tstat, pval, dof)
