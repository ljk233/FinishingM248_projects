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
# # Using the skull size of Etruscan and Italian males to analyse if there was an ancestral link  # noqa: E501
#
# ## Notes
#
# ### Question of interest
#
# Do the ancient Etruscan people of Italy share a common ancestral link
# with those of native Italians?
#
# ### Data
#
# - Data consists of 154 observations of male skull breadths
#   - Modern Italian males used as substitute for ancient Italian males
# - Data fields:
#   - **type** `str` : origin of skull, Etruscan or Italian
#   - **size** `float` : skull breadth (mm)
#
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
# ### Results
#
# - The data returned the descriptive column `type` as an `object`,
#   rather than a `str`
#   - This did not affect the analysis, so not remedial action was taken
# - None of the visualisations show the assumption that both samples are
#   normally distributed is inappropriate
#   - Frequency histogram shows both samples are unimodal and symmetric
#   - Probability plots show the data to closely follow a fitted straight
# - Description of samples:
#   - `Etruscan(size=84, mean=143.8, tconfint_mean=(142.5, 145.1))`
#   - `Italian(size=70, mean=132.4, tconfint_mean=(131.1, 133.8))`
# - Description of difference between the samples
#   - `SampleDiff(mean_diff=11.3, tconfint_diff=(9.5, 13.2))`
# - **t**-test result
#   - `ResultSummary(tstat=11.924, pval=0.000000, dof=152)`
#
# ### Discussion
#
# - Null hypothesis is rejected with **p**-value < 0.000001
# - Very strong evidence that the mean Etruscan skull breadth is not
#   equal to that of the mean Italian skull breadth
# - Given **t** > 0, we conclude that there is evidence that the
#   Etruscans had larger skulls on average than Italians
# - Result suggests there is not a common ancestral link between the two
#   ancient peoples
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
# initialise DescrStatsW objects
dsw_etr: sm.DescrStatsW = ttest.d1
dsw_ita: sm.DescrStatsW = ttest.d2

# %%
# check for common population variance. Expect < 3
max(dsw_etr.var / dsw_ita.var, dsw_ita.var / dsw_etr.var) < 3


# %%
describe.TSample(
    "Etruscans", dsw_etr.nobs, dsw_etr.mean, dsw_etr.tconfint_mean())

# %%
describe.TSample(
    "Italians", dsw_ita.nobs, dsw_ita.mean, dsw_ita.tconfint_mean())

# %%
describe.TDiff(
    dsw_etr.mean - dsw_ita.mean, ttest.tconfint_diff())

# %%
# run the test
tstat, pval, dof = ttest.ttest_ind()
summarise.TTest(tstat, pval, dof)
