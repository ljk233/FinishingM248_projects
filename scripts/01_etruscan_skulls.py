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
# # Comparing skull sizes of Etruscans and Italians
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
# - Data consists of 154 observations of skull breadth
# - Modern male Italians used as substitute for ancient Italians
# - Fields:
#   - type, `str` : origin of skull, Etruscan or Italian
#   - size, `float` : skull breadth (mm)
#
#
# ### Method
#
# - Given data is biological, reasonable to suppose that it can modelled
#   normally
#   - Normality of samples checked using a frequency histogram and normal
#     probability plot
# - Mean and 95% **t**-interval returned both samples
# - Checked assumption of common population variance
# - Performed **t**-test: mean skull breadth of the Etruscan skulls is
#   equal to that of the Italian skulls
#
# ### Results
#
# - Frequency histogram shows both samples are unimodal and symmetric,
#   as we would expect for normally distributed data
# - Probability plots show the data to closely follow a fitted straight
#   line. Neither show the assumption of normality is inappropriate.
# - Description of Etruscan skull sample:
#   - size=84
#   - mean=143.7738
#   - confint_mean=(142.4781, 145.0695)
# - Description of Italian skull sample:
#   - size=70
#   - mean=132.4429
#   - confint_mean=(132.4429, 133.8139)
# - **t**-test results
#   - **t** = 11.925, with 152 degrees of freedom
#   - **p**-value < 0.0001
#
# ### Discussion
#
# - Null hypothesis is rejected with **p**-value < 0.0001
# - Very strong evidence that the mean Etruscan skull breadth is not
#   equal to that of the mean Italian skull breadth
# - Given **t** > 0, we conclude that there is evidence that the Etruscans
#   had larger skulls on average than Italians
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
# set seaborn theme
sns.set_theme()

# %%
# import custom modules not in root
sys.path[0] = "..\\"  # update path
from src import load, describe  # noqa: E402

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

# %% [markdown]
# Column `Type` has come through as `object`, rather than `str`.
# This will not affect the analysis, so we do not cast the column to `str`.

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
ttest = sm.CompareMeans.from_data(data1=etr, data2=ita)

# %%
# check for common population variance. Expect < 3
max(etr.var() / ita.var(), etr.var() / ita.var()) < 3


# %%
describe.parametric_sample(ttest.d1, sample_label="Etruscans")

# %%
describe.parametric_sample(ttest.d2, sample_label="Italians")

# %%
# run the test
tstat, pval, dof = ttest.ttest_ind()
tstat, pval, dof
