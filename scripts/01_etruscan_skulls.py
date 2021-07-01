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
# ## Summary
#
# ### Data
#
# Etruscan, `float` :
# breadth of etruscan skulls (mm)
#
# Italian, `float` :
# breadth of italian skulls (mm)
#
#
# ### Method
#
# - Data modelled using a normal distribution
# - Normality Checked using a histogram and normal probability plot
# - Assumption of common population variance checked
# - Mean and 95% **t**-interval returned for samples
# - Two sample, two-tailed **t**-test used to test the hypothesis that
#   the mean skull breadth of Etruscan skulls is equal to that of
#   Italian skulls
#
# ### Summary results
#
# ```python
# test_results{'tstat': 11.925, 'pval': 0.000, 'dof': 152.0}
# ```
#
# ### Output
#
# <!--Add path to FinishingM248-->
#
# ### Reference
#
# m248.b.act22
#
# -----

# %% [markdown]
# ## Results

# %% [markdown]
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
data = pd.read_csv("data\\skulls.csv")
etr = data["Etruscans"]
ita = data["Italians"].dropna()

# %% [markdown]
# ### Preview and describe the data

# %%
data.head()

# %%
data.describe().T

# %% [markdown]
# ### Visualise the data

# %%
# frequency histograms
mdata = data.melt()
mdata.dropna(inplace=True)
g = sns.FacetGrid(mdata, col="variable")
g.map_dataframe(sns.histplot, x="value", bins=10)
g.set_axis_labels("Size", "Count")
os.chdir("figures")
plt.savefig("skulls_fig1")
os.chdir("..")
plt.show()

# %%
# probability plot for each sample
f, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
stats.probplot(x=etr, plot=ax1)
stats.probplot(x=ita, plot=ax2)
ax1.set(title="Probability plot = Etruscans")
ax2.set(title="Probability plot = Italians")
os.chdir("figures")
plt.savefig("skulls_fig2")
os.chdir("..")
plt.show()

# %% [markdown]
# ### Check for common population variance

# %%
max(etr.var() / ita.var(), ita.var() / etr.var())

# %% [markdown]
# ### Run the hypothesis test

# %%
ttest = sm.CompareMeans.from_data(data1=etr, data2=ita)

# %% [markdown]
# #### Get confidence intervals

# %%
# Etruscan
ttest.d1.tconfint_mean()

# %%
# Italian
ttest.d2.tconfint_mean()

# %% [markdown]
# #### Describe the difference

# %%
# get mean difference
etr.mean() - ita.mean()

# %%
# get confint of mean difference
ttest.tconfint_diff()

# %% [markdown]
# #### Run the test

# %%
tstat, pval, dof = ttest.ttest_ind()

# %%
tstat, pval, dof
