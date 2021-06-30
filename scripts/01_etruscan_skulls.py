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
# source: `data/skulls.csv`
#
# fields:
#
# - Etruscan, `float` : "breadth of etruscan skulls (mm)"
# - Italian, `float` : "breadth of italian skulls (mm)"
#
#
# ### Method
#
# - Data modelled using a normal distribution.
# - Normality Checked using a histogram and normal probability plot.
# - Two sample, two-tailed **t**-test used to test the hypothesis that
#   the mean skull breadth of Etruscan skulls is equal to that of Italian
#   skulls.
#
# ### Summary results
#
# ```python
# res_descr{
#     'etr_size': 84, 'etr_mean': 143.774, 'etr_confint': (142.478, 145.069),
#     'ita_size': 70, 'ita_mean': 132.443, 'ita_confint': (131.072, 133.814),
#     'diff_mean': 11.331, 'diff_confint': (9.454, 13.208)}
# ```
#
# ```python
# res_test{
#     'tstat': 11.925, 'pval': 0.000, 'dof': 152.0}
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
from scipy import stats
import statsmodels.stats.weightstats as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set_theme()

# %%
# change wkdir
os.chdir("..\\")

# %%
# import the data
data = pd.read_csv("data\\skulls.csv")

# %% [markdown]
# ### Describe the data

# %%
data.describe().T

# %% [markdown]
# ### Declare local variables

# %%
etr = data["Etruscans"]
ita = data["Italians"].dropna()
ttest = sm.CompareMeans.from_data(data1=etr, data2=ita)
res_descr = dict()
res_test = dict()

# %% [markdown]
# ### Gather descriptive statistics

# %%
# describe Etruscan sample
res_descr["etr_size"] = etr.size
res_descr["etr_mean"] = etr.mean()
res_descr["etr_confint"] = ttest.d1.tconfint_mean()

# %%
# describe Italian sample
res_descr["ita_size"] = ita.size
res_descr["ita_mean"] = ita.mean()
res_descr["ita_confint"] = ttest.d2.tconfint_mean()

# %%
# describe differences between the samples
res_descr["diff_mean"] = etr.mean() - ita.mean()
res_descr["diff_confint"] = ttest.tconfint_diff()

# %%
# output descriptive stats
res_descr

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
# probability plot of samples
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
# ### Run the hypothesis test

# %%
# check sample variances < 3
max(etr.var() / ita.var(), ita.var() / etr.var())

# %%
# run the test
res_test["tstat"], res_test["pval"], res_test["dof"] = ttest.ttest_ind()

# %%
# output results
res_test
