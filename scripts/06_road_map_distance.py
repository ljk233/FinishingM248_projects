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
# # Linear modelling: Road distance and map distance
#
# ## Summary
#
# ### Data
#
# Road, `float` :
# road distance between two locations in Sheffield
#
# Map, `float` :
# map distance (ie, straight line) between two locations in Sheffield
#
# ### Method
#
# - Data modelled using simple linear regression through the origin
# - Data, model, and prediction intervals plotted
# - Assumptions of the distribution of the residuals checked with a 
#   residual plot, normal probability plot, and hypothesis test
#   $H_{0} : \beta = 0$
#
# ### Summary results
#
# ```python
# results{'beta': 1.289, 'tstat_beta': 42.803, 'pval_beta': 0.000, "dof": 11}
# ```
#
# ### Output
#
# <!--Add path to FinishingM248-->
#
# ### Reference
#
# m248.c.act5
#
# -----

# %% [markdown]
# ## Results

# %% [markdown]
# ### Setup the notebook

# %%
# import packages and modules
from scipy import stats
import statsmodels.api as sm
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
data = pd.read_csv("data\\distance.csv")

# %% [markdown]
# ### Preview and describe the data

# %%
data.head()

# %%
data.describe().T

# %% [markdown]
# ### Model the data

# %%
X = data["Map"]
y = data["Road"]

# %%
model = sm.OLS(y, X).fit()
model.params

# %% [markdown]
# ### Visualise the model

# %%
# plot data and model
f, ax = plt.subplots()
sns.scatterplot(x=X, y=y, label="data")
sns.lineplot(x=X, y=model.predict(), color="r", label="prediction")
os.chdir("figures")
plt.savefig("distance_fig1")
os.chdir("..")
plt.show()

# %% [markdown]
# ### Check assumptions of residual values

# %%
# residual plot
f, ax = plt.subplots()
sns.scatterplot(x=model.predict(), y=model.resid, ax=ax)
ax.set(xlabel="Fitted", ylabel="Residual", title="Residual plot")
ax.axhline(y=0, color="black", linestyle="--")
os.chdir("figures")
plt.savefig("distance_fig2")
os.chdir("..")
plt.show()

# %%
# probability plot
f, ax = plt.subplots()
stats.probplot(x=model.resid, plot=ax)
ax.set(ylabel="Residual")
os.chdir("figures")
plt.savefig("distance_fig3")
os.chdir("..")
plt.show()

# %% [markdown]
# #### Hypothesis test: $H_{0} : \beta = 0$

# %%
tstat, pval = model.tvalues[0], model.pvalues[0]

# %%
tstat, pval
