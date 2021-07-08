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
# # Linear modelling: Road distance and map distance
#
# ## Summary
#
# ### Question of interest
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
# ### Discussion
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
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
import sqlite3
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

# %% [markdown]
# ### Import the data

# %%
conn = sqlite3.connect("data\\source.db3")
data = pd.read_sql_query("SELECT * FROM road_map_distance", conn)

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
model.get_prediction()

# %%
# get prediction interval for mean response
predstd, pred_iv_l, pred_iv_u  = sm_pred(model)

# %%
# plot data and model
plt.subplots(figsize=(10,6))
sns.scatterplot(x=X, y=y, label="data")
sns.lineplot(x=X, y=model.predict(), color="g", label="prediction")
# add predint mean response
sns.lineplot(x=X, y=pred_iv_u, color="r", linestyle="--")
sns.lineplot(x=X, y=pred_iv_l, color="r", linestyle="--")
os.chdir("figures")
plt.savefig("distance_fig1")
os.chdir("..")
plt.show()

# %% [markdown]
# ### Check assumptions of residual values

# %%
# residual plot and probability plot
f, axs = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
sns.scatterplot(x=model.predict(), y=model.resid, ax=axs[0])
axs[0].set(xlabel="Fitted", ylabel="Residual", title="Residual plot")
axs[0].axhline(y=0, color="black", linestyle="--")
stats.probplot(x=model.resid, plot=axs[1])
axs[1].set(ylabel="Residual")
os.chdir("figures")
plt.savefig("distance_fig2")
os.chdir("..")
plt.show()

# %%
# hypoth test: beta = 0
tstat, pval = model.tvalues[0], model.pvalues[0]

# %%
tstat, pval
