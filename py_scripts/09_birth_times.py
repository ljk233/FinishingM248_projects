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
# # Do more births occur in night time hours?
#
# ## Notes
#
# ### Question of interest
#
# It is often said that more births occur at night-time hours.
# Is this old-wives tale supported by the evidence?
#
# ### Data
#
# - Birth times in England, 2005–14
# - Data consists of 5 093 615 observation
# - Data fields:
#   - **time-** `str` : location of test centre
#
# ### Method
#
# ### Results
#
# ### Discussion
#
# ### Reference
#
# m248.b.act23
#
# -----

# %% [markdown]
# ## Results
#
# ### Setup the notebook

# %%
# import packages and modules
import statsmodels.stats.proportion as sm
import pandas as pd
import sys

# %%
# import custom modules not in root
sys.path[0] = "..\\"  # update path
from src import load, describe  # noqa: E402

# %% [markdown]
# ### Import the data

# %%
# get data
birth_times: pd.DataFrame = load.Data.get("birth_times")
birth_times

# %% [markdown]
# ### Transform the data

# %%
# add day, night labels
wh: list = [0, 0, 0, 1, 1, 2, 0, 0]
birth_times["working_time"] = wh

# %% [markdown]
# ### Analyse the data

# %%
# number of births
n: int = 5093615

# %%
# estimate births between 3pm-5pm
work: float = round(11.95 * 2/3, 2)
non_work: float = round(11.95 - work, 2)

# %%
x: float = (birth_times.query('working_time == 1')["percentage"].sum()/100
            + work/100)
x = int(x*n)

# %%
describe.Proportion(x/n, sm.proportion_confint(x, n))
