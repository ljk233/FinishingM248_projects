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
# # Analysing the home advantage in the English Premier Leage
#
# ## Notes
#
# ### Question of interest
#
# The coronavirus panademic meant that all game in the English Premier
# League we played in front of empty stadiums.
# It is claimed that this resulted in a diminishing of the home team
# advantage.
# In the second analysis, we test if the proportion of home team wins
# in the 17/18 and 18/19 seasons (*pre-COVID*) was equal to that of the
# 20/21 season (*post-COVID*).
#
# ### Data
#
# - Data comprised of the full results for the 18/18, 18/19, and 20/21
#   seasons
# - Data fields
#   - **game_id** `int` : game number, unique per season
#   - **season** `int` : the season of the game, one of
#     [1718, 1819, 2021]
#   - **homegoals** `int` : number of goals scored by the home team
#   - **awaygoals** `int` : number of goals scored by the away team
#
# ### Method
#
# - Data comprised on 1140 game results
# - Identified the result of the game, home win or not
#   - Calculated the goal difference (homegoals - awaygoals)
#   - If goal difference > 0, then home win, else not home win
# - Each sample modelled by a normal approximation to the binomial
#   - Justified by the sample sizes (size == 380)
# - Calcuated the proportion  won by the home team, and 95%
#   **z**-interval for the proportion for the combined 17/18
#   and 18/19 seasons (*pre-COVID*) and the 20/21 season (*post-COVID*)
# - Tested the hypothesis that the proportion of games won by the
#   home team *pre-COVID* was equal to that *post-COVID*

# %% [markdown]
# ## Full results

# %% [markdown]
# ### Setup the notebook

# %%
# import packages and modules
import statsmodels.stats.proportion as sm
import pandas as pd
import sys

# %%
# import custom modules not in root
sys.path[0] = "..\\"  # update path
from src import load, describe, summarise  # noqa: E402


# %%
# declare functions
def is_home_win(x: int) -> int:
    if x > 0:
        return 1
    else:
        return 0


# %% [markdown]
# ### Import the data

# %%
# get data
epl: pd.DataFrame = load.Data.get("epl_results")

# %%
# preview data
epl.head()

# %%
# check dtypes
epl.dtypes

# %% [markdown]
# ### Transform the data

# %%
# get goal difference
epl["goaldiff"] = epl["homegoals"] - epl["awaygoals"]

# %%
# identify if home win or not
epl["homewin"] = epl["goaldiff"].apply(is_home_win)

# %% [markdown]
# ### Analyse the data

# %%
# get number of games per season
n: int = int(epl.index.size/3)

# %%
# get number of home wins per sample
w_pre: int = epl.query('season == 1718 | season == 1819')["homewin"].sum()
w_post: int = epl.query('season == 2021')["homewin"].sum()

# %%
describe.Proportion(w_pre/(2*n), sm.proportion_confint(w_pre, 2*n))

# %%
describe.Proportion(w_post/(n), sm.proportion_confint(w_post, n))

# %%
# get test results
zstat, pval = sm.test_proportions_2indep(w_pre, 2*n, w_post, n)
summarise.PropTest(zstat, pval)
