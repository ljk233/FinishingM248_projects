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
# It is a common refrain that home teams enjoy an advantage over their
# away team opponents.
# In this analysis, we test if there is such a thing as a *home
# team advantage* by testing if the proportion of games won by the home
# team in the 17/18, 18/19, and 20/21 EPL seasons was greater than a
# third, the expected result if there was no home advantage.
#
# ### Data
#
# - Data comprised of the full results for the 17/18, 18/19, and 20/21
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
# - Identified the result of the game (home win or not)
# - Each sample modelled by a normal approximation to the binomial
#   - Justified by the sample sizes (`size=380`)
# - Calculated the proportion of games won by the home team, and 95%
#   **z**-interval for the proportion
# - Tested the hypothesis that the proportion of games won by the
#   home team is equal to 1/3

# %% [markdown]
# ## Full results

# %% [markdown]
# ### Setup the notebook

# %%
# import packages and modules
from __future__ import annotations
import statsmodels.stats.proportion as sm
import pandas as pd
import sys

# %%
# import custom modules not in root
sys.path[0] = "..\\"  # update path
from src import load  # noqa: E402


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
# identify if home win or not
epl["goaldiff"] = epl["homegoals"] - epl["awaygoals"]
epl["homewin"] = epl["goaldiff"].apply(is_home_win)

# %% [markdown]
# ### Analyse the data

# %%
# get number of games per season
n: int = int(epl.index.size/3)

# %%
sample_descr: pd.DataFrame = pd.DataFrame(index=["17/18", "18/19", "20/21"])
sample_tests: pd.DataFrame = pd.DataFrame(index=["17/18", "18/19", "20/21"])

# %%
# get number of home wins per sample
w17: int = epl.query('season == 1718')["homewin"].sum()
w18: int = epl.query('season == 1819')["homewin"].sum()
w20: int = epl.query('season == 2021')["homewin"].sum()
homewins: list[int] = [w17, w18, w20]
sample_descr["homewins"] = homewins

# %%
# get point, interval estimates for each sample
prop_ests: list[float] = list()
lower_props: list[float] = list()
upper_props: list[float] = list()

for res in homewins:
    prop_ests.append(res/n)
    ci: tuple[float] = sm.proportion_confint(res, n)
    lower_props.append(ci[0])
    upper_props.append(ci[1])

sample_descr["prop_est"] = prop_ests
sample_descr["[0.025"] = lower_props
sample_descr["0.975]"] = upper_props
sample_descr

# %%
# get test results
p0: float = 1/3
zstats: list[float] = list()
pvals: list[float] = list()

for res in homewins:
    zstat, pval = sm.proportions_ztest(
                    res, n, value=p0, alternative="larger", prop_var=p0)
    zstats.append(zstat)
    pvals.append(pval)

sample_tests["zstats"] = zstats
sample_tests["pvals"] = pvals
sample_tests
