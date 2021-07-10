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
# #### Claim #1
#
# It is a common refrain that home teams enjoy an advantage over their
# away team opponents.
# In the first analysis, we test if there is such a thing as a *home
# team advantage* by testing if the proportion of games won by the home
# team in the  17/18, 18/19, and 20/21 EPL seasons was greater than a
# third.
#
# #### Claim #2
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
# - **Claim #1**
#   - Calculated the proportion of games won by the home team, and 95%
#     **z**-interval for the proportion
#   - Tested the hypothesis that the proportion of games won by the
#     home team is equal to 1/3
# - **Claim #2**
#   - Calcuated the proportion  won by the home team, and 95%
#     **z**-interval for the proportion for the combined 17/18
#     and 18/19 seasons (*pre-COVID*)
#   - Tested the hypothesis that the proportion of games won by the
#     home
#     team in the 17/18 and 18/19 seasons was equal to that of the
#     20/21 season
#
# ### Results
#
# - **Claim #1**
#   - Each sample comprised of 380 observations
#   - Season: 17/18
#     - `homewins=173`
#     - `Proportion(p_hat=0.455263, zconfint_prop=(0.405193, 0.505334))`
#     - `ResultSummary(zstat=5.042060, pval=0.000000)`
#   - Season: 18/19
#     - `homewins=181`
#     - `Proportion(p_hat=0.476316, zconfint_prop=(0.426100, 0.526531))`
#     - `ResultSummary(zstat=5.912631, pval=0.000000)`
#   - Season: 20/21
#     - `homewins=144`
#     - `Proportion(p_hat=0.378947, zconfint_prop=(0.330171, 0.427724))`
#     - `ResultSummary(zstat=1.886238, pval=0.029631)`
# - **Claim #2**
#   - Pre-COVID sample size = 760
#   - Description of Pre-COVID season
#     - `homewins=354`
#     - `Proportion(p_hat=0.455263, zconfint_prop=(0.405193, 0.505334))`
#   - Hypothesis test results
#     - `ResultSummary(zstat=2.810184, pval=0.004951)`
#
# ### Discussion
#
# - **Claim #1**
#   - For the pre-COVID seasons, very strong evidence against the
#     hypothesis that the proportion of games won by the home team is
#     equal to a third
#   - For the post-COVID season, only moderate evidence against the
#     hypothesis that the proportion of games won by the home team is
#     equal to a third
#   - All three samples suggest that the home team do enjoy an
#     advantage over the visitors
#   - However, post-COVID sample provides relatively weaker evidence
#     against the null hypothesis
#   - Suggests that there may have been a weakening of the home
#     advantage post-COVID
# - **Claim #2**
#   - Given **p**-value < 0.01, there is strong evidence against the
#     null hypothesis
#   - Data suggests that the proportion of games won by the home team
#     pre-COVID is not equal to the proportion post-COVID
#   - With a positive **z**-value, data suggests that the pre-COVID
#     seasons had a higher proportion of home wins compared to
#     post-COVID

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

# %% [markdown]
# #### Declare parameters

# %%
# get number of home wins per sample
w17: int = epl.query('season == 1718')["homewin"].sum()
w18: int = epl.query('season == 1819')["homewin"].sum()
w20: int = epl.query('season == 2021')["homewin"].sum()
w1718: int = epl.query('season == 1718 | season == 1819')["homewin"].sum()

# %%
# get number of games per season
n: int = int(epl.index.size/3)  # number of games per sample

# %%
# hypothesised proportion
p0: float = 1/3

# %% [markdown]
# #### Claim #1: Is there a home advantage?

# %%
# describe 17/18 season
w17, describe.Proportion(w17/n, sm.proportion_confint(w17, n))

# %%
# test 17/18 season
zstat, pval = sm.proportions_ztest(
                w17, n, alternative="larger", value=p0, prop_var=p0)
summarise.PropTest(zstat, pval)

# %%
# describe 18/19 season
w18, describe.Proportion(w18/n, sm.proportion_confint(w18, n))

# %%
# test 18/19 season
zstat, pval = sm.proportions_ztest(
                w18, n, alternative="larger", value=p0, prop_var=p0)
summarise.PropTest(zstat, pval)

# %%
# describe 20/21 season
w20, describe.Proportion(w20/n, sm.proportion_confint(w20, n))

# %%
# test 20/21 season
zstat, pval = sm.proportions_ztest(
                w20, n, alternative="larger", value=p0, prop_var=p0)
summarise.PropTest(zstat, pval)

# %% [markdown]
# #### Claim: Is the home advantage tthe same for pre and post-COVID?

# %%
# describe 17/18 and 18/19 seasons combined
w1718, describe.Proportion(w1718/(2*n), sm.proportion_confint(w1718, 2*n))

# %%
# test 17/18, 18/19 = 20/21
zstat, pval = sm.test_proportions_2indep(w1718, 2*n, w20, n)
summarise.PropTest(zstat, pval)
