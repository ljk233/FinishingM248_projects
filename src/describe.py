
"""
A module desgined to return a description of a sample of data as a
disctionary.
It is expected that the passed samples are an object from statsmodels
package.
"""

from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
from .classes import Sample, ZDiff, TDiff
from typing import Union
import pandas as pd


def parametric_sample(
        d1: DescrStatsW,
        a: float = 0.05,
        use_t: bool = True,
        as_df: bool = True,
        sample_label: str = "Sample"
) -> Union[Sample, pd.DataFrame]:
    """
    returns a description of a sample of data.

    params
    ======
    d1, DescrStatsW :
        sample as object of class DescrStatsW to describe
    a, float, default = 0.05
        significance level of the returned CI, as in (1-a)%
    use_t, bool, default = True
        Whether to return a t-interval or z-interval
    as_df, bool, default = True
        Whether to return the description as a pandas dataframe
    sample_label, str, default = "Sample"
        An optional descriptive label for the sample

    return
    ======
    Sample or pandas dataframe
    """
    size: int = int(d1.nobs)
    mean: float = d1.mean
    s: Sample = None
    df: pd.DataFrame() = pd.DataFrame(index=[sample_label])

    if use_t:
        s = Sample(size, mean, d1.tconfint_mean(alpha=a))
    else:
        s = Sample(size, mean, d1.zconfint_mean(alpha=a))

    if as_df:
        df["size"] = [s.size]
        df["mean"] = [s.mean]
        df["[0.025"] = [s.confint_mean[0]]
        df["0.975]"] = [s.confint_mean[1]]
        return df
    else:
        return s


def parametric_samples_difference(
        cm: CompareMeans,
        a: float = 0.05,
        use_t: bool = True
) -> Union[ZDiff, TDiff]:
    """
    params
    ======
    cm, CompareMeans :
        object of class CompareMeans to describe the difference
    a, float, default = 0.05
        significance level of the returned CI, as in (1-a)%
    use_t, bool, default = True
        Whether to return a t-interval or z-interval

    retun
    =====
    ZDiff or TDiff
    """

    diff: float = cm.d1.mean - cm.d2.mean

    if use_t:
        return TDiff(diff, cm.tconfint_diff(alpha=a))
    else:
        return ZDiff(diff, cm.zconfint_diff(alpha=a))
