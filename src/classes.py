
"""
Module docstring
"""


from dataclasses import dataclass


@dataclass
class Sample():
    """
    A dataclass to hold the parametric description of a sample modelled
    approximately normal.
    """
    size: int
    mean: float
    confint_mean: tuple


@dataclass
class ZDiff():
    """
    A dataclass to hold the parametric description of the Difference
    between two samples approximately normal.
    """
    diff: float
    zconfint_diff: tuple


@dataclass
class TDiff():
    """
    A dataclass to hold the parametric description of the Difference
    between two samples exactly normal.
    """
    diff: float
    tconfint_diff: tuple
