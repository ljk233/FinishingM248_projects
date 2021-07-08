
"""
A module of dataclasses used for describing samples during analysis.
They are replacements for NamedTuples, and only used for aesthetic
purposes.
The classes return floats to 6dp.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Sample():
    """
    A dataclass to hold the parametric description of a sample.
    """
    label: str
    size: int
    mean: float


@dataclass
class ZSample(Sample):
    """
    A dataclass to hold the parametric description of a sample modelled
    approximately normal.
    """
    zconfint_mean: Tuple[float]

    def __repr__(self) -> str:
        return (
            f"{self.label}("
            f"size={int(self.size)}"
            f", mean={self.mean:.6f}"
            f", zconfint_mean=({self.zconfint_mean[0]:.6f}"
            f", {self.zconfint_mean[1]:.6f}))"
        )


@dataclass
class TSample(Sample):
    """
    A dataclass to hold the parametric description of a sample modelled
    exactly normal.
    """
    tconfint_mean: Tuple[float]

    def __repr__(self) -> str:
        return (
            f"{self.label}("
            f"size={int(self.size)}"
            f", mean={self.mean:.6f}"
            f", tconfint_mean=({self.tconfint_mean[0]:.6f}"
            f", {self.tconfint_mean[1]:.6f}))"
        )


@dataclass
class Diff:
    """
    A dataclass to hold the description of the difference
    between two samples parametric samples
    """
    mean_diff: float


@dataclass
class ZDiff(Diff):
    """
    A dataclass to hold the description of the difference
    between two samples parametric samples modelled approximately
    normally
    """
    zconfint_diff: Tuple[float]

    def __repr__(self) -> str:
        return (
            f"Difference("
            f"mean_diff={self.mean_diff:.6f}"
            f", zconfint_diff=({self.zconfint_diff[0]:.6f}"
            f", {self.zconfint_diff[1]:.6f}))"
        )


@dataclass
class TDiff(Diff):
    """
    A dataclass to hold the description of the difference
    between two samples parametric samples modelled normally
    """
    tconfint_diff: Tuple[float]

    def __repr__(self) -> str:
        return (
            f"Difference("
            f"mean_diff={self.mean_diff:.6f}"
            f", tconfint_diff=({self.tconfint_diff[0]:.6f}"
            f", {self.tconfint_diff[1]:.6f}))"
        )


@dataclass
class Proportion():
    """
    A dataclass to hold the description of a proportion modelled by
    a normal approximation of the binomial
    """
    p_hat: float
    zconfint_prop: Tuple[float]

    def __repr__(self) -> str:
        return (
            f"Proportion("
            f"p_hat={self.p_hat:.6f}"
            f", zconfint_prop=({self.zconfint_prop[0]:.6f}"
            f", {self.zconfint_prop[1]:.6f}))"
        )
