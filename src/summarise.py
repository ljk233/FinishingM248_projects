
"""
A module of dataclasses used for describing the results of hypothesis
tests.
They are replacements for NamedTuples, and only used for aesthetic
purposes.
The classes return floats to 6dp when __repr__ is invoked.
"""

from dataclasses import dataclass


@dataclass
class TTest():
    """
    A dataclass to hold the results of a t-test
    """
    tstat: float
    pval: float
    dof: int

    def __repr__(self) -> str:
        return (
            f"ResultSummary("
            f"tstat={self.tstat:.6f}"
            f", pval={self.pval:.6f}"
            f", dof={int(self.dof)})")


@dataclass
class ZTest():
    """
    A dataclass to hold the results of a z-test
    """
    zstat: float
    pval: float

    def __repr__(self) -> str:
        return (
            f"ResultSummary("
            f"zstat={self.zstat:.6f}"
            f", pval={self.pval:.6f})")


@dataclass
class PropTest():
    """
    A dataclass to hold the results of a test of a proportion.
    """
    zstat: float
    pval: float

    def __repr__(self) -> str:
        return (
            f"ResultSummary("
            f"zstat={self.zstat:.6f}"
            f", pval={self.pval:.6f})")


@dataclass
class ChisqGoodnessOfFit():
    """
    A dataclass to hold the results of a chi-square gof test
    """
    chisq: float
    pval: float
    dof: int

    def __repr__(self) -> str:
        return (
            f"ResultSummary("
            f"chisq={self.chisq:.6f}"
            f", pval={self.pval:.6f}"
            f", dof={self.dof})")
