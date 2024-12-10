"""
IndicA is a A library for **In**tegrated **Di**agnosti**C** **A**nalysis
of magnetic-confined fusion devices
"""

from .abstractio import BaseIO
from .data import InDiCAArrayAccessor
from .data import InDiCADatasetAccessor
from .equilibrium import Equilibrium
from .plasma import Plasma
from .plasma import PlasmaProfiler

__all__ = [
    "InDiCAArrayAccessor",
    "InDiCADatasetAccessor",
    "BaseIO",
    "Equilibrium",
    "Plasma",
    "PlasmaProfiler",
]
