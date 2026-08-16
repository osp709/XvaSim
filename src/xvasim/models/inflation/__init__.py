"""Inflation models package for XvaSim.

This package provides stochastic models for Consumer Price Index (CPI) dynamics,
real & nominal interest rates, and inflation derivative pricing.

Public API
----------
- :class:`JarrowYildirimModel` — modular two-economy Jarrow-Yildirim model.
- :class:`JarrowYildirimParams` — parameters for Jarrow-Yildirim model.
- :class:`BlackInflationModel` — log-normal forward CPI model.
- :class:`BlackInflationParams` — parameters for Black inflation model.
- :class:`InflationSimulationResult` — container for simulation trajectories.
"""

from .black_inflation import BlackInflationModel, BlackInflationParams
from .jarrow_yildirim import (
    InflationSimulationResult,
    JarrowYildirimModel,
    JarrowYildirimParams,
)

__all__ = [
    "BlackInflationModel",
    "BlackInflationParams",
    "InflationSimulationResult",
    "JarrowYildirimModel",
    "JarrowYildirimParams",
]
