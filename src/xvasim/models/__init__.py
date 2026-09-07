"""Modular stochastic modeling framework for XvaSim.

This package provides extensible base classes, concrete implementations, and
a central registry for financial stochastic models across interest rates,
foreign exchange, credit, and inflation risk factors.

Public API
----------
- Base abstractions:
  - :class:`RiskFactorType`
  - :class:`StochasticModel`
  - :class:`InterestRateModel`
  - :class:`CreditModel`
  - :class:`FXModel`
  - :class:`InflationModel`
- Registry & Factory:
  - :class:`ModelRegistry`
  - :func:`create_ir_model`
  - :func:`create_credit_model`
  - :func:`create_fx_model`
  - :func:`create_inflation_model`
  - :func:`list_available_models`
- Interest rate models:
  - :class:`LGMModel`, :class:`LGMParams`
  - :class:`HullWhite1FModel`, :class:`HullWhite1FParams`
  - :class:`VasicekModel`, :class:`VasicekParams`
  - :class:`CIRInterestRateModel`, :class:`CIRInterestRateParams`
- Credit models:
  - :class:`CIRHazardRateModel`, :class:`CIRHazardRateParams`
- FX models:
  - :class:`TwoCurrencyFXModel`
  - :class:`GarmanKohlhagenFXModel`, :class:`GarmanKohlhagenFXParams`
  - :class:`HestonFXModel`, :class:`HestonFXParams`
- Inflation models:
  - :class:`JarrowYildirimModel`, :class:`JarrowYildirimParams`
  - :class:`BlackInflationModel`, :class:`BlackInflationParams`
  - :class:`InflationSimulationResult`
"""

from .base import (
    CreditModel,
    FXModel,
    InflationModel,
    InterestRateModel,
    RiskFactorType,
    StochasticModel,
)
from .credit import (
    CIRHazardRateModel,
    CIRHazardRateParams,
)
from .fx import (
    GarmanKohlhagenFXModel,
    GarmanKohlhagenFXParams,
    HestonFXModel,
    HestonFXParams,
    TwoCurrencyFXModel,
    TwoCurrencyFXParams,
)
from .inflation import (
    BlackInflationModel,
    BlackInflationParams,
    InflationSimulationResult,
    JarrowYildirimModel,
    JarrowYildirimParams,
)
from .ir import (
    CIRInterestRateModel,
    CIRInterestRateParams,
    HullWhite1FModel,
    HullWhite1FParams,
    LGMModel,
    LGMParams,
    VasicekModel,
    VasicekParams,
)
from .registry import (
    ModelRegistry,
    create_credit_model,
    create_fx_model,
    create_inflation_model,
    create_ir_model,
    list_available_models,
)

__all__ = [
    "BlackInflationModel",
    "BlackInflationParams",
    "CIRHazardRateModel",
    "CIRHazardRateParams",
    "CIRInterestRateModel",
    "CIRInterestRateParams",
    "CreditModel",
    "FXModel",
    "GarmanKohlhagenFXModel",
    "GarmanKohlhagenFXParams",
    "HestonFXModel",
    "HestonFXParams",
    "HullWhite1FModel",
    "HullWhite1FParams",
    "InflationModel",
    "InflationSimulationResult",
    "InterestRateModel",
    "JarrowYildirimModel",
    "JarrowYildirimParams",
    "LGMModel",
    "LGMParams",
    "ModelRegistry",
    "RiskFactorType",
    "StochasticModel",
    "TwoCurrencyFXModel",
    "TwoCurrencyFXParams",
    "VasicekModel",
    "VasicekParams",
    "create_credit_model",
    "create_fx_model",
    "create_inflation_model",
    "create_ir_model",
    "list_available_models",
]
