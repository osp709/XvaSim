"""Cox-Ingersoll-Ross (CIR) hazard rate and credit model implementation.

This module implements the CIR default-intensity model as an object-oriented
:class:`~xvasim.models.base.CreditModel`.

Public API
----------
- :class:`CIRParams` — CIR credit parameters dataclass (re-exported).
- :class:`CIRHazardRateModel` — object-oriented CIR credit model.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import minimize

from ...jit import (
    cir_calibration_objective_kernel,
    cir_survival_probability_kernel,
)
from ..base import CreditModel
from ..registry import ModelRegistry


@dataclasses.dataclass(frozen=True)
class CIRParams:
    r"""Calibrated parameters for the Cox-Ingersoll-Ross hazard-rate model.

    The CIR process for the default intensity is:

    .. math::
        d\lambda_t = \kappa_{\text{ann}}
        (\theta_{\text{ann}} - \lambda_t)\,dt
        + \sigma_{\text{ann}}\sqrt{\lambda_t}\,dW_t

    Attributes:
        kappa_ann: Speed of mean reversion (annualised, e.g. 0.5).
        theta_ann: Long-term mean hazard rate (annualised decimal,
            e.g. 0.03 for 3 % per annum).
        sigma_ann: Volatility of the hazard-rate process (annualised,
            e.g. 0.10).
        lambda_0_ann: Initial hazard rate at time 0 (annualised decimal,
            e.g. 0.02).
    """

    kappa_ann: float
    theta_ann: float
    sigma_ann: float
    lambda_0_ann: float


@ModelRegistry.register("credit", "cir")
class CIRHazardRateModel(CreditModel):
    """CIR-based hazard rate and credit model."""

    def __init__(
        self,
        params: CIRParams | None = None,
        *,
        kappa_ann: float = 0.5,
        theta_ann: float = 0.03,
        sigma_ann: float = 0.10,
        lambda_0_ann: float = 0.02,
    ) -> None:
        """Initialize a CIR hazard rate model."""
        if params is not None:
            self._params = params
        else:
            self._params = CIRParams(
                kappa_ann=kappa_ann,
                theta_ann=theta_ann,
                sigma_ann=sigma_ann,
                lambda_0_ann=lambda_0_ann,
            )

    @property
    def model_name(self) -> str:
        """Returns 'cir'."""
        return "cir"

    @property
    def params(self) -> CIRParams:
        """The underlying :class:`CIRParams`."""
        return self._params

    @property
    def kappa_ann(self) -> float:
        """Speed of mean reversion."""
        return self._params.kappa_ann

    @property
    def theta_ann(self) -> float:
        """Long-term mean hazard rate."""
        return self._params.theta_ann

    @property
    def sigma_ann(self) -> float:
        """Volatility of hazard rate process."""
        return self._params.sigma_ann

    @property
    def lambda_0_ann(self) -> float:
        """Initial hazard rate."""
        return self._params.lambda_0_ann

    def survival_probability(self, tenors_yrs: np.ndarray) -> np.ndarray:
        r"""Compute survival probabilities using the CIR closed-form solution.

        .. math::
            P_{\text{surv}}(0, t) = A(t)\,e^{-B(t)\,\lambda_{0,\text{ann}}}
        """
        tenors_arr = np.asarray(tenors_yrs, dtype=np.float64)
        return cir_survival_probability_kernel(
            tenors_arr,
            self.kappa_ann,
            self.theta_ann,
            self.sigma_ann,
            self.lambda_0_ann,
        )

    @classmethod
    def calibrate_from_spreads(
        cls,
        credit_spreads_ann: np.ndarray,
        tenors_yrs: np.ndarray,
    ) -> CIRHazardRateModel:
        """Calibrate CIR model parameters to market credit spreads."""
        spreads = np.asarray(credit_spreads_ann, dtype=np.float64)
        tenors = np.asarray(tenors_yrs, dtype=np.float64)

        if (
            len(spreads) == 0
            or len(tenors) == 0
            or np.isnan(spreads).any()
            or np.isnan(tenors).any()
        ):
            msg = (
                "CIR calibration failed: input spreads or tenors contain NaN "
                "or are empty"
            )
            raise RuntimeError(msg)

        def objective(params_vec: np.ndarray) -> float:
            return cir_calibration_objective_kernel(params_vec, tenors, spreads)

        x0 = [
            0.1,
            float(np.mean(spreads)),
            0.05,
            float(spreads[0]),
        ]
        bounds = [
            (1e-4, 5.0),
            (1e-4, 2.0),
            (1e-4, 1.0),
            (1e-4, 2.0),
        ]

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        if not result.success or np.isnan(result.x).any():
            msg = f"CIR calibration failed: {result.message}"
            raise RuntimeError(msg)

        return cls(params=CIRParams(*result.x))
