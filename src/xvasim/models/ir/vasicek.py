r"""Vasicek short-rate interest rate model implementation.

This module implements the classic Vasicek (1977) short-rate model:

.. math::
    dr(t) = \kappa_{\text{ann}}(\theta_{\text{ann}} - r(t))\,dt
    + \sigma_{\text{ann}}\,dW(t)

Public API
----------
- :class:`VasicekParams` — parameter dataclass for the Vasicek model.
- :class:`VasicekModel` — object-oriented Vasicek interest rate model.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ...jit import discount_path_kernel, vasicek_simulate_paths_kernel
from ...qmc import RandomSequenceType, generate_brownian_increments
from ..base import InterestRateModel
from ..registry import ModelRegistry


@dataclasses.dataclass(frozen=True)
class VasicekParams:
    """Parameters for the Vasicek short-rate model.

    Attributes:
        kappa_ann: Speed of mean reversion (annualised, e.g. 0.15).
        theta_ann: Long-term mean short rate (annualised decimal, e.g. 0.03).
        sigma_ann: Short-rate volatility (annualised decimal, e.g. 0.015).
        r0_ann: Initial short rate at t=0 (annualised decimal, e.g. 0.025).
        discount_curve_yrs: Optional tenors array. If None, derived from model.
        discount_factors: Optional discount factors array.
    """

    kappa_ann: float
    theta_ann: float
    sigma_ann: float
    r0_ann: float
    discount_curve_yrs: np.ndarray | None = None
    discount_factors: np.ndarray | None = None


@ModelRegistry.register("interest_rate", "vasicek")
class VasicekModel(InterestRateModel):
    """Vasicek (1977) short-rate interest rate model."""

    def __init__(
        self,
        params: VasicekParams | None = None,
        *,
        kappa_ann: float = 0.15,
        theta_ann: float = 0.03,
        sigma_ann: float = 0.015,
        r0_ann: float = 0.025,
        discount_curve_yrs: np.ndarray | None = None,
        discount_factors: np.ndarray | None = None,
    ) -> None:
        """Initialize a Vasicek interest rate model."""
        if params is not None:
            self._params = params
        else:
            self._params = VasicekParams(
                kappa_ann=kappa_ann,
                theta_ann=theta_ann,
                sigma_ann=sigma_ann,
                r0_ann=r0_ann,
                discount_curve_yrs=discount_curve_yrs,
                discount_factors=discount_factors,
            )

        if (
            self._params.discount_curve_yrs is None
            or self._params.discount_factors is None
        ):
            # Build analytical model-implied term structure
            grid = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
            dfs = self._analytical_bond_price(0.0, grid, self._params.r0_ann)
            self._curve_yrs = grid
            self._curve_dfs = dfs
        else:
            self._curve_yrs = np.asarray(
                self._params.discount_curve_yrs, dtype=np.float64
            )
            self._curve_dfs = np.asarray(
                self._params.discount_factors, dtype=np.float64
            )

    @property
    def model_name(self) -> str:
        """Returns 'vasicek'."""
        return "vasicek"

    @property
    def params(self) -> VasicekParams:
        """The underlying :class:`VasicekParams`."""
        return self._params

    @property
    def kappa_ann(self) -> float:
        """Speed of mean reversion."""
        return self._params.kappa_ann

    @property
    def theta_ann(self) -> float:
        """Long-term mean rate."""
        return self._params.theta_ann

    @property
    def sigma_ann(self) -> float:
        """Short-rate volatility."""
        return self._params.sigma_ann

    @property
    def r0_ann(self) -> float:
        """Initial short rate."""
        return self._params.r0_ann

    @property
    def discount_curve_yrs(self) -> np.ndarray:
        """Discount curve tenors in years."""
        return self._curve_yrs

    @property
    def discount_factors(self) -> np.ndarray:
        """Discount factors along the curve."""
        return self._curve_dfs

    def _analytical_bond_price(
        self,
        t: float,
        maturity_yrs: np.ndarray | float,
        r_t: np.ndarray | float,
    ) -> np.ndarray:
        """Compute analytical Vasicek bond price P(t, T)."""
        tau = np.maximum(np.asarray(maturity_yrs, dtype=np.float64) - t, 0.0)
        kappa = self.kappa_ann
        theta = self.theta_ann
        sigma = self.sigma_ann

        if abs(kappa) < 1e-12:
            b_tau = tau
            a_tau = np.exp(-theta * tau + (sigma**2 / 6.0) * (tau**3))
        else:
            b_tau = (1.0 - np.exp(-kappa * tau)) / kappa
            exponent = (theta - (sigma**2) / (2.0 * kappa**2)) * (b_tau - tau) - (
                (sigma**2) / (4.0 * kappa)
            ) * (b_tau**2)
            a_tau = np.exp(exponent)

        return np.asarray(a_tau * np.exp(-b_tau * r_t), dtype=np.float64)

    def short_rate(self, t: float, state: np.ndarray) -> np.ndarray:
        """In Vasicek, state variable is directly the short rate r(t)."""
        return np.asarray(state, dtype=np.float64)

    def zero_coupon_bond(
        self,
        t: float,
        maturity_yrs: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """Compute zero-coupon bond price P(t, T) given short rate r(t)."""
        return self._analytical_bond_price(t, maturity_yrs, state)

    def discount_path(
        self,
        times: np.ndarray,
        state_paths: np.ndarray,
    ) -> np.ndarray:
        """Compute path-wise discount factors D(0, t_i) along paths."""
        times_arr = np.asarray(times, dtype=np.float64)
        paths_arr = np.asarray(state_paths, dtype=np.float64)
        return discount_path_kernel(times_arr, paths_arr)

    def simulate_paths(
        self,
        times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator | None = None,
        dw: np.ndarray | None = None,
        random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
        seed: int | None = None,
        scramble: bool = True,
    ) -> np.ndarray:
        """Simulate short rate r(t) paths under Vasicek."""
        times_arr = np.asarray(times, dtype=np.float64)
        n_steps = len(times_arr) - 1
        dt_vec = np.diff(times_arr)

        if dw is None:
            dw_matrix = generate_brownian_increments(
                n_paths=n_paths,
                dt_vec=dt_vec,
                num_factors=1,
                random_type=random_type,
                seed=seed,
                scramble=scramble,
                rng=rng,
            )
        else:
            dw_matrix = np.asarray(dw, dtype=np.float64)

        return vasicek_simulate_paths_kernel(
            n_paths=n_paths,
            n_steps=n_steps,
            dt_vec=dt_vec,
            kappa=self.kappa_ann,
            theta=self.theta_ann,
            sigma=self.sigma_ann,
            r0=self.r0_ann,
            dw_matrix=dw_matrix,
        )
