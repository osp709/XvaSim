"""Cox-Ingersoll-Ross (CIR) interest-rate model implementation.

This module implements the classic CIR (1985) square-root short-rate model:

.. math::
    dr(t) = \\kappa_{\\text{ann}}(\\theta_{\\text{ann}} - r(t))\\,dt
    + \\sigma_{\\text{ann}}\\sqrt{r(t)}\\,dW(t)

Public API
----------
- :class:`CIRInterestRateParams` — parameters for the CIR interest rate model.
- :class:`CIRInterestRateModel` — object-oriented CIR interest rate model.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..base import InterestRateModel
from ..registry import ModelRegistry


@dataclasses.dataclass(frozen=True)
class CIRInterestRateParams:
    """Parameters for the Cox-Ingersoll-Ross short-rate model.

    Attributes:
        kappa_ann: Speed of mean reversion (annualised decimal, e.g. 0.20).
        theta_ann: Long-term mean short rate (annualised decimal, e.g. 0.03).
        sigma_ann: Volatility coefficient (annualised decimal, e.g. 0.08).
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


@ModelRegistry.register("interest_rate", "cir")
class CIRInterestRateModel(InterestRateModel):
    """Cox-Ingersoll-Ross (1985) short-rate interest rate model."""

    def __init__(
        self,
        params: CIRInterestRateParams | None = None,
        *,
        kappa_ann: float = 0.20,
        theta_ann: float = 0.03,
        sigma_ann: float = 0.08,
        r0_ann: float = 0.025,
        discount_curve_yrs: np.ndarray | None = None,
        discount_factors: np.ndarray | None = None,
    ) -> None:
        """Initialize a CIR interest rate model."""
        if params is not None:
            self._params = params
        else:
            self._params = CIRInterestRateParams(
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
        """Returns 'cir'."""
        return "cir"

    @property
    def params(self) -> CIRInterestRateParams:
        """The underlying :class:`CIRInterestRateParams`."""
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
        """Volatility coefficient."""
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
        """Compute analytical CIR bond price P(t, T)."""
        tau = np.maximum(np.asarray(maturity_yrs, dtype=np.float64) - t, 0.0)
        kappa = self.kappa_ann
        theta = self.theta_ann
        sigma = self.sigma_ann

        gamma = np.sqrt(kappa**2 + 2.0 * sigma**2)
        exp_gamma_tau = np.exp(gamma * tau)
        denom = (kappa + gamma) * (exp_gamma_tau - 1.0) + 2.0 * gamma

        power = (2.0 * kappa * theta) / (sigma**2)
        num_a = 2.0 * gamma * np.exp((kappa + gamma) * tau / 2.0)
        a_tau = (num_a / denom) ** power
        b_tau = 2.0 * (exp_gamma_tau - 1.0) / denom

        return a_tau * np.exp(-b_tau * r_t)  # type: ignore[no-any-return]

    def short_rate(self, t: float, state: np.ndarray) -> np.ndarray:
        """In CIR, state variable is directly the short rate r(t)."""
        return np.maximum(np.asarray(state, dtype=np.float64), 0.0)

    def zero_coupon_bond(
        self,
        t: float,
        maturity_yrs: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """Compute zero-coupon bond price P(t, T) given short rate r(t)."""
        return self._analytical_bond_price(t, maturity_yrs, np.maximum(state, 0.0))

    def discount_path(
        self,
        times: np.ndarray,
        state_paths: np.ndarray,
    ) -> np.ndarray:
        """Compute path-wise discount factors D(0, t_i) along paths."""
        n_paths, n_times = state_paths.shape
        dt_vec = np.diff(times)
        cum_integral = np.zeros((n_paths, n_times))
        clamped_paths = np.maximum(state_paths, 0.0)
        for j in range(1, n_times):
            cum_integral[:, j] = cum_integral[:, j - 1] + 0.5 * dt_vec[j - 1] * (
                clamped_paths[:, j - 1] + clamped_paths[:, j]
            )
        return np.exp(-cum_integral)

    def simulate_paths(
        self,
        times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator,
        dw: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate short rate r(t) paths under CIR using full truncation."""
        n_steps = len(times) - 1
        dt_vec = np.diff(times)
        r_paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        r_paths[:, 0] = self.r0_ann

        if dw is None:
            std_normals = rng.standard_normal((n_paths, n_steps))
            dw_matrix = std_normals * np.sqrt(dt_vec)
        else:
            dw_matrix = dw

        for step in range(n_steps):
            dt = dt_vec[step]
            r_pos = np.maximum(r_paths[:, step], 0.0)
            drift = self.kappa_ann * (self.theta_ann - r_pos) * dt
            diffusion = self.sigma_ann * np.sqrt(r_pos) * dw_matrix[:, step]
            r_paths[:, step + 1] = np.maximum(r_paths[:, step] + drift + diffusion, 0.0)

        return r_paths
