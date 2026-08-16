"""Base abstractions and interfaces for modular stochastic models in XvaSim.

This module defines the abstract base classes and enumerations that form
the foundation of the modular stochastic modeling framework. It provides
unified interfaces for interest rate, FX, credit, and other risk factor
models.

Public API
----------
- :class:`RiskFactorType` — enumeration of supported risk factor categories.
- :class:`StochasticModel` — abstract base class for all stochastic models.
- :class:`InterestRateModel` — abstract base class for 1-factor interest rate models.
- :class:`CreditModel` — abstract base class for credit / hazard rate models.
- :class:`FXModel` — abstract base class for foreign exchange spot / market models.
- :class:`InflationModel` — abstract base class for inflation index & rate models.
"""

from __future__ import annotations

import abc
import enum
import typing

import numpy as np


class RiskFactorType(enum.Enum):
    """Enumeration of financial risk factor categories.

    Members:
        INTEREST_RATE: Interest rate and short-rate models.
        FX: Foreign exchange rate and cross-currency models.
        CREDIT: Credit spread, hazard rate, and default intensity models.
        INFLATION: Inflation index, nominal/real rate, and CPI models.
        EQUITY: Equity and stock price models.
        COMMODITY: Commodity price and convenience yield models.
    """

    INTEREST_RATE = "interest_rate"
    FX = "fx"
    CREDIT = "credit"
    INFLATION = "inflation"
    EQUITY = "equity"
    COMMODITY = "commodity"


class StochasticModel(abc.ABC):
    """Abstract base class for all stochastic risk factor models."""

    @property
    @abc.abstractmethod
    def risk_factor_type(self) -> RiskFactorType:
        """The risk factor category modeled by this instance."""
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier of the model (e.g. 'lgm', 'hull_white')."""
        ...

    @property
    def num_factors(self) -> int:
        """Number of stochastic Brownian factors driving the model (default: 1)."""
        return 1


class InterestRateModel(StochasticModel):
    """Abstract base class for single-currency interest rate models.

    Defines the contract for simulating rate dynamics, computing instantaneous
    short rates, evaluating zero-coupon bond prices, and generating path-wise
    discount factors under the risk-neutral measure.
    """

    @property
    def risk_factor_type(self) -> RiskFactorType:
        """Returns :attr:`RiskFactorType.INTEREST_RATE`."""
        return RiskFactorType.INTEREST_RATE

    @property
    @abc.abstractmethod
    def discount_curve_yrs(self) -> np.ndarray:
        """1-D array of tenor pillar points (years) for the base discount curve."""
        ...

    @property
    @abc.abstractmethod
    def discount_factors(self) -> np.ndarray:
        """1-D array of discount factors corresponding to :attr:`discount_curve_yrs`."""
        ...

    def interpolate_discount_factor(self, t: np.ndarray | float) -> np.ndarray:
        """Log-linearly interpolate (and flat-extrapolate) the base discount curve.

        Args:
            t: Query time(s) in years (scalar or array).

        Returns:
            Interpolated discount factor(s) with the same shape as *t*.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        log_dfs = np.log(np.maximum(self.discount_factors, 1e-18))
        interp_log = np.interp(t_arr, self.discount_curve_yrs, log_dfs)
        return np.exp(interp_log)  # type: ignore[no-any-return]

    def instantaneous_forward(self, t: float) -> float:
        """Approximate the instantaneous forward rate f(0, t) from the curve.

        Uses a finite-difference bump of 1 day (≈ 1/365.25 yr).

        Args:
            t: Time in years.

        Returns:
            Instantaneous forward rate (annualised decimal).
        """
        bump = 1.0 / 365.25
        t_lo = max(t - bump / 2.0, 0.0)
        t_hi = t + bump / 2.0
        df_lo = float(self.interpolate_discount_factor(t_lo))
        df_hi = float(self.interpolate_discount_factor(t_hi))
        return float(-np.log(df_hi / max(df_lo, 1e-18)) / (t_hi - t_lo))

    @abc.abstractmethod
    def short_rate(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the instantaneous short rate r(t) given the state variable(s).

        Args:
            t: Time in years.
            state: State variable values for all paths, shape ``(n_paths,)``
                or ``(n_paths, num_factors)``.

        Returns:
            Short rate array of shape ``(n_paths,)``.
        """
        ...

    @abc.abstractmethod
    def zero_coupon_bond(
        self,
        t: float,
        maturity_yrs: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """Compute zero-coupon bond price P(t, T) given the state variable(s) at time t.

        Args:
            t: Current time in years.
            maturity_yrs: Bond maturity T in years (T >= t).
            state: State variable values for all paths, shape ``(n_paths,)``.

        Returns:
            Bond price array of shape ``(n_paths,)``.
        """
        ...

    @abc.abstractmethod
    def discount_path(
        self,
        times: np.ndarray,
        state_paths: np.ndarray,
    ) -> np.ndarray:
        """Compute path-wise cumulative discount factors D(0, t_i) along paths.

        .. math::
            D(0, t_i) = \\exp\\left(-\\int_0^{t_i} r(s)\\,ds\\right)

        Args:
            times: 1-D array of simulation time grid points, shape ``(n_steps + 1,)``.
            state_paths: State variable paths, shape ``(n_paths, n_steps + 1)``.

        Returns:
            Array of discount factors with shape ``(n_paths, n_steps + 1)``.
        """
        ...

    @abc.abstractmethod
    def simulate_paths(
        self,
        times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator,
        dw: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate state variable paths on a given time grid.

        Args:
            times: 1-D array of simulation times in years, shape ``(n_steps + 1,)``.
            n_paths: Number of Monte Carlo paths.
            rng: Random number generator.
            dw: Optional pre-generated Brownian increments of shape
                ``(n_paths, n_steps)``. If None, standard normal increments
                are generated from *rng*.

        Returns:
            Array of simulated state paths, shape ``(n_paths, n_steps + 1)``.
        """
        ...


class CreditModel(StochasticModel):
    """Abstract base class for credit, hazard-rate, and default intensity models."""

    @property
    def risk_factor_type(self) -> RiskFactorType:
        """Returns :attr:`RiskFactorType.CREDIT`."""
        return RiskFactorType.CREDIT

    @abc.abstractmethod
    def survival_probability(self, tenors_yrs: np.ndarray) -> np.ndarray:
        """Compute survival probabilities P_surv(0, t) at given tenors.

        Args:
            tenors_yrs: 1-D array of maturities / tenors in years.

        Returns:
            1-D array of survival probabilities with same shape as *tenors_yrs*.
        """
        ...

    def marginal_pd(self, tenors_yrs: np.ndarray) -> np.ndarray:
        """Compute marginal default probabilities over tenor buckets [t_{i-1}, t_i].

        Args:
            tenors_yrs: 1-D array of maturities / tenors in years.

        Returns:
            1-D array of marginal default probabilities.
        """
        surv_prob = self.survival_probability(tenors_yrs)
        cum_pd = 1.0 - surv_prob
        return np.diff(cum_pd, prepend=0.0)


class FXModel(StochasticModel):
    """Abstract base class for foreign exchange (FX) market models."""

    @property
    def risk_factor_type(self) -> RiskFactorType:
        """Returns :attr:`RiskFactorType.FX`."""
        return RiskFactorType.FX

    @property
    @abc.abstractmethod
    def spot_fx(self) -> float:
        """Current spot FX rate (units of domestic per 1 foreign)."""
        ...

    @abc.abstractmethod
    def simulate_paths(
        self,
        maturity_yrs: float,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
    ) -> typing.Any:
        """Simulate joint state and FX spot paths under the domestic measure."""
        ...


class InflationModel(StochasticModel):
    """Abstract base class for inflation models (CPI index & rate dynamics)."""

    @property
    def risk_factor_type(self) -> RiskFactorType:
        """Returns :attr:`RiskFactorType.INFLATION`."""
        return RiskFactorType.INFLATION

    @property
    @abc.abstractmethod
    def base_cpi(self) -> float:
        """Base / current Consumer Price Index (CPI) level I(0)."""
        ...

    @abc.abstractmethod
    def forward_cpi(self, maturity_yrs: float) -> float:
        """Compute expected forward CPI E[I(T)] under the nominal measure.

        Args:
            maturity_yrs: Maturity horizon in years.

        Returns:
            Forward CPI level.
        """
        ...

    @abc.abstractmethod
    def zero_coupon_inflation_swap_rate(self, maturity_yrs: float) -> float:
        """Compute the analytical fair zero-coupon inflation swap (ZCIS) rate K_ZC(T).

        .. math::
            K_{ZC}(T) = \\left(\\frac{\\mathbb{E}[I(T)]}{I(0)}\\right)^{1/T} - 1

        Args:
            maturity_yrs: Swap maturity in years.

        Returns:
            Fair annualised zero-coupon swap rate.
        """
        ...

    @abc.abstractmethod
    def simulate_paths(
        self,
        maturity_yrs: float,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
    ) -> typing.Any:
        """Simulate joint nominal rate, real rate, and CPI index paths.

        Args:
            maturity_yrs: Simulation horizon in years.
            n_paths: Number of Monte Carlo simulation paths.
            n_steps: Number of discrete time steps.
            rng: NumPy random generator.

        Returns:
            Simulation results containing time grid, state paths, and CPI index paths.
        """
        ...

