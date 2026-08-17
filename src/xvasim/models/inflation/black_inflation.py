"""Black / log-normal Consumer Price Index (CPI) inflation model.

This module implements the benchmark Black-76 log-normal forward CPI model for
analytical and Monte Carlo pricing of inflation derivatives (Zero-Coupon Inflation
Swaps, CPI Options / Inflation Caps & Floors).

Public API
----------
- :class:`BlackInflationModel` — log-normal forward CPI inflation model.
- :class:`BlackInflationParams` — dataclass holding model parameters.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.stats import norm

from ...qmc import RandomSequenceType, generate_brownian_increments
from ..base import InflationModel
from ..registry import ModelRegistry
from .jarrow_yildirim import InflationSimulationResult


@dataclasses.dataclass(frozen=True)
class BlackInflationParams:
    """Parameters for the Black log-normal inflation model.

    Attributes:
        nominal_discount_curve_yrs: 1-D array of tenor pillar points for nominal curve.
        nominal_discount_factors: 1-D array of nominal discount factors.
        real_discount_curve_yrs: 1-D array of tenor pillar points for real curve.
        real_discount_factors: 1-D array of real discount factors.
        base_cpi: Base Consumer Price Index level I(0) (e.g. 100.0).
        cpi_vol_ann: Annualised log-normal CPI volatility.
    """

    nominal_discount_curve_yrs: np.ndarray
    nominal_discount_factors: np.ndarray
    real_discount_curve_yrs: np.ndarray
    real_discount_factors: np.ndarray
    base_cpi: float
    cpi_vol_ann: float


@ModelRegistry.register("inflation", "black")
@ModelRegistry.register("inflation", "lognormal")
class BlackInflationModel(InflationModel):
    """Black log-normal forward CPI inflation model.

    Provides exact analytical formulas and Monte Carlo simulation for zero-coupon
    inflation swaps and European CPI options.
    """

    def __init__(
        self,
        nominal_discount_curve_yrs: np.ndarray,
        nominal_discount_factors: np.ndarray,
        real_discount_curve_yrs: np.ndarray,
        real_discount_factors: np.ndarray,
        base_cpi: float = 100.0,
        cpi_vol_ann: float = 0.02,
    ) -> None:
        """Initialize a Black inflation model.

        Args:
            nominal_discount_curve_yrs: Tenor pillars for nominal discount curve.
            nominal_discount_factors: Nominal discount factors at tenor pillars.
            real_discount_curve_yrs: Tenor pillars for real discount curve.
            real_discount_factors: Real discount factors at tenor pillars.
            base_cpi: Base Consumer Price Index level I(0) (must be > 0).
            cpi_vol_ann: Annualised log-normal CPI volatility (must be >= 0).
        """
        if base_cpi <= 0:
            msg = f"base_cpi must be positive, got {base_cpi}"
            raise ValueError(msg)
        if cpi_vol_ann < 0:
            msg = f"cpi_vol_ann must be non-negative, got {cpi_vol_ann}"
            raise ValueError(msg)

        self._nom_curve = np.asarray(nominal_discount_curve_yrs, dtype=np.float64)
        self._nom_dfs = np.asarray(nominal_discount_factors, dtype=np.float64)
        self._real_curve = np.asarray(real_discount_curve_yrs, dtype=np.float64)
        self._real_dfs = np.asarray(real_discount_factors, dtype=np.float64)
        self._base_cpi = float(base_cpi)
        self._cpi_vol_ann = float(cpi_vol_ann)

    @property
    def model_name(self) -> str:
        """Returns 'black'."""
        return "black"

    @property
    def base_cpi(self) -> float:
        """Base CPI index level I(0)."""
        return self._base_cpi

    @property
    def cpi_vol_ann(self) -> float:
        """Annualised CPI volatility."""
        return self._cpi_vol_ann

    @property
    def nominal_discount_curve_yrs(self) -> np.ndarray:
        """Nominal tenor pillars."""
        return self._nom_curve

    @property
    def nominal_discount_factors(self) -> np.ndarray:
        """Nominal discount factors."""
        return self._nom_dfs

    @property
    def real_discount_curve_yrs(self) -> np.ndarray:
        """Real tenor pillars."""
        return self._real_curve

    @property
    def real_discount_factors(self) -> np.ndarray:
        """Real discount factors."""
        return self._real_dfs

    def interpolate_nominal_df(self, t: float | np.ndarray) -> np.ndarray:
        """Interpolate nominal discount factor at time t."""
        t_arr = np.asarray(t, dtype=np.float64)
        log_dfs = np.log(np.maximum(self._nom_dfs, 1e-18))
        interp_log = np.interp(t_arr, self._nom_curve, log_dfs)
        return np.exp(interp_log)

    def interpolate_real_df(self, t: float | np.ndarray) -> np.ndarray:
        """Interpolate real discount factor at time t."""
        t_arr = np.asarray(t, dtype=np.float64)
        log_dfs = np.log(np.maximum(self._real_dfs, 1e-18))
        interp_log = np.interp(t_arr, self._real_curve, log_dfs)
        return np.exp(interp_log)

    def forward_cpi(self, maturity_yrs: float) -> float:
        """Compute the expected forward CPI level E[I(T)].

        .. math::
            \\mathbb{E}^{\\mathbb{Q}^n}[I(T)] = I(0) \\frac{P_r(0, T)}{P_n(0, T)}

        Args:
            maturity_yrs: Maturity in years.

        Returns:
            Forward CPI level.
        """
        p_nom = float(self.interpolate_nominal_df(maturity_yrs))
        p_real = float(self.interpolate_real_df(maturity_yrs))
        return self._base_cpi * (p_real / max(p_nom, 1e-18))

    def zero_coupon_inflation_swap_rate(self, maturity_yrs: float) -> float:
        """Compute fair zero-coupon inflation swap (ZCIS) rate K_ZC(T).

        .. math::
            K_{ZC}(T) = \\left( \\frac{P_r(0, T)}{P_n(0, T)} \\right)^{1/T} - 1

        Args:
            maturity_yrs: Swap maturity in years.

        Returns:
            Annualised fair swap rate (decimal).
        """
        if maturity_yrs <= 0:
            return 0.0
        fwd_cpi = self.forward_cpi(maturity_yrs)
        ratio = fwd_cpi / self._base_cpi
        return float(ratio ** (1.0 / maturity_yrs) - 1.0)

    def price_consumer_price_index_option_analytical(
        self,
        strike_rate_ann: float,
        maturity_yrs: float,
        notional: float = 1.0,
        is_call: bool = True,
    ) -> float:
        """Analytical Black formula for zero-coupon CPI option (Caplet / Floorlet).

        Payoff at maturity T:
        - Call (Caplet): :math:`N \\times \\max(I(T)/I(0) - (1+K)^T, 0)`
        - Put (Floorlet): :math:`N \\times \\max((1+K)^T - I(T)/I(0), 0)`

        Args:
            strike_rate_ann: Annualised inflation strike rate K (e.g. 0.025 for 2.5%).
            maturity_yrs: Maturity T in years.
            notional: Trade notional.
            is_call: True for Call / Caplet, False for Put / Floorlet.

        Returns:
            Analytical option present value in nominal currency.
        """
        if maturity_yrs <= 0:
            payoff = max(1.0 - (1.0 + strike_rate_ann) ** 0, 0.0) if is_call else 0.0
            return float(notional * payoff)

        p_nom = float(self.interpolate_nominal_df(maturity_yrs))
        fwd_ratio = self.forward_cpi(maturity_yrs) / self._base_cpi
        k_compound = (1.0 + strike_rate_ann) ** maturity_yrs
        total_std = self._cpi_vol_ann * np.sqrt(maturity_yrs)

        if total_std < 1e-12:
            intrinsic = (
                max(fwd_ratio - k_compound, 0.0)
                if is_call
                else max(k_compound - fwd_ratio, 0.0)
            )
            return float(notional * p_nom * intrinsic)

        d1 = (np.log(fwd_ratio / k_compound) + 0.5 * total_std**2) / total_std
        d2 = d1 - total_std

        if is_call:
            pv = p_nom * (
                fwd_ratio * float(norm.cdf(d1)) - k_compound * float(norm.cdf(d2))
            )
        else:
            pv = p_nom * (
                k_compound * float(norm.cdf(-d2)) - fwd_ratio * float(norm.cdf(-d1))
            )

        return float(notional * pv)

    # Convenience alias for backwards compatibility
    price_cpi_option_analytical = price_consumer_price_index_option_analytical

    def simulate_paths(
        self,
        maturity_yrs: float,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator | None = None,
        random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
        seed: int | None = None,
        scramble: bool = True,
    ) -> InflationSimulationResult:
        """Simulate Black CPI paths matching forward curve and volatility.

        Args:
            maturity_yrs: Simulation horizon in years.
            n_paths: Number of Monte Carlo paths.
            n_steps: Number of time steps.
            rng: Optional NumPy random Generator.
            random_type: Random sequence type (:class:`RandomSequenceType` or str).
            seed: Optional random seed.
            scramble: If True, scrambles QMC sequences.

        Returns:
            :class:`InflationSimulationResult` containing simulated paths.
        """
        dt = maturity_yrs / n_steps
        times = np.linspace(0.0, maturity_yrs, n_steps + 1)
        dt_vec = np.diff(times)

        fwd_cpis = np.array([self.forward_cpi(t) for t in times])
        vol = self._cpi_vol_ann

        ln_cpi = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        ln_cpi[:, 0] = np.log(self._base_cpi)

        # Generate Brownian motion increments
        dw = generate_brownian_increments(
            n_paths=n_paths,
            dt_vec=dt_vec,
            num_factors=1,
            random_type=random_type,
            seed=seed,
            scramble=scramble,
            rng=rng,
        )

        # Discrete forward step evolution
        for step in range(n_steps):
            # Forward ratio drift
            ratio_fwd = fwd_cpis[step + 1] / max(fwd_cpis[step], 1e-18)
            drift_step = np.log(ratio_fwd) - 0.5 * vol**2 * dt
            ln_cpi[:, step + 1] = ln_cpi[:, step] + drift_step + vol * dw[:, step]

        cpi_paths = np.exp(ln_cpi)

        # Nominal and real instantaneous short rate approximations from curves
        nom_dfs = np.array([float(self.interpolate_nominal_df(t)) for t in times])
        nom_df_paths = np.tile(nom_dfs, (n_paths, 1))

        # Forward rates
        r_nom_arr = np.zeros(n_steps + 1, dtype=np.float64)
        for i in range(n_steps):
            dt_step = times[i + 1] - times[i]
            r_nom_arr[i] = -np.log(nom_dfs[i + 1] / max(nom_dfs[i], 1e-18)) / dt_step
        r_nom_arr[-1] = r_nom_arr[-2] if n_steps > 0 else 0.0

        r_nom_paths = np.tile(r_nom_arr, (n_paths, 1))
        r_real_paths = np.zeros_like(r_nom_paths)
        x_zeros = np.zeros_like(r_nom_paths)

        return InflationSimulationResult(
            times=times,
            nominal_states=x_zeros,
            real_states=x_zeros,
            cpi_index=cpi_paths,
            nominal_short_rates=r_nom_paths,
            real_short_rates=r_real_paths,
            nominal_discount_factors=nom_df_paths,
        )
