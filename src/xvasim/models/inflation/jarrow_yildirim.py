"""Jarrow-Yildirim (JY) modular inflation model.

This module implements the two-economy Jarrow-Yildirim (2003) framework for
modeling inflation dynamics. The economy consists of:
1. A nominal economy driven by a modular :class:`~xvasim.models.base.InterestRateModel`.
2. A real economy driven by a modular :class:`~xvasim.models.base.InterestRateModel`
   with nominal-measure quanto/market-price-of-risk drift adjustment.
3. A Consumer Price Index (CPI) :math:`I(t)` modeled as a geometric diffusion
   under the nominal risk-neutral measure:
   .. math::
       dI(t) = I(t) \\left[ (r_n(t) - r_r(t)) dt + \\sigma_I dW_I(t) \\right]

Public API
----------
- :class:`JarrowYildirimModel` — modular Jarrow-Yildirim inflation model.
- :class:`JarrowYildirimParams` — dataclass for Jarrow-Yildirim model parameters.
- :class:`InflationSimulationResult` — structured container for simulation paths.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from ...qmc import RandomSequenceType, generate_normal_draws
from ..base import InflationModel, InterestRateModel
from ..ir.hull_white import HullWhite1FModel
from ..ir.lgm import LGMModel, LGMParams
from ..registry import ModelRegistry


@dataclasses.dataclass(frozen=True)
class InflationSimulationResult:
    """Simulation trajectory container for inflation models.

    Supports tuple unpacking: ``times, nom_states, real_states, cpi = result``.

    Attributes:
        times: 1-D array of simulation time grid points, shape ``(n_steps + 1,)``.
        nominal_states: Simulated nominal rate state paths,
            shape ``(n_paths, n_steps + 1)``.
        real_states: Simulated real rate state paths,
            shape ``(n_paths, n_steps + 1)``.
        cpi_index: Simulated Consumer Price Index paths,
            shape ``(n_paths, n_steps + 1)``.
        nominal_short_rates: Simulated instantaneous nominal short rates,
            shape ``(n_paths, n_steps + 1)``.
        real_short_rates: Simulated instantaneous real short rates,
            shape ``(n_paths, n_steps + 1)``.
        nominal_discount_factors: Path-wise cumulative nominal discount factors,
            shape ``(n_paths, n_steps + 1)``.
    """

    times: np.ndarray
    nominal_states: np.ndarray
    real_states: np.ndarray
    cpi_index: np.ndarray
    nominal_short_rates: np.ndarray
    real_short_rates: np.ndarray
    nominal_discount_factors: np.ndarray

    def __iter__(self) -> typing.Iterator[np.ndarray]:
        """Support 4-tuple unpacking (times, x_nom, x_real, cpi)."""
        return iter((self.times, self.nominal_states, self.real_states, self.cpi_index))

    def __getitem__(self, index: int) -> np.ndarray:
        """Support tuple indexing for the primary 4 outputs."""
        return (self.times, self.nominal_states, self.real_states, self.cpi_index)[
            index
        ]

    def __len__(self) -> int:
        """Return 4 for standard tuple unpacking."""
        return 4


@dataclasses.dataclass(frozen=True)
class JarrowYildirimParams:
    """Dataclass holding parameters for a Jarrow-Yildirim inflation model.

    Attributes:
        nominal_ir_model: Interest rate model for the nominal economy.
        real_ir_model: Interest rate model for the real economy.
        base_cpi: Base / current Consumer Price Index level I(0) (e.g. 100.0).
        cpi_vol_ann: Annualised log-normal volatility of the CPI index.
        correlation_matrix: 3×3 correlation matrix ordered as
            ``[nominal_rate, real_rate, cpi_index]``.
    """

    nominal_ir_model: InterestRateModel
    real_ir_model: InterestRateModel
    base_cpi: float
    cpi_vol_ann: float
    correlation_matrix: np.ndarray


@ModelRegistry.register("inflation", "jarrow_yildirim")
@ModelRegistry.register("inflation", "jy")
@ModelRegistry.register("inflation", "two_factor_hw")
class JarrowYildirimModel(InflationModel):
    """Modular Jarrow-Yildirim inflation model.

    Combines modular nominal and real interest rate models with log-normal CPI
    dynamics under the nominal risk-neutral measure.
    """

    def __init__(
        self,
        nominal_ir_model: InterestRateModel,
        real_ir_model: InterestRateModel,
        base_cpi: float = 100.0,
        cpi_vol_ann: float = 0.02,
        correlation_matrix: np.ndarray | None = None,
    ) -> None:
        """Initialize a Jarrow-Yildirim inflation model.

        Args:
            nominal_ir_model: Interest rate model for nominal currency.
            real_ir_model: Interest rate model for real currency / goods economy.
            base_cpi: Base Consumer Price Index level I(0) (must be > 0).
            cpi_vol_ann: Annualised log-normal CPI index volatility (must be >= 0).
            correlation_matrix: 3×3 correlation matrix ordered as
                ``[nominal_rate, real_rate, cpi_index]``. Defaults to 3×3 identity.
        """
        if base_cpi <= 0:
            msg = f"base_cpi must be positive, got {base_cpi}"
            raise ValueError(msg)
        if cpi_vol_ann < 0:
            msg = f"cpi_vol_ann must be non-negative, got {cpi_vol_ann}"
            raise ValueError(msg)

        self._nominal_ir = nominal_ir_model
        self._real_ir = real_ir_model
        self._base_cpi = float(base_cpi)
        self._cpi_vol_ann = float(cpi_vol_ann)

        if correlation_matrix is None:
            self._correlation_matrix = np.eye(3, dtype=np.float64)
        else:
            self._correlation_matrix = np.asarray(correlation_matrix, dtype=np.float64)
            if self._correlation_matrix.shape != (3, 3):
                msg = (
                    "correlation_matrix must be 3x3, got "
                    f"{self._correlation_matrix.shape}"
                )
                raise ValueError(msg)

    @property
    def model_name(self) -> str:
        """Returns 'jarrow_yildirim'."""
        return "jarrow_yildirim"

    @property
    def nominal_ir_model(self) -> InterestRateModel:
        """The nominal interest rate model."""
        return self._nominal_ir

    @property
    def real_ir_model(self) -> InterestRateModel:
        """The real interest rate model."""
        return self._real_ir

    @property
    def base_cpi(self) -> float:
        """Base CPI index level I(0)."""
        return self._base_cpi

    @property
    def cpi_vol_ann(self) -> float:
        """Annualised CPI volatility."""
        return self._cpi_vol_ann

    @property
    def correlation_matrix(self) -> np.ndarray:
        """3×3 correlation matrix."""
        return self._correlation_matrix

    @classmethod
    def from_lgm_params(
        cls,
        nominal: LGMParams,
        real: LGMParams,
        base_cpi: float = 100.0,
        cpi_vol_ann: float = 0.02,
        correlation_matrix: np.ndarray | None = None,
    ) -> JarrowYildirimModel:
        """Construct a JarrowYildirimModel from nominal and real LGMParams instances."""
        return cls(
            nominal_ir_model=LGMModel(nominal),
            real_ir_model=LGMModel(real),
            base_cpi=base_cpi,
            cpi_vol_ann=cpi_vol_ann,
            correlation_matrix=correlation_matrix,
        )

    def forward_cpi(self, maturity_yrs: float) -> float:
        """Compute the theoretical forward CPI level E[I(T)] at maturity T.

        Under the nominal risk-neutral measure:
        .. math::
            \\mathbb{E}^{\\mathbb{Q}^n}[I(T)] = I(0) \\frac{P_r(0, T)}{P_n(0, T)}

        Args:
            maturity_yrs: Maturity in years.

        Returns:
            Forward CPI level.
        """
        p_nom = float(self._nominal_ir.interpolate_discount_factor(maturity_yrs))
        p_real = float(self._real_ir.interpolate_discount_factor(maturity_yrs))
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

    def total_variance_at(self, maturity_yrs: float) -> float:
        """Compute integrated CPI variance V^2(T) under the forward measure.

        For Hull-White / LGM components, calculates the analytical integrated
        forward CPI variance.
        """
        if maturity_yrs <= 0:
            return 0.0

        # Base CPI diffusion variance
        var_cpi = self._cpi_vol_ann**2 * maturity_yrs

        # Check if analytical HW / LGM variance adjustment is available
        sig_n = 0.0
        kappa_n = 0.0
        if isinstance(self._nominal_ir, LGMModel):
            kappa_n = self._nominal_ir.kappa_ann
            sig_n = float(self._nominal_ir.sigma_at(maturity_yrs / 2.0))
        elif isinstance(self._nominal_ir, HullWhite1FModel):
            kappa_n = self._nominal_ir.a_ann
            sig_n = self._nominal_ir.sigma_ann

        sig_r = 0.0
        kappa_r = 0.0
        if isinstance(self._real_ir, LGMModel):
            kappa_r = self._real_ir.kappa_ann
            sig_r = float(self._real_ir.sigma_at(maturity_yrs / 2.0))
        elif isinstance(self._real_ir, HullWhite1FModel):
            kappa_r = self._real_ir.a_ann
            sig_r = self._real_ir.sigma_ann

        if sig_n == 0.0 and sig_r == 0.0:
            return var_cpi

        # Integration over [0, T]
        n_integration_steps = 100
        t_grid = np.linspace(0.0, maturity_yrs, n_integration_steps + 1)
        dt_int = maturity_yrs / n_integration_steps

        rho_n_r = self._correlation_matrix[0, 1]
        rho_n_i = self._correlation_matrix[0, 2]
        rho_r_i = self._correlation_matrix[1, 2]

        total_v = 0.0
        for idx in range(n_integration_steps):
            t_mid = 0.5 * (t_grid[idx] + t_grid[idx + 1])
            tau = maturity_yrs - t_mid

            b_n = (
                (1.0 - np.exp(-kappa_n * tau)) / kappa_n
                if abs(kappa_n) > 1e-10
                else tau
            )
            b_r = (
                (1.0 - np.exp(-kappa_r * tau)) / kappa_r
                if abs(kappa_r) > 1e-10
                else tau
            )

            # Volatility vector for [W_n, W_r, W_I]
            # CPI forward drift components
            v_n = sig_n * b_n
            v_r = -sig_r * b_r
            v_i = self._cpi_vol_ann

            inst_var = (
                v_n**2
                + v_r**2
                + v_i**2
                + 2.0 * v_n * v_r * rho_n_r
                + 2.0 * v_n * v_i * rho_n_i
                + 2.0 * v_r * v_i * rho_r_i
            )
            total_v += inst_var * dt_int

        return max(total_v, 1e-12)

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
        """Simulate joint state, short rates, and CPI index paths.

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
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, maturity_yrs, n_steps + 1)

        chol = np.linalg.cholesky(self._correlation_matrix)

        x_nom = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        x_real = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        r_nom = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        r_real = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        ln_cpi = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        ln_cpi[:, 0] = np.log(self._base_cpi)

        rho_r_cpi = self._correlation_matrix[1, 2]
        vol_cpi = self._cpi_vol_ann

        # Initial short rates at t=0
        r_nom[:, 0] = self._nominal_ir.short_rate(0.0, x_nom[:, 0])
        r_real[:, 0] = self._real_ir.short_rate(0.0, x_real[:, 0])

        z_all = generate_normal_draws(
            n_paths=n_paths,
            dimension=3 * n_steps,
            random_type=random_type,
            seed=seed,
            scramble=scramble,
            rng=rng,
        ).reshape(n_paths, n_steps, 3)

        for step in range(n_steps):
            t = times[step]

            z_indep = z_all[:, step, :]
            z_corr = z_indep @ chol.T

            dw_n = z_corr[:, 0] * sqrt_dt
            dw_r = z_corr[:, 1] * sqrt_dt
            dw_cpi = z_corr[:, 2] * sqrt_dt

            # Nominal state evolution
            if isinstance(self._nominal_ir, LGMModel):
                sig_n = self._nominal_ir.sigma_at(t)
                kn = self._nominal_ir.kappa_ann
                x_nom[:, step + 1] = (
                    x_nom[:, step] - kn * x_nom[:, step] * dt + sig_n * dw_n
                )
            else:
                sim_step = self._nominal_ir.simulate_paths(
                    np.array([t, t + dt]),
                    n_paths,
                    rng,
                    dw=dw_n.reshape(-1, 1),
                )
                x_nom[:, step + 1] = sim_step[:, -1]

            # Real state evolution with quanto drift adjustment under nominal measure
            if isinstance(self._real_ir, LGMModel):
                sig_r = self._real_ir.sigma_at(t)
                kr = self._real_ir.kappa_ann
                quanto_drift = -sig_r * rho_r_cpi * vol_cpi
                x_real[:, step + 1] = (
                    x_real[:, step]
                    - kr * x_real[:, step] * dt
                    + quanto_drift * dt
                    + sig_r * dw_r
                )
            else:
                sim_step = self._real_ir.simulate_paths(
                    np.array([t, t + dt]),
                    n_paths,
                    rng,
                    dw=dw_r.reshape(-1, 1),
                )
                x_real[:, step + 1] = sim_step[:, -1]

            # Compute short rates
            df_n_t = float(self._nominal_ir.interpolate_discount_factor(t))
            df_n_t1 = float(self._nominal_ir.interpolate_discount_factor(t + dt))
            df_r_t = float(self._real_ir.interpolate_discount_factor(t))
            df_r_t1 = float(self._real_ir.interpolate_discount_factor(t + dt))

            fwd_n = -np.log(df_n_t1 / max(df_n_t, 1e-18)) / dt
            fwd_r = -np.log(df_r_t1 / max(df_r_t, 1e-18)) / dt

            if isinstance(self._nominal_ir, LGMModel):
                kn = self._nominal_ir.kappa_ann
                h_prime_n = np.exp(-kn * t)
                zeta_n = self._nominal_ir.zeta(t)
                r_n_step = (
                    fwd_n + h_prime_n * x_nom[:, step] + 0.5 * h_prime_n**2 * zeta_n
                )
            else:
                r_n_step = self._nominal_ir.short_rate(t, x_nom[:, step])

            if isinstance(self._real_ir, LGMModel):
                kr = self._real_ir.kappa_ann
                h_prime_r = np.exp(-kr * t)
                zeta_r = self._real_ir.zeta(t)
                r_r_step = (
                    fwd_r + h_prime_r * x_real[:, step] + 0.5 * h_prime_r**2 * zeta_r
                )
            else:
                r_r_step = self._real_ir.short_rate(t, x_real[:, step])

            r_nom[:, step] = r_n_step
            r_real[:, step] = r_r_step

            # CPI index evolution
            drift_cpi = r_n_step - r_r_step - 0.5 * vol_cpi**2
            ln_cpi[:, step + 1] = (
                ln_cpi[:, step] + drift_cpi * dt + vol_cpi * dw_cpi
            )

        # Final short rates
        t_end = times[-1]
        r_nom[:, -1] = self._nominal_ir.short_rate(t_end, x_nom[:, -1])
        r_real[:, -1] = self._real_ir.short_rate(t_end, x_real[:, -1])

        cpi_paths = np.exp(ln_cpi)

        # Path-wise cumulative nominal discount factors
        dt_arr = np.diff(times)
        integrated_r_nom = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        integrated_r_nom[:, 1:] = np.cumsum(r_nom[:, :-1] * dt_arr, axis=1)
        nom_dfs = np.exp(-integrated_r_nom)

        return InflationSimulationResult(
            times=times,
            nominal_states=x_nom,
            real_states=x_real,
            cpi_index=cpi_paths,
            nominal_short_rates=r_nom,
            real_short_rates=r_real,
            nominal_discount_factors=nom_dfs,
        )
