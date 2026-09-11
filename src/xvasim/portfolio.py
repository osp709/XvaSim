"""Portfolio / netting-set exposure layer for the XVA pipeline.

This module connects the modular stochastic models in :mod:`xvasim.models`
with the path-wise XVA aggregation engine in :mod:`xvasim.cva_engine` by:

1. simulating joint market-factor paths once per scenario batch
   (:func:`simulate_market`),
2. valuing each trade along every path at each exposure date
   (:class:`Trade`, :class:`FXForwardTrade`, :class:`FXEuropeanOptionTrade`),
3. aggregating trade marks into a netting set / portfolio
   (:class:`Portfolio`, :func:`compute_portfolio_exposure`), and
4. driving the full XVA ledger (CVA / DVA / FVA / KVA / MVA) from the
   simulated exposure (:func:`compute_portfolio_xva`).

Trades are valued *conditional on the simulated state at each exposure
date* using model-consistent analytical bond prices (Gaussian interest
rate components) and Black-76 forward/option formulas. Two market
configurations are supported:

- **Stochastic rates** (:class:`~xvasim.models.fx.TwoCurrencyFXModel`):
  conditional domestic/foreign zero-coupon bond prices come from the
  component interest-rate models' :meth:`InterestRateModel.zero_coupon_bond`.
- **Deterministic curves** (:class:`~xvasim.models.fx.GarmanKohlhagenFXModel`,
  :class:`~xvasim.models.fx.HestonFXModel`): the conditional bond is the
  ratio of deterministic discount factors.

Public API
----------
- :class:`Trade` — abstract derivative contract base.
- :class:`FXForwardTrade` — foreign-exchange forward contract.
- :class:`FXEuropeanOptionTrade` — vanilla European FX option.
- :class:`Portfolio` — netting set aggregating a collection of trades.
- :class:`MarketSimulation` — shared simulated market-factor context.
- :class:`PortfolioExposureResult` — simulated exposure result object.
- :func:`simulate_market` — simulate the market-factor paths once.
- :func:`compute_portfolio_exposure` — value trades and aggregate exposure.
- :func:`simulate_portfolio_exposure` — simulate + value + aggregate.
- :func:`compute_portfolio_xva` — full CVA/DVA/FVA/KVA/MVA ledger.

Units & Conventions
-------------------
- Time / tenor in **years** (suffix ``_yrs``).
- Rates / spreads as **annualised decimals** (suffix ``_ann``).
- FX notions are in **foreign currency**; values are in **domestic currency**.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
from scipy.stats import norm

from .cva_engine import compute_exposure_profile, compute_total_xva
from .models.base import FXModel
from .models.credit.cir import CIRHazardRateModel
from .models.fx.garman_kohlhagen import GarmanKohlhagenFXModel
from .models.fx.two_currency import TwoCurrencyFXModel
from .pricing_engine import OptionType
from .qmc import RandomSequenceType

__all__ = [
    "FXEuropeanOptionTrade",
    "FXForwardTrade",
    "MarketSimulation",
    "Portfolio",
    "PortfolioExposureResult",
    "Trade",
    "compute_portfolio_exposure",
    "compute_portfolio_xva",
    "simulate_market",
    "simulate_portfolio_exposure",
]


# ---------------------------------------------------------------------------
# Trade contracts
# ---------------------------------------------------------------------------


class Trade(typing.Protocol):
    """Structural contract for a derivative contract valued on market paths.

    Any object exposing ``trade_id``, ``notional``, ``maturity_yrs`` and a
    ``value_paths(sim)`` method satisfies this protocol. Concrete trades in
    this module are :class:`FXForwardTrade` and
    :class:`FXEuropeanOptionTrade`; users may implement the same interface
    with custom payoff logic.
    """

    @property
    def trade_id(self) -> str:
        """Unique trade identifier within the portfolio."""
        ...

    @property
    def notional(self) -> float:
        """Notional amount in foreign currency units."""
        ...

    @property
    def maturity_yrs(self) -> float:
        """Maturity in years."""
        ...

    def value_paths(self, sim: MarketSimulation) -> np.ndarray:
        """Return the trade's mark-to-market at every exposure date.

        Args:
            sim: Shared simulated market context
                (:class:`MarketSimulation`).

        Returns:
            Array of shape ``(n_paths, n_dates)`` with the trade value in
            domestic currency at each simulation date.
        """
        ...


def _validate_trade(
    trade_id: str,
    notional: float,
    strike_fx: float,
    maturity_yrs: float,
) -> None:
    """Validate common trade contract parameters."""
    if not trade_id or not isinstance(trade_id, str):
        msg = f"trade_id must be a non-empty string, got {trade_id!r}"
        raise ValueError(msg)
    if not np.isfinite(notional):
        msg = f"notional must be finite, got {notional}"
        raise ValueError(msg)
    if strike_fx <= 0.0:
        msg = f"strike_fx must be strictly positive, got {strike_fx}"
        raise ValueError(msg)
    if maturity_yrs <= 0.0:
        msg = f"maturity_yrs must be strictly positive, got {maturity_yrs}"
        raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class FXForwardTrade:
    """Foreign-exchange forward: pays ``N × (S(T) − K)`` in domestic currency.

    Attributes:
        trade_id: Unique trade identifier within the portfolio.
        notional: Notional amount in foreign currency units.
        strike_fx: Forward strike (domestic per foreign).
        maturity_yrs: Maturity in years.
    """

    trade_id: str
    notional: float
    strike_fx: float
    maturity_yrs: float

    def __post_init__(self) -> None:
        _validate_trade(self.trade_id, self.notional, self.strike_fx, self.maturity_yrs)

    def value_paths(self, sim: MarketSimulation) -> np.ndarray:
        """Path-wise forward value ``N × (S(t)·P_f(t,T) − K·P_d(t,T))``."""
        return _fx_forward_value_paths(
            sim=sim,
            notional=self.notional,
            strike_fx=self.strike_fx,
            maturity_yrs=self.maturity_yrs,
        )


@dataclasses.dataclass(frozen=True)
class FXEuropeanOptionTrade:
    """Vanilla European FX option on spot, valued with Black-76.

    The option is valued conditional on the simulated spot at each exposure
    date using the constant-volatility Black-76 formula under the conditional
    forward ``F(t) = S(t)·P_f(t,T)/P_d(t,T)``. Option pricing therefore
    requires a model exposing a constant ``fx_vol_ann``
    (:class:`~xvasim.models.fx.TwoCurrencyFXModel` and
    :class:`~xvasim.models.fx.GarmanKohlhagenFXModel`); stochastic-volatility
    models are not yet supported for option trades.

    Attributes:
        trade_id: Unique trade identifier within the portfolio.
        notional: Notional amount in foreign currency units.
        strike_fx: Option strike (domestic per foreign).
        maturity_yrs: Option expiry in years.
        option_type: Call or put (:class:`~xvasim.pricing_engine.OptionType`
            or ``"call"`` / ``"put"``).
    """

    trade_id: str
    notional: float
    strike_fx: float
    maturity_yrs: float
    option_type: OptionType | str = OptionType.CALL

    def __post_init__(self) -> None:
        _validate_trade(self.trade_id, self.notional, self.strike_fx, self.maturity_yrs)

    def value_paths(self, sim: MarketSimulation) -> np.ndarray:
        """Path-wise Black-76 option value at every exposure date."""
        return _fx_option_value_paths(
            sim=sim,
            notional=self.notional,
            strike_fx=self.strike_fx,
            maturity_yrs=self.maturity_yrs,
            option_type=self.option_type,
        )


# ---------------------------------------------------------------------------
# Shared market simulation context
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MarketSimulation:
    """Shared simulated market-factor context for a scenario batch.

    Attributes:
        times_yrs: 1-D exposure date grid (years), shape ``(n_dates,)``.
        n_paths: Number of Monte Carlo paths.
        fx_model: FX market model that produced the simulation.
        fx_spot_paths: FX spot paths, shape ``(n_paths, n_dates)``.
        domestic_state_paths: Domestic IR state paths (zeros for
            deterministic-curve models), shape ``(n_paths, n_dates)``.
        foreign_state_paths: Foreign IR state paths (zeros for
            deterministic-curve models), shape ``(n_paths, n_dates)``.
        domestic_df_paths: Bank-account / deterministic domestic discount
            factors D_d(0, t_j), shape ``(n_paths, n_dates)``.
        foreign_df_paths: Bank-account / deterministic foreign discount
            factors D_f(0, t_j), shape ``(n_paths, n_dates)``.
    """

    times_yrs: np.ndarray
    n_paths: int
    fx_model: FXModel
    fx_spot_paths: np.ndarray
    domestic_state_paths: np.ndarray
    foreign_state_paths: np.ndarray
    domestic_df_paths: np.ndarray
    foreign_df_paths: np.ndarray

    def __post_init__(self) -> None:
        n_dates = self.times_yrs.shape[0]
        for field_name in (
            "fx_spot_paths",
            "domestic_state_paths",
            "foreign_state_paths",
            "domestic_df_paths",
            "foreign_df_paths",
        ):
            arr = getattr(self, field_name)
            if arr.shape != (self.n_paths, n_dates):
                msg = (
                    f"{field_name} must have shape ({self.n_paths}, {n_dates}), "
                    f"got {arr.shape}"
                )
                raise ValueError(msg)


def _conditional_domestic_bond(
    sim: MarketSimulation, col: int, maturity_yrs: float
) -> np.ndarray:
    """Conditional domestic zero-coupon bond price P_d(t_j, T) along paths."""
    fx = sim.fx_model
    t = float(sim.times_yrs[col])
    if isinstance(fx, TwoCurrencyFXModel):
        states = sim.domestic_state_paths[:, col]
        return np.asarray(
            fx.domestic_ir_model.zero_coupon_bond(t, maturity_yrs, states),
            dtype=np.float64,
        )
    df_t = float(fx.domestic_discount_factor(t))
    df_mat = float(fx.domestic_discount_factor(maturity_yrs))
    return np.full(sim.n_paths, df_mat / max(df_t, 1e-18))


def _conditional_foreign_bond(
    sim: MarketSimulation, col: int, maturity_yrs: float
) -> np.ndarray:
    """Conditional foreign zero-coupon bond price P_f(t_j, T) along paths."""
    fx = sim.fx_model
    t = float(sim.times_yrs[col])
    if isinstance(fx, TwoCurrencyFXModel):
        states = sim.foreign_state_paths[:, col]
        return np.asarray(
            fx.foreign_ir_model.zero_coupon_bond(t, maturity_yrs, states),
            dtype=np.float64,
        )
    df_t = float(fx.foreign_discount_factor(t))
    df_mat = float(fx.foreign_discount_factor(maturity_yrs))
    return np.full(sim.n_paths, df_mat / max(df_t, 1e-18))


def _fx_forward_value_paths(
    sim: MarketSimulation,
    notional: float,
    strike_fx: float,
    maturity_yrs: float,
) -> np.ndarray:
    """Path-wise value of an FX forward at each exposure date."""
    n_dates = sim.times_yrs.shape[0]
    out = np.empty((sim.n_paths, n_dates), dtype=np.float64)
    for j in range(n_dates):
        t = float(sim.times_yrs[j])
        if t >= maturity_yrs:
            if abs(t - maturity_yrs) < 1e-12:
                out[:, j] = notional * (sim.fx_spot_paths[:, j] - strike_fx)
            else:
                out[:, j] = 0.0
            continue
        p_d = _conditional_domestic_bond(sim, j, maturity_yrs)
        p_f = _conditional_foreign_bond(sim, j, maturity_yrs)
        out[:, j] = notional * (
            sim.fx_spot_paths[:, j] * p_f - strike_fx * p_d
        )
    return out


def _option_is_call(option_type: typing.Any) -> bool:
    """Resolve an option type to a call/put boolean."""
    if isinstance(option_type, str):
        return option_type.strip().lower() == "call"
    opt_val = getattr(option_type, "value", str(option_type)).lower()
    return opt_val.endswith("call")


def _fx_option_value_paths(
    sim: MarketSimulation,
    notional: float,
    strike_fx: float,
    maturity_yrs: float,
    option_type: typing.Any,
) -> np.ndarray:
    """Path-wise Black-76 value of a European FX option at each exposure date."""
    if not isinstance(sim.fx_model, (TwoCurrencyFXModel, GarmanKohlhagenFXModel)):
        msg = (
            "FXEuropeanOptionTrade requires a model exposing a constant "
            f"fx_vol_ann, got {sim.fx_model.__class__.__name__}"
        )
        raise NotImplementedError(msg)

    vol = sim.fx_model.fx_vol_ann
    is_call = _option_is_call(option_type)
    omega = 1.0 if is_call else -1.0

    n_dates = sim.times_yrs.shape[0]
    out = np.empty((sim.n_paths, n_dates), dtype=np.float64)
    for j in range(n_dates):
        t = float(sim.times_yrs[j])
        if t >= maturity_yrs:
            if abs(t - maturity_yrs) < 1e-12:
                intrinsic = np.maximum(
                    omega * (sim.fx_spot_paths[:, j] - strike_fx), 0.0
                )
                out[:, j] = notional * intrinsic
            else:
                out[:, j] = 0.0
            continue

        p_d = _conditional_domestic_bond(sim, j, maturity_yrs)
        p_f = _conditional_foreign_bond(sim, j, maturity_yrs)
        fwd = sim.fx_spot_paths[:, j] * p_f / np.maximum(p_d, 1e-18)

        rem_yrs = maturity_yrs - t
        vol_sqrt = vol * np.sqrt(rem_yrs)
        if vol_sqrt < 1e-12:
            intrinsic = np.maximum(omega * (fwd - strike_fx), 0.0)
            out[:, j] = notional * p_d * intrinsic
            continue

        d1 = (np.log(fwd / max(strike_fx, 1e-18)) + 0.5 * vol**2 * rem_yrs) / vol_sqrt
        d2 = d1 - vol_sqrt
        black_price = omega * (
            fwd * norm.cdf(omega * d1) - strike_fx * norm.cdf(omega * d2)
        )
        out[:, j] = notional * p_d * black_price
    return out


# ---------------------------------------------------------------------------
# Portfolio & netting set
# ---------------------------------------------------------------------------


class Portfolio:
    """A portfolio of trades aggregated as a single netting set.

    Args:
        portfolio_id: Optional identifier for the portfolio / netting set.
        trades: Sequence of :class:`Trade` instances to aggregate.
        netting: If True (default), exposure is the positive part of the
            net portfolio mark (trades offset one another). If False, each
            trade's positive exposure is summed without offsetting.

    Raises:
        ValueError: If trade identifiers are duplicated.
    """

    def __init__(
        self,
        portfolio_id: str = "portfolio_0",
        trades: typing.Sequence[Trade] = (),
        netting: bool = True,
    ) -> None:
        if not portfolio_id or not isinstance(portfolio_id, str):
            msg = f"portfolio_id must be a non-empty string, got {portfolio_id!r}"
            raise ValueError(msg)
        trade_list = list(trades)
        seen: set[str] = set()
        for trade in trade_list:
            if not isinstance(trade, (FXForwardTrade, FXEuropeanOptionTrade)):
                msg = (
                    f"trades must implement the Trade contract "
                    f"(FXForwardTrade / FXEuropeanOptionTrade), got "
                    f"{type(trade).__name__}"
                )
                raise ValueError(msg)
            if trade.trade_id in seen:
                msg = f"duplicate trade_id in portfolio: {trade.trade_id!r}"
                raise ValueError(msg)
            seen.add(trade.trade_id)
        self._portfolio_id = portfolio_id
        self._trades = trade_list
        self._netting = netting

    @property
    def portfolio_id(self) -> str:
        """The portfolio / netting-set identifier."""
        return self._portfolio_id

    @property
    def trades(self) -> list[Trade]:
        """The trades contained in this portfolio (insertion order)."""
        return list(self._trades)

    @property
    def trade_ids(self) -> list[str]:
        """Identifiers of the trades in the portfolio (insertion order)."""
        return [trade.trade_id for trade in self._trades]

    @property
    def netting(self) -> bool:
        """Whether positive and negative trade marks offset within the set."""
        return self._netting


@dataclasses.dataclass(frozen=True)
class PortfolioExposureResult:
    """Simulated exposure result for a portfolio / netting set.

    Attributes:
        times_yrs: 1-D exposure date grid (years), shape ``(n_dates,)``.
        n_paths: Number of Monte Carlo paths.
        trade_mtm: Mapping of ``trade_id`` to its path-wise mark-to-market,
            shape ``(n_paths, n_dates)`` each.
        net_mtm: Signed net portfolio mark (sum of trade marks),
            shape ``(n_paths, n_dates)``.
        exposure: Positive (credit) exposure ``max(net_mtm, 0)`` (or the
            gross sum when netting is disabled), shape ``(n_paths, n_dates)``.
        negative_exposure: Negative exposure ``max(-net_mtm, 0)`` (or the
            gross negative sum when netting is disabled).
        domestic_df_paths: Domestic discount factors D_d(0, t_j),
            shape ``(n_paths, n_dates)`` — the discount used for XVA.
        foreign_df_paths: Foreign discount factors, shape ``(n_paths, n_dates)``.
        fx_spot_paths: Simulated FX spot paths, shape ``(n_paths, n_dates)``.
    """

    times_yrs: np.ndarray
    n_paths: int
    trade_mtm: dict[str, np.ndarray]
    net_mtm: np.ndarray
    exposure: np.ndarray
    negative_exposure: np.ndarray
    domestic_df_paths: np.ndarray
    foreign_df_paths: np.ndarray
    fx_spot_paths: np.ndarray


# ---------------------------------------------------------------------------
# Public exposure API
# ---------------------------------------------------------------------------


def simulate_market(
    fx_model: FXModel,
    maturity_yrs: float,
    n_steps: int,
    n_paths: int,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    seed: int | None = 42,
    scramble: bool = True,
    rng: np.random.Generator | None = None,
) -> MarketSimulation:
    """Simulate joint market-factor paths once for a scenario batch.

    Args:
        fx_model: Any modular FX model
            (:class:`~xvasim.models.base.FXModel`).
        maturity_yrs: Simulation horizon in years.
        n_steps: Number of uniform simulation time steps.
        n_paths: Number of Monte Carlo paths.
        random_type: Random sequence type (:class:`RandomSequenceType` or str).
        seed: Optional random seed.
        scramble: If True, applies scrambling to QMC sequences.
        rng: Optional pre-configured NumPy generator.

    Returns:
        :class:`MarketSimulation` on a uniform grid
        ``linspace(0, maturity_yrs, n_steps + 1)``.
    """
    if maturity_yrs <= 0.0:
        msg = f"maturity_yrs must be strictly positive, got {maturity_yrs}"
        raise ValueError(msg)
    if n_steps < 1:
        msg = f"n_steps must be at least 1, got {n_steps}"
        raise ValueError(msg)
    if n_paths < 1:
        msg = f"n_paths must be at least 1, got {n_paths}"
        raise ValueError(msg)

    times, dom_states, for_states, fx_spot = fx_model.simulate_paths(
        maturity_yrs=maturity_yrs,
        n_paths=n_paths,
        n_steps=n_steps,
        rng=rng,
        random_type=random_type,
        seed=seed,
        scramble=scramble,
    )

    times_arr = np.asarray(times, dtype=np.float64)
    dom_states_arr = np.asarray(dom_states, dtype=np.float64)
    for_states_arr = np.asarray(for_states, dtype=np.float64)
    fx_spot_arr = np.asarray(fx_spot, dtype=np.float64)

    if isinstance(fx_model, TwoCurrencyFXModel):
        dom_df = np.asarray(
            fx_model.domestic_ir_model.discount_path(times_arr, dom_states_arr),
            dtype=np.float64,
        )
        for_df = np.asarray(
            fx_model.foreign_ir_model.discount_path(times_arr, for_states_arr),
            dtype=np.float64,
        )
    else:
        dom_df = np.tile(
            np.asarray(fx_model.domestic_discount_factor(times_arr), dtype=np.float64),
            (n_paths, 1),
        )
        for_df = np.tile(
            np.asarray(fx_model.foreign_discount_factor(times_arr), dtype=np.float64),
            (n_paths, 1),
        )

    return MarketSimulation(
        times_yrs=times_arr,
        n_paths=n_paths,
        fx_model=fx_model,
        fx_spot_paths=fx_spot_arr,
        domestic_state_paths=dom_states_arr,
        foreign_state_paths=for_states_arr,
        domestic_df_paths=dom_df,
        foreign_df_paths=for_df,
    )


def _check_trade_horizon(sim: MarketSimulation, portfolio: Portfolio) -> None:
    """Ensure every trade maturity stays within the simulation horizon."""
    horizon = float(sim.times_yrs[-1])
    for trade in portfolio.trades:
        if trade.maturity_yrs > horizon + 1e-12:
            msg = (
                f"trade {trade.trade_id!r} maturity {trade.maturity_yrs:.6f} exceeds "
                f"simulation horizon {horizon:.6f}"
            )
            raise ValueError(msg)


def compute_portfolio_exposure(
    portfolio: Portfolio, sim: MarketSimulation
) -> PortfolioExposureResult:
    """Value every trade and aggregate marks into a netting-set exposure.

    Args:
        portfolio: Portfolio / netting set to value.
        sim: Simulated market context (:class:`MarketSimulation`).

    Returns:
        :class:`PortfolioExposureResult` with per-trade marks, net mark,
        positive/negative exposure, and the discount-factor paths.
    """
    _check_trade_horizon(sim, portfolio)

    n_dates = sim.times_yrs.shape[0]
    trade_mtm: dict[str, np.ndarray] = {}
    for trade in portfolio.trades:
        mtm = np.asarray(trade.value_paths(sim), dtype=np.float64)
        if mtm.shape != (sim.n_paths, n_dates):
            msg = (
                f"trade {trade.trade_id!r} returned shape {mtm.shape}, expected "
                f"({sim.n_paths}, {n_dates})"
            )
            raise ValueError(msg)
        trade_mtm[trade.trade_id] = mtm

    if trade_mtm:
        net_mtm: np.ndarray = np.sum(
            list(trade_mtm.values()), axis=0, dtype=np.float64
        )
    else:
        net_mtm = np.zeros((sim.n_paths, n_dates), dtype=np.float64)

    if portfolio.netting:
        exposure = np.maximum(net_mtm, 0.0)
        negative_exposure = np.maximum(-net_mtm, 0.0)
    else:
        if trade_mtm:
            exposure = np.sum(
                [np.maximum(mtm, 0.0) for mtm in trade_mtm.values()],
                axis=0,
                dtype=np.float64,
            )
            negative_exposure = np.sum(
                [np.maximum(-mtm, 0.0) for mtm in trade_mtm.values()],
                axis=0,
                dtype=np.float64,
            )
        else:
            exposure = np.zeros((sim.n_paths, n_dates), dtype=np.float64)
            negative_exposure = np.zeros((sim.n_paths, n_dates), dtype=np.float64)

    return PortfolioExposureResult(
        times_yrs=sim.times_yrs,
        n_paths=sim.n_paths,
        trade_mtm=trade_mtm,
        net_mtm=net_mtm,
        exposure=np.asarray(exposure, dtype=np.float64),
        negative_exposure=np.asarray(negative_exposure, dtype=np.float64),
        domestic_df_paths=sim.domestic_df_paths,
        foreign_df_paths=sim.foreign_df_paths,
        fx_spot_paths=sim.fx_spot_paths,
    )


def simulate_portfolio_exposure(
    portfolio: Portfolio,
    fx_model: FXModel,
    maturity_yrs: float,
    n_steps: int,
    n_paths: int,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    seed: int | None = 42,
    scramble: bool = True,
    rng: np.random.Generator | None = None,
) -> PortfolioExposureResult:
    """Simulate market factors and compute the portfolio exposure in one call.

    Convenience wrapper around :func:`simulate_market` followed by
    :func:`compute_portfolio_exposure`.
    """
    sim = simulate_market(
        fx_model=fx_model,
        maturity_yrs=maturity_yrs,
        n_steps=n_steps,
        n_paths=n_paths,
        random_type=random_type,
        seed=seed,
        scramble=scramble,
        rng=rng,
    )
    return compute_portfolio_exposure(portfolio, sim)


# ---------------------------------------------------------------------------
# End-to-end XVA ledger
# ---------------------------------------------------------------------------


def _marginal_pd_aligned(
    credit_spreads_ann: np.ndarray,
    source_tenors_yrs: np.ndarray,
    target_tenors_yrs: np.ndarray,
) -> np.ndarray:
    """Calibrate a CIR hazard model and return marginal PD on target tenors."""
    spreads_arr = np.asarray(credit_spreads_ann, dtype=np.float64)
    source_arr = np.asarray(source_tenors_yrs, dtype=np.float64)
    target_arr = np.asarray(target_tenors_yrs, dtype=np.float64)

    if spreads_arr.shape != source_arr.shape:
        msg = (
            f"credit_spreads_ann (shape {spreads_arr.shape}) must align with "
            f"credit_tenors_yrs (shape {source_arr.shape})"
        )
        raise ValueError(msg)

    model = CIRHazardRateModel.calibrate_from_spreads(spreads_arr, source_arr)
    surv = model.survival_probability(target_arr)
    return np.diff(1.0 - surv, prepend=0.0)


def compute_portfolio_xva(
    portfolio: Portfolio,
    fx_model: FXModel,
    maturity_yrs: float,
    n_steps: int,
    counterparty_credit_spreads_ann: np.ndarray,
    own_credit_spreads_ann: np.ndarray,
    credit_tenors_yrs: np.ndarray | None = None,
    counterparty_lgd: float = 0.60,
    own_lgd: float = 0.60,
    funding_spread_borrow_ann: float = 0.005,
    funding_spread_deposit_ann: float = 0.0,
    capital_charge_ann: float = 0.08,
    regulatory_lgd: float = 0.75,
    im_scaling: float = 1.0,
    im_percentile: float = 99.0,
    n_paths: int = 10_000,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    seed: int | None = 42,
    scramble: bool = True,
    chunk_size: int | None = None,
    use_numexpr: bool = True,
    percentiles: typing.Sequence[float] = (95.0, 97.5, 99.0),
) -> dict[str, typing.Any]:
    r"""Compute the full XVA ledger for a portfolio in one pass.

    Simulates market factors, values the portfolio (netting set), computes
    counterparty/own marginal default probabilities from credit spreads over
    the CIR hazard-rate model, and aggregates CVA, DVA, FVA (FCA/FBA), KVA,
    and MVA via :func:`xvasim.cva_engine.compute_total_xva`.

    Args:
        portfolio: Portfolio / netting set to simulate.
        fx_model: FX market model driving the simulation.
        maturity_yrs: Simulation horizon in years.
        n_steps: Number of simulation time steps.
        counterparty_credit_spreads_ann: Counterparty CDS-style credit
            spreads (annualised) aligned with *credit_tenors_yrs*.
        own_credit_spreads_ann: The firm's own credit spreads (annualised).
        credit_tenors_yrs: Tenors (years) for the credit spreads. Defaults to
            the simulation dates strictly after ``t=0``, i.e.
            ``len(times_yrs) - 1`` pillars.
        counterparty_lgd: Counterparty loss given default (decimal).
        own_lgd: Firm's own loss given default (decimal).
        funding_spread_borrow_ann: Annualised borrowing spread (decimal).
        funding_spread_deposit_ann: Annualised deposit spread (decimal).
        capital_charge_ann: Annualised regulatory capital charge rate (decimal).
        regulatory_lgd: Regulatory loss given default (decimal).
        im_scaling: Initial-margin scaling factor α (default 1.0).
        im_percentile: Percentile confidence level for PFE-based IM (default 99.0).
        n_paths: Number of Monte Carlo paths.
        random_type: Random sequence type (:class:`RandomSequenceType` or str).
        seed: Optional random seed.
        scramble: If True, applies scrambling to QMC sequences.
        chunk_size: Optional path batch size for chunked aggregation.
        use_numexpr: If True, accelerates aggregation with numexpr.
        percentiles: Percentile levels for the PFE exposure profile.

    Returns:
        Dictionary with the XVA ledger keys (``"cva"``, ``"dva"``,
        ``"fca"``, ``"fba"``, ``"fva"``, ``"kva"``, ``"mva"``,
        ``"total_xva"``) plus simulation objects:

        - ``"times_yrs"`` — exposure date grid.
        - ``"net_mtm"`` / ``"exposure"`` / ``"negative_exposure"`` — signed
          net mark, positive credit exposure, and negative exposure arrays.
        - ``"trade_mtm"`` — per-trade mark-to-market mapping.
        - ``"domestic_df_paths"`` — domestic discount-factor paths.
        - Exposure-profile keys from
          :func:`xvasim.cva_engine.compute_exposure_profile`
          (``"expected_exposure"``, ``"epe"``, ``"max_pfe"``, ``"pfe_*"``)
          computed on the positive exposure grid ``t > 0``.
    """
    sim_res = simulate_portfolio_exposure(
        portfolio=portfolio,
        fx_model=fx_model,
        maturity_yrs=maturity_yrs,
        n_steps=n_steps,
        n_paths=n_paths,
        random_type=random_type,
        seed=seed,
        scramble=scramble,
    )

    times_arr = sim_res.times_yrs
    n_dates = times_arr.shape[0]
    if n_dates < 2:
        msg = "t >= 1 for the exposure grid."
        raise ValueError(msg)

    if credit_tenors_yrs is None:
        credit_tenors = times_arr[1:]
    else:
        credit_tenors = np.asarray(credit_tenors_yrs, dtype=np.float64)

    exposure_steps = sim_res.exposure[:, 1:]
    df_steps = np.asarray(sim_res.domestic_df_paths[:, 1:], dtype=np.float64)
    dt_steps = np.diff(times_arr)

    cp_pd = _marginal_pd_aligned(
        counterparty_credit_spreads_ann, credit_tenors, times_arr[1:]
    )
    own_pd = _marginal_pd_aligned(
        own_credit_spreads_ann, credit_tenors, times_arr[1:]
    )
    cp_pd_mat = np.tile(cp_pd, (n_paths, 1))
    own_pd_mat = np.tile(own_pd, (n_paths, 1))

    xva_ledger = compute_total_xva(
        exposure=exposure_steps,
        time_steps_yrs=dt_steps,
        discount_factor=df_steps,
        counterparty_marginal_pd=cp_pd_mat,
        own_marginal_pd=own_pd_mat,
        counterparty_lgd=counterparty_lgd,
        own_lgd=own_lgd,
        funding_spread_borrow_ann=funding_spread_borrow_ann,
        funding_spread_deposit_ann=funding_spread_deposit_ann,
        capital_charge_ann=capital_charge_ann,
        regulatory_lgd=regulatory_lgd,
        im_scaling=im_scaling,
        im_percentile=im_percentile,
        chunk_size=chunk_size,
        use_numexpr=use_numexpr,
    )

    profile = compute_exposure_profile(exposure_steps, percentiles)

    result: dict[str, typing.Any] = {
        **xva_ledger,
        "times_yrs": times_arr,
        "net_mtm": sim_res.net_mtm,
        "exposure": sim_res.exposure,
        "negative_exposure": sim_res.negative_exposure,
        "trade_mtm": sim_res.trade_mtm,
        "domestic_df_paths": sim_res.domestic_df_paths,
        "foreign_df_paths": sim_res.foreign_df_paths,
        **profile,
    }
    return result
