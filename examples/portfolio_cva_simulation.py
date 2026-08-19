"""Multi-Asset Portfolio CVA Simulation Example.

This example demonstrates how to:
1. Build and calibrate multi-asset market models (Domestic IR, Foreign IR, Spot FX).
2. Construct a diverse portfolio of derivatives (IRS, FX Forwards, FX Options, XCCY).
3. Simulate correlated stochastic state paths and discount factors via QMC (Sobol).
4. Perform path-wise Mark-to-Future valuation across a joint simulation grid.
5. Aggregate portfolio exposures under bilateral netting agreements.
6. Calibrate a CIR Hazard Rate credit model to counterparty CDS spreads.
7. Compute CVA, quantify Netting Benefits, and report risk metrics.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
from scipy.stats import norm

from xvasim.cva_engine import compute_cva
from xvasim.models.credit.cir import CIRHazardRateModel
from xvasim.models.fx.two_currency import TwoCurrencyFXModel
from xvasim.models.ir.lgm import LGMModel
from xvasim.qmc import RandomSequenceType

# ===========================================================================
# 1. Trade & Portfolio Abstractions
# ===========================================================================


class Trade:
    """Base class for derivative contracts in a bilateral netting set."""

    def __init__(self, trade_id: str, trade_type: str, maturity_yrs: float) -> None:
        self.trade_id = trade_id
        self.trade_type = trade_type
        self.maturity_yrs = maturity_yrs

    def value_at(
        self,
        t: float,
        dom_ir: LGMModel,
        for_ir: LGMModel,
        x_dom: np.ndarray,
        x_for: np.ndarray,
        spot_fx: np.ndarray,
    ) -> np.ndarray:
        """Evaluate path-wise present value at future time t conditional on state.

        Args:
            t: Valuation time in years.
            dom_ir: Domestic interest rate model.
            for_ir: Foreign interest rate model.
            x_dom: Domestic state variable vector of shape (n_paths,).
            x_for: Foreign state variable vector of shape (n_paths,).
            spot_fx: Simulated FX spot rate vector of shape (n_paths,).

        Returns:
            1-D array of path-wise NPVs in domestic currency.
        """
        raise NotImplementedError


class InterestRateSwapTrade(Trade):
    """Vanilla fixed-for-floating Interest Rate Swap (IRS) in domestic currency."""

    def __init__(
        self,
        trade_id: str,
        notional: float,
        fixed_rate_ann: float,
        tenor_yrs: float,
        pay_freq_yrs: float = 0.5,
        spread_ann: float = 0.0,
        is_payer: bool = True,
    ) -> None:
        super().__init__(
            trade_id=trade_id,
            trade_type="IRS (Payer)" if is_payer else "IRS (Receiver)",
            maturity_yrs=tenor_yrs,
        )
        self.notional = notional
        self.fixed_rate_ann = fixed_rate_ann
        self.tenor_yrs = tenor_yrs
        self.pay_freq_yrs = pay_freq_yrs
        self.spread_ann = spread_ann
        self.is_payer = is_payer

        n_periods = max(1, int(np.round(tenor_yrs / pay_freq_yrs)))
        self.pay_times = np.array(
            [(k + 1) * pay_freq_yrs for k in range(n_periods)],
            dtype=np.float64,
        )

    def value_at(
        self,
        t: float,
        dom_ir: LGMModel,
        for_ir: LGMModel,
        x_dom: np.ndarray,
        x_for: np.ndarray,
        spot_fx: np.ndarray,
    ) -> np.ndarray:
        n_paths = len(x_dom)
        if t >= self.tenor_yrs:
            return np.zeros(n_paths, dtype=np.float64)

        rem_pays = self.pay_times[self.pay_times > t]
        if len(rem_pays) == 0:
            return np.zeros(n_paths, dtype=np.float64)

        # Annuity A(t) = sum tau_j * P(t, T_j)
        annuity = np.zeros(n_paths, dtype=np.float64)
        for tp in rem_pays:
            annuity += self.pay_freq_yrs * dom_ir.zero_coupon_bond(t, tp, x_dom)

        # Terminal zero coupon bond
        p_terminal = dom_ir.zero_coupon_bond(t, rem_pays[-1], x_dom)

        # Clean floating leg PV = 1.0 - P(t, T_terminal) + spread * Annuity
        pv_float = self.notional * (1.0 - p_terminal + self.spread_ann * annuity)
        pv_fixed = self.notional * self.fixed_rate_ann * annuity

        if self.is_payer:
            return pv_float - pv_fixed
        return pv_fixed - pv_float


class FXForwardTrade(Trade):
    """Outright Foreign Exchange forward contract."""

    def __init__(
        self,
        trade_id: str,
        notional_foreign: float,
        strike: float,
        maturity_yrs: float,
    ) -> None:
        super().__init__(
            trade_id=trade_id,
            trade_type="FX Forward",
            maturity_yrs=maturity_yrs,
        )
        self.notional_foreign = notional_foreign
        self.strike = strike

    def value_at(
        self,
        t: float,
        dom_ir: LGMModel,
        for_ir: LGMModel,
        x_dom: np.ndarray,
        x_for: np.ndarray,
        spot_fx: np.ndarray,
    ) -> np.ndarray:
        n_paths = len(x_dom)
        if t >= self.maturity_yrs:
            return np.zeros(n_paths, dtype=np.float64)

        p_for = for_ir.zero_coupon_bond(t, self.maturity_yrs, x_for)
        p_dom = dom_ir.zero_coupon_bond(t, self.maturity_yrs, x_dom)

        # PV = N_for * (S(t) * P_f(t, T) - K * P_d(t, T))
        return self.notional_foreign * (spot_fx * p_for - self.strike * p_dom)


class FXOptionTrade(Trade):
    """European FX option priced conditionally via Garman-Kohlhagen."""

    def __init__(
        self,
        trade_id: str,
        notional_foreign: float,
        strike: float,
        maturity_yrs: float,
        fx_vol_ann: float,
        is_call: bool = True,
    ) -> None:
        super().__init__(
            trade_id=trade_id,
            trade_type="FX Call" if is_call else "FX Put",
            maturity_yrs=maturity_yrs,
        )
        self.notional_foreign = notional_foreign
        self.strike = strike
        self.fx_vol_ann = fx_vol_ann
        self.is_call = is_call

    def value_at(
        self,
        t: float,
        dom_ir: LGMModel,
        for_ir: LGMModel,
        x_dom: np.ndarray,
        x_for: np.ndarray,
        spot_fx: np.ndarray,
    ) -> np.ndarray:
        if t >= self.maturity_yrs:
            # Expired / terminal intrinsic payoff
            if self.is_call:
                intrinsic = np.maximum(spot_fx - self.strike, 0.0)
            else:
                intrinsic = np.maximum(self.strike - spot_fx, 0.0)
            return self.notional_foreign * intrinsic

        p_for = for_ir.zero_coupon_bond(t, self.maturity_yrs, x_for)
        p_dom = dom_ir.zero_coupon_bond(t, self.maturity_yrs, x_dom)

        tau = self.maturity_yrs - t
        vol_sqrt_tau = self.fx_vol_ann * np.sqrt(tau)

        forward_fx = spot_fx * (p_for / np.maximum(p_dom, 1e-18))
        d1 = (np.log(forward_fx / self.strike) + 0.5 * vol_sqrt_tau**2) / vol_sqrt_tau
        d2 = d1 - vol_sqrt_tau

        if self.is_call:
            unit_price = p_dom * (
                forward_fx * norm.cdf(d1) - self.strike * norm.cdf(d2)
            )
        else:
            unit_price = p_dom * (
                self.strike * norm.cdf(-d2) - forward_fx * norm.cdf(-d1)
            )

        return self.notional_foreign * unit_price


class CrossCurrencySwapTrade(Trade):
    """Cross-Currency Swap (XCCY): Fixed Domestic vs. Floating Foreign + Notional."""

    def __init__(
        self,
        trade_id: str,
        foreign_notional: float,
        spot_ref: float,
        domestic_rate_ann: float,
        foreign_spread_ann: float,
        tenor_yrs: float,
        pay_freq_yrs: float = 0.5,
        is_domestic_payer: bool = True,
    ) -> None:
        super().__init__(
            trade_id=trade_id,
            trade_type="XCCY Swap",
            maturity_yrs=tenor_yrs,
        )
        self.foreign_notional = foreign_notional
        self.domestic_notional = foreign_notional * spot_ref
        self.domestic_rate_ann = domestic_rate_ann
        self.foreign_spread_ann = foreign_spread_ann
        self.tenor_yrs = tenor_yrs
        self.pay_freq_yrs = pay_freq_yrs
        self.is_domestic_payer = is_domestic_payer

        n_periods = max(1, int(np.round(tenor_yrs / pay_freq_yrs)))
        self.pay_times = np.array(
            [(k + 1) * pay_freq_yrs for k in range(n_periods)],
            dtype=np.float64,
        )

    def value_at(
        self,
        t: float,
        dom_ir: LGMModel,
        for_ir: LGMModel,
        x_dom: np.ndarray,
        x_for: np.ndarray,
        spot_fx: np.ndarray,
    ) -> np.ndarray:
        n_paths = len(x_dom)
        if t >= self.tenor_yrs:
            return np.zeros(n_paths, dtype=np.float64)

        rem_pays = self.pay_times[self.pay_times > t]
        if len(rem_pays) == 0:
            return np.zeros(n_paths, dtype=np.float64)

        # Domestic fixed leg PV
        annuity_dom = np.zeros(n_paths, dtype=np.float64)
        for tp in rem_pays:
            annuity_dom += self.pay_freq_yrs * dom_ir.zero_coupon_bond(t, tp, x_dom)
        pv_dom_leg = self.domestic_notional * self.domestic_rate_ann * annuity_dom

        # Foreign floating leg PV (in EUR)
        annuity_for = np.zeros(n_paths, dtype=np.float64)
        for tp in rem_pays:
            annuity_for += self.pay_freq_yrs * for_ir.zero_coupon_bond(t, tp, x_for)
        p_for_term = for_ir.zero_coupon_bond(t, rem_pays[-1], x_for)
        pv_for_leg_eur = self.foreign_notional * (
            1.0 - p_for_term + self.foreign_spread_ann * annuity_for
        )

        # Foreign leg in domestic currency
        pv_for_leg_dom = spot_fx * pv_for_leg_eur

        # Terminal notional exchange: Receive foreign notional, pay domestic notional
        p_dom_term = dom_ir.zero_coupon_bond(t, rem_pays[-1], x_dom)
        pv_notional_exchange = (
            self.foreign_notional * spot_fx * p_for_term
            - self.domestic_notional * p_dom_term
        )

        if self.is_domestic_payer:
            # Pay domestic, receive foreign
            return pv_for_leg_dom - pv_dom_leg + pv_notional_exchange
        return pv_dom_leg - pv_for_leg_dom - pv_notional_exchange


# ===========================================================================
# 2. Portfolio Container & Simulation Analytics
# ===========================================================================


@dataclasses.dataclass
class PortfolioExposureResult:
    """Simulation results containing exposure profiles and CVA metrics."""

    sim_times: np.ndarray
    expected_exposure_netted: np.ndarray
    expected_exposure_gross: np.ndarray
    pfe_95_netted: np.ndarray
    pfe_99_netted: np.ndarray
    marginal_pds: np.ndarray
    cva_netted: float
    cva_gross: float
    netting_benefit_amount: float
    netting_benefit_pct: float
    trade_summary: list[dict[str, typing.Any]]


class Portfolio:
    """A portfolio of multi-asset derivative securities with a single counterparty."""

    def __init__(self, trades: list[Trade] | None = None) -> None:
        self.trades: list[Trade] = list(trades) if trades is not None else []

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the netting set."""
        self.trades.append(trade)

    def run_cva_simulation(
        self,
        fx_market_model: TwoCurrencyFXModel,
        counterparty_cds_spreads_ann: np.ndarray,
        counterparty_cds_tenors_yrs: np.ndarray,
        loss_given_default: float = 0.60,
        horizon_yrs: float = 7.0,
        n_steps: int = 28,
        n_paths: int = 2048,
        random_type: RandomSequenceType | str = RandomSequenceType.SOBOL,
        seed: int = 42,
    ) -> PortfolioExposureResult:
        """Simulate portfolio mark-to-future paths and compute CVA with netting.

        Args:
            fx_market_model: Joint two-currency stochastic market model.
            counterparty_cds_spreads_ann: Market CDS spreads across tenors.
            counterparty_cds_tenors_yrs: Tenors for the credit spreads (years).
            loss_given_default: Loss Given Default (decimal, e.g. 0.60).
            horizon_yrs: Simulation horizon in years.
            n_steps: Number of simulation time steps.
            n_paths: Number of Monte Carlo / QMC paths.
            random_type: QMC sequence generator type (default: SOBOL).
            seed: Simulation seed.

        Returns:
            PortfolioExposureResult dataclass with exposure curves and CVA.
        """
        dom_ir = typing.cast(LGMModel, fx_market_model.domestic_ir_model)
        for_ir = typing.cast(LGMModel, fx_market_model.foreign_ir_model)

        # 1. Joint multi-asset path simulation
        sim_times, x_dom, x_for, spot_fx = fx_market_model.simulate_paths(
            maturity_yrs=horizon_yrs,
            n_paths=n_paths,
            n_steps=n_steps,
            random_type=random_type,
            seed=seed,
        )

        # Path-wise domestic discount factors D(0, t_k)
        disc_paths = dom_ir.discount_path(sim_times, x_dom)

        n_times = len(sim_times)
        trade_pvs: list[np.ndarray] = []
        trade_summary: list[dict[str, typing.Any]] = []

        # 2. Mark-to-Future path-wise valuation for each trade
        for trade in self.trades:
            val_grid = np.zeros((n_paths, n_times), dtype=np.float64)
            for k in range(n_times):
                t = float(sim_times[k])
                val_grid[:, k] = trade.value_at(
                    t=t,
                    dom_ir=dom_ir,
                    for_ir=for_ir,
                    x_dom=x_dom[:, k],
                    x_for=x_for[:, k],
                    spot_fx=spot_fx[:, k],
                )

            trade_pvs.append(val_grid)
            t0_npv = float(np.mean(val_grid[:, 0]))
            trade_summary.append(
                {
                    "id": trade.trade_id,
                    "type": trade.trade_type,
                    "maturity_yrs": trade.maturity_yrs,
                    "t0_npv": t0_npv,
                }
            )

        # 3. Bilateral Netting vs. Gross Exposure
        # Portfolio net NPV V_net(t, w) = sum_m V_m(t, w)
        portfolio_net_pvs = np.sum(trade_pvs, axis=0)
        netted_exposure = np.maximum(portfolio_net_pvs, 0.0)

        # Gross unnetted exposure = sum_m max(V_m(t, w), 0)
        gross_exposure = np.zeros((n_paths, n_times), dtype=np.float64)
        for tpv in trade_pvs:
            gross_exposure += np.maximum(tpv, 0.0)

        # Exposure summary metrics across time
        ee_netted = np.mean(netted_exposure, axis=0)
        ee_gross = np.mean(gross_exposure, axis=0)
        pfe_95 = np.percentile(netted_exposure, 95.0, axis=0)
        pfe_99 = np.percentile(netted_exposure, 99.0, axis=0)

        # 4. Calibrate Counterparty CIR Hazard Rate Model & Compute Marginal PDs
        cir_credit = CIRHazardRateModel.calibrate_from_spreads(
            credit_spreads_ann=counterparty_cds_spreads_ann,
            tenors_yrs=counterparty_cds_tenors_yrs,
        )

        eval_tenors = sim_times[1:]  # evaluation steps > 0
        marginal_pds = cir_credit.marginal_pd(eval_tenors)

        # 5. CVA Aggregation using library CVA Engine
        exp_net_eval = netted_exposure[:, 1:]
        exp_gross_eval = gross_exposure[:, 1:]
        df_eval = disc_paths[:, 1:]

        cva_netted = compute_cva(
            exposure=exp_net_eval,
            marginal_pd=marginal_pds,
            discount_factor=df_eval,
            loss_given_default=loss_given_default,
        )

        cva_gross = compute_cva(
            exposure=exp_gross_eval,
            marginal_pd=marginal_pds,
            discount_factor=df_eval,
            loss_given_default=loss_given_default,
        )

        netting_benefit_amount = cva_gross - cva_netted
        netting_benefit_pct = (
            (netting_benefit_amount / cva_gross) * 100.0 if cva_gross > 0 else 0.0
        )

        return PortfolioExposureResult(
            sim_times=sim_times,
            expected_exposure_netted=ee_netted,
            expected_exposure_gross=ee_gross,
            pfe_95_netted=pfe_95,
            pfe_99_netted=pfe_99,
            marginal_pds=marginal_pds,
            cva_netted=cva_netted,
            cva_gross=cva_gross,
            netting_benefit_amount=netting_benefit_amount,
            netting_benefit_pct=netting_benefit_pct,
            trade_summary=trade_summary,
        )


# ===========================================================================
# 3. Main Executable Example Workflow
# ===========================================================================


def run_example() -> None:
    """Execute complete portfolio generation and CVA calculation workflow."""
    print("=" * 80)
    print("  XvaSim: Multi-Asset Portfolio CVA Simulation & Risk Analytics")
    print("=" * 80)

    # 1. Market Curves Setup
    curve_tenors = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])
    usd_dfs = np.exp(-0.035 * curve_tenors)  # USD Base 3.50%
    eur_dfs = np.exp(-0.020 * curve_tenors)  # EUR Base 2.00%

    print("\n[1] Market Environment Setup:")
    print("    - Domestic Yield Curve (USD): Flat 3.50% p.a.")
    print("    - Foreign Yield Curve (EUR):  Flat 2.00% p.a.")
    print("    - Spot FX Rate (EUR/USD):     1.1000")
    print("    - FX Spot Volatility:         11.00% p.a.")

    # 2. Setup Stochastic Models
    usd_lgm = LGMModel(
        kappa_ann=0.03,
        sigma_grid_yrs=np.array([2.0, 5.0, 10.0]),
        sigma_values_ann=np.array([0.0090, 0.0100, 0.0115]),
        discount_curve_yrs=curve_tenors,
        discount_factors=usd_dfs,
    )

    eur_lgm = LGMModel(
        kappa_ann=0.025,
        sigma_grid_yrs=np.array([2.0, 5.0, 10.0]),
        sigma_values_ann=np.array([0.0075, 0.0085, 0.0095]),
        discount_curve_yrs=curve_tenors,
        discount_factors=eur_dfs,
    )

    # 3x3 Correlation: [USD_Rate, EUR_Rate, EUR/USD_Spot]
    corr_matrix = np.array(
        [
            [1.00, 0.40, 0.15],
            [0.40, 1.00, -0.20],
            [0.15, -0.20, 1.00],
        ]
    )

    fx_market_model = TwoCurrencyFXModel(
        domestic_ir_model=usd_lgm,
        foreign_ir_model=eur_lgm,
        spot_fx=1.10,
        fx_vol_ann=0.11,
        correlation_matrix=corr_matrix,
    )

    # 3. Construct Counterparty Portfolio
    portfolio = Portfolio(
        [
            InterestRateSwapTrade(
                trade_id="IRS-01",
                notional=10_000_000.0,  # $10M
                fixed_rate_ann=0.035,  # 3.50%
                tenor_yrs=5.0,
                pay_freq_yrs=0.5,
                is_payer=True,
            ),
            InterestRateSwapTrade(
                trade_id="IRS-02",
                notional=6_000_000.0,  # $6M
                fixed_rate_ann=0.036,  # 3.60%
                tenor_yrs=7.0,
                pay_freq_yrs=0.5,
                is_payer=False,
            ),
            CrossCurrencySwapTrade(
                trade_id="XCCY-01",
                foreign_notional=5_000_000.0,  # EUR 5M (~ $5.5M)
                spot_ref=1.10,
                domestic_rate_ann=0.034,  # Pay USD 3.4% fixed
                foreign_spread_ann=0.0020,  # Receive EUR float + 20 bps
                tenor_yrs=3.0,
                pay_freq_yrs=0.5,
                is_domestic_payer=True,
            ),
            FXForwardTrade(
                trade_id="FX-FWD-01",
                notional_foreign=3_000_000.0,  # EUR 3M
                strike=1.1150,
                maturity_yrs=2.0,
            ),
            FXOptionTrade(
                trade_id="FX-OPT-01",
                notional_foreign=2_000_000.0,  # EUR 2M
                strike=1.1200,
                maturity_yrs=1.5,
                fx_vol_ann=0.11,
                is_call=True,
            ),
        ]
    )

    # 4. Counterparty Credit Curve (CDS Spreads)
    cds_tenors = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    cds_spreads = np.array([0.0120, 0.0140, 0.0165, 0.0190, 0.0230, 0.0260, 0.0285])
    lgd = 0.60  # 40% Recovery

    spreads_str = ", ".join(f"{s*10000:.0f}" for s in cds_spreads)
    tenors_str = ", ".join(f"{t:.1f}" for t in cds_tenors)
    print("\n[2] Counterparty Credit Risk Profile:")
    print(f"    - CDS Spreads (bps): {spreads_str}")
    print(f"    - CDS Tenors (yrs):   {tenors_str}")
    print(f"    - Loss Given Default: {lgd*100:.0f}% (Recovery: {(1-lgd)*100:.0f}%)")

    # 5. Execute Simulation & Aggregation
    n_paths = 2048
    horizon_yrs = 7.0
    n_steps = 28  # Quarterly grid

    print(f"\n[3] Running Simulation ({n_paths} Sobol QMC paths, {n_steps} steps)...")
    res = portfolio.run_cva_simulation(
        fx_market_model=fx_market_model,
        counterparty_cds_spreads_ann=cds_spreads,
        counterparty_cds_tenors_yrs=cds_tenors,
        loss_given_default=lgd,
        horizon_yrs=horizon_yrs,
        n_steps=n_steps,
        random_type=RandomSequenceType.SOBOL,
        seed=42,
    )

    # 6. Display Trade Breakdown
    print("\n[4] Portfolio Trades Summary at Inception (t=0):")
    print("    " + "-" * 72)
    print(
        f"    {'Trade ID':<12} {'Type':<18} {'Tenor':<10} {'T0 Net PV (USD)':>20}"
    )
    print("    " + "-" * 72)
    total_t0_npv = 0.0
    for ts in res.trade_summary:
        print(
            f"    {ts['id']:<12} {ts['type']:<18} {ts['maturity_yrs']:>4.1f}y     "
            f"${ts['t0_npv']:>18,.2f}"
        )
        total_t0_npv += ts["t0_npv"]
    print("    " + "-" * 72)
    print(f"    {'TOTAL PORTFOLIO NPV':<36} ${total_t0_npv:>18,.2f}")

    # 7. Display Exposure Grid Sample
    print("\n[5] Exposure Profile Evolution Across Time:")
    print("    " + "-" * 72)
    print(
        f"    {'Time (y)':<10} {'EE Netted ($)':>16} {'EE Gross ($)':>16} "
        f"{'PFE 95% ($)':>16} {'Marginal PD':>10}"
    )
    print("    " + "-" * 72)
    sample_indices = [0, 2, 4, 8, 12, 20, 28]
    for idx in sample_indices:
        t = res.sim_times[idx]
        ee_net = res.expected_exposure_netted[idx]
        ee_grs = res.expected_exposure_gross[idx]
        pfe = res.pfe_95_netted[idx]
        mpd_str = f"{res.marginal_pds[idx-1]:.4f}" if idx > 0 else "-"
        print(
            f"    {t:<10.2f} ${ee_net:>14,.2f} ${ee_grs:>14,.2f} "
            f"${pfe:>14,.2f} {mpd_str:>10}"
        )
    print("    " + "-" * 72)

    # 8. Display CVA & Netting Results
    max_pfe = float(np.max(res.pfe_95_netted))
    print("\n[6] CVA & Credit Risk Valuation Summary:")
    print("    " + "=" * 50)
    print(f"    Netted Portfolio CVA:        ${res.cva_netted:>14,.2f}")
    print(f"    Gross (Unnetted) CVA:        ${res.cva_gross:>14,.2f}")
    print(f"    Netting Benefit (Savings):   ${res.netting_benefit_amount:>14,.2f}")
    print(f"    Netting Reduction Ratio:      {res.netting_benefit_pct:>13.2f}%")
    print(f"    Maximum PFE (95% Confidence):${max_pfe:>14,.2f}")
    print("    " + "=" * 50)
    print("\n[OK] Portfolio CVA simulation completed successfully.")


if __name__ == "__main__":
    run_example()
