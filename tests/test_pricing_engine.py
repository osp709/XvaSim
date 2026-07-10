"""Tests for the LGM-based Monte Carlo FX pricing engine."""

import unittest

import numpy as np

from xvasim.pricing_engine import (
    FXLGMParams,
    LGMParams,
    OptionType,
    _compute_h_function,
    _compute_zeta,
    _interpolate_discount_factor,
    calibrate_lgm_to_swaptions,
    price_fx_forward,
    price_fx_option,
)


def _flat_lgm_params(
    rate_ann: float = 0.03,
    kappa: float = 0.03,
    sigma: float = 0.01,
) -> LGMParams:
    """Build LGM params with a flat discount curve and constant vol."""
    tenors = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 30.0])
    dfs = np.exp(-rate_ann * tenors)
    return LGMParams(
        kappa_ann=kappa,
        sigma_grid_yrs=np.array([30.0]),
        sigma_values_ann=np.array([sigma]),
        discount_curve_yrs=tenors,
        discount_factors=dfs,
    )


def _flat_fx_params(
    spot: float = 1.10,
    dom_rate: float = 0.03,
    for_rate: float = 0.01,
    fx_vol: float = 0.10,
    kappa: float = 0.03,
    sigma: float = 0.01,
) -> FXLGMParams:
    """Build FXLGMParams with flat curves, constant vols, no correlation."""
    dom = _flat_lgm_params(rate_ann=dom_rate, kappa=kappa, sigma=sigma)
    foreign = _flat_lgm_params(rate_ann=for_rate, kappa=kappa, sigma=sigma)
    corr = np.eye(3)
    return FXLGMParams(
        domestic=dom,
        foreign=foreign,
        spot_fx=spot,
        fx_vol_ann=fx_vol,
        correlation_matrix=corr,
    )


# -----------------------------------------------------------------------
# Helper function tests
# -----------------------------------------------------------------------


class TestComputeHFunction(unittest.TestCase):
    def test_at_zero(self) -> None:
        """H(0) should be 0 for any kappa."""
        h = _compute_h_function(0.0, kappa=0.05)
        np.testing.assert_allclose(h, 0.0, atol=1e-14)

    def test_limit_kappa_zero(self) -> None:
        """When kappa → 0, H(t) → t."""
        t = np.array([1.0, 2.0, 5.0])
        h = _compute_h_function(t, kappa=0.0)
        np.testing.assert_allclose(h, t, atol=1e-12)

    def test_known_value(self) -> None:
        """H(1) with kappa=0.05 = (1 - exp(-0.05)) / 0.05."""
        expected = (1.0 - np.exp(-0.05)) / 0.05
        h = _compute_h_function(1.0, kappa=0.05)
        np.testing.assert_allclose(h, expected, rtol=1e-12)

    def test_monotonically_increasing(self) -> None:
        """H(t) should increase with t."""
        t = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        h = _compute_h_function(t, kappa=0.05)
        self.assertTrue(np.all(np.diff(h) > 0))


class TestComputeZeta(unittest.TestCase):
    def test_at_zero(self) -> None:
        """ζ(0) = 0."""
        z = _compute_zeta(
            0.0,
            sigma_grid_yrs=np.array([1.0]),
            sigma_values_ann=np.array([0.01]),
            kappa=0.05,
        )
        self.assertAlmostEqual(z, 0.0, places=14)

    def test_positive_for_positive_t(self) -> None:
        """ζ(t) > 0 for t > 0."""
        z = _compute_zeta(
            2.0,
            sigma_grid_yrs=np.array([5.0]),
            sigma_values_ann=np.array([0.01]),
            kappa=0.05,
        )
        self.assertGreater(z, 0.0)

    def test_increases_with_time(self) -> None:
        """ζ should increase with t."""
        grid = np.array([10.0])
        vals = np.array([0.01])
        z1 = _compute_zeta(1.0, grid, vals, kappa=0.05)
        z2 = _compute_zeta(5.0, grid, vals, kappa=0.05)
        self.assertGreater(z2, z1)

    def test_kappa_zero_limit(self) -> None:
        """When κ → 0, ζ(t) → σ² t."""
        sigma = 0.01
        t = 3.0
        z = _compute_zeta(
            t,
            sigma_grid_yrs=np.array([10.0]),
            sigma_values_ann=np.array([sigma]),
            kappa=0.0,
        )
        np.testing.assert_allclose(z, sigma**2 * t, rtol=1e-10)

    def test_t_before_end_of_multi_segment_grid(self) -> None:
        """Test _compute_zeta where t is smaller than the grid max, hitting the early break."""
        grid = np.array([1.0, 2.0, 5.0])
        vals = np.array([0.01, 0.02, 0.03])
        # t = 1.5. At index 2, s_start (1.5) >= t (1.5), so it should break.
        z = _compute_zeta(1.5, grid, vals, kappa=0.05)
        self.assertGreater(z, 0.0)


class TestInterpolateDiscountFactor(unittest.TestCase):
    def test_exact_pillar(self) -> None:
        """At an exact pillar point the DF should match."""
        tenors = np.array([0.0, 1.0, 2.0, 5.0])
        dfs = np.exp(-0.03 * tenors)
        result = _interpolate_discount_factor(1.0, tenors, dfs)
        np.testing.assert_allclose(result, np.exp(-0.03), rtol=1e-10)

    def test_flat_extrapolation(self) -> None:
        """Beyond the last pillar, np.interp flat-extrapolates."""
        tenors = np.array([0.0, 1.0, 5.0])
        dfs = np.exp(-0.03 * tenors)
        result = _interpolate_discount_factor(10.0, tenors, dfs)
        np.testing.assert_allclose(result, np.exp(-0.03 * 5.0), rtol=1e-10)

    def test_df_at_zero(self) -> None:
        """D(0) = 1.0 for a well-formed curve."""
        tenors = np.array([0.0, 1.0, 5.0])
        dfs = np.exp(-0.03 * tenors)
        result = _interpolate_discount_factor(0.0, tenors, dfs)
        np.testing.assert_allclose(result, 1.0, atol=1e-12)


# -----------------------------------------------------------------------
# Forward pricing tests
# -----------------------------------------------------------------------


class TestPriceFxForward(unittest.TestCase):
    def test_atm_forward_price_near_zero(self) -> None:
        """An ATM forward (K = F) should have a price near zero."""
        params = _flat_fx_params(
            spot=1.10, dom_rate=0.03, for_rate=0.01, fx_vol=0.10, sigma=0.0001
        )
        T = 1.0
        # Analytical forward: F = S₀ Pf/Pd
        fwd = params.spot_fx * np.exp(-(0.01 - 0.03) * T)

        result = price_fx_forward(
            params,
            strike=fwd,
            maturity_yrs=T,
            notional=1.0,
            n_paths=200_000,
            n_steps=100,
            seed=123,
        )
        # Price should be close to zero (within MC noise)
        self.assertAlmostEqual(result["price"], 0.0, delta=0.005)

    def test_forward_convergence_to_analytical(self) -> None:
        """MC forward price should converge to analytical PV."""
        params = _flat_fx_params(
            spot=1.10, dom_rate=0.03, for_rate=0.01, fx_vol=0.10, sigma=0.0001
        )
        T = 1.0
        K = 1.05
        fwd = params.spot_fx * np.exp(-(0.01 - 0.03) * T)
        # Analytical PV = (F - K) × Pd(0,T)
        analytical = (fwd - K) * np.exp(-0.03 * T)

        result = price_fx_forward(
            params,
            strike=K,
            maturity_yrs=T,
            notional=1.0,
            n_paths=200_000,
            n_steps=100,
            seed=456,
        )
        # Should be within a few std errors
        self.assertAlmostEqual(result["price"], analytical, delta=0.01)


# -----------------------------------------------------------------------
# Option pricing tests
# -----------------------------------------------------------------------


class TestPriceFxOption(unittest.TestCase):
    def test_call_non_negative(self) -> None:
        """Option price should always be non-negative."""
        params = _flat_fx_params(fx_vol=0.10, sigma=0.0001)
        result = price_fx_option(
            params,
            strike=1.15,
            maturity_yrs=1.0,
            notional=1.0,
            option_type=OptionType.CALL,
            n_paths=50_000,
            n_steps=50,
            seed=1,
        )
        self.assertGreaterEqual(result["price"], 0.0)

    def test_put_non_negative(self) -> None:
        """Put price should be non-negative."""
        params = _flat_fx_params(fx_vol=0.10, sigma=0.0001)
        result = price_fx_option(
            params,
            strike=1.05,
            maturity_yrs=1.0,
            notional=1.0,
            option_type=OptionType.PUT,
            n_paths=50_000,
            n_steps=50,
            seed=2,
        )
        self.assertGreaterEqual(result["price"], 0.0)

    def test_put_call_parity(self) -> None:
        """Put-call parity: C - P = PV(F - K)."""
        params = _flat_fx_params(
            spot=1.10, dom_rate=0.03, for_rate=0.01, fx_vol=0.10, sigma=0.0001
        )
        T = 1.0
        K = 1.08

        call = price_fx_option(
            params,
            strike=K,
            maturity_yrs=T,
            notional=1.0,
            option_type=OptionType.CALL,
            n_paths=200_000,
            n_steps=100,
            seed=99,
        )
        put = price_fx_option(
            params,
            strike=K,
            maturity_yrs=T,
            notional=1.0,
            option_type=OptionType.PUT,
            n_paths=200_000,
            n_steps=100,
            seed=99,
        )

        fwd = params.spot_fx * np.exp(-(0.01 - 0.03) * T)
        pv_fwd_minus_k = (fwd - K) * np.exp(-0.03 * T)

        # C - P ≈ PV(F - K)
        diff = call["price"] - put["price"]
        self.assertAlmostEqual(diff, pv_fwd_minus_k, delta=0.01)

    def test_invalid_option_type_raises(self) -> None:
        """Passing an invalid option type string or type should raise errors."""
        params = _flat_fx_params()
        with self.assertRaises(ValueError):
            price_fx_option(
                params,
                strike=1.10,
                maturity_yrs=1.0,
                notional=1.0,
                option_type="straddle",
            )
        with self.assertRaises(TypeError):
            price_fx_option(
                params,
                strike=1.10,
                maturity_yrs=1.0,
                notional=1.0,
                option_type=123,  # type: ignore[arg-type]
            )

    def test_string_option_type_backwards_compatibility(self) -> None:
        """Passing strings 'call'/'put' should behave exactly as enum equivalents."""
        params = _flat_fx_params(fx_vol=0.10, sigma=0.0001)
        res_enum = price_fx_option(
            params,
            strike=1.10,
            maturity_yrs=1.0,
            notional=1.0,
            option_type=OptionType.CALL,
            n_paths=1000,
            n_steps=10,
            seed=42,
        )
        res_str = price_fx_option(
            params,
            strike=1.10,
            maturity_yrs=1.0,
            notional=1.0,
            option_type=" CALL ",
            n_paths=1000,
            n_steps=10,
            seed=42,
        )
        self.assertEqual(res_enum["price"], res_str["price"])

    def test_result_keys(self) -> None:
        """Result dict should contain expected keys."""
        params = _flat_fx_params(sigma=0.0001)
        result = price_fx_option(
            params,
            strike=1.10,
            maturity_yrs=1.0,
            notional=1.0,
            option_type=OptionType.CALL,
            n_paths=1_000,
            n_steps=10,
            seed=7,
        )
        self.assertIn("price", result)
        self.assertIn("std_error", result)
        self.assertIn("fx_terminal", result)


# -----------------------------------------------------------------------
# Calibration tests
# -----------------------------------------------------------------------


class TestCalibrateLgmToSwaptions(unittest.TestCase):
    def test_round_trip_calibration(self) -> None:
        """Calibrate to synthetic swaptions and verify σ is recovered."""
        # Use a known constant σ to generate synthetic swaption prices,
        # then calibrate and check we recover a similar σ.
        kappa = 0.03
        true_sigma = 0.0080  # 80 bp
        curve_yrs = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
        curve_dfs = np.exp(-0.03 * curve_yrs)

        expiries = np.array([1.0, 2.0, 5.0])
        swap_tenors = np.array([5.0, 5.0, 5.0])
        fixed_rates = np.full(3, 0.03)

        # Generate synthetic normal vols from the known model
        from xvasim.pricing_engine import _lgm_swaption_price_normal

        market_vols = np.zeros(3)
        for i in range(3):
            _, _mkt_p = _lgm_swaption_price_normal(
                expiry_yrs=expiries[i],
                swap_tenor_yrs=swap_tenors[i],
                market_normal_vol_ann=0.0,  # placeholder
                kappa=kappa,
                sigma_grid_yrs=np.array([30.0]),
                sigma_values_ann=np.array([true_sigma]),
                curve_yrs=curve_yrs,
                curve_dfs=curve_dfs,
                fixed_rate_ann=fixed_rates[i],
            )
            # Back out normal vol from model price
            mdl_p, _ = _lgm_swaption_price_normal(
                expiry_yrs=expiries[i],
                swap_tenor_yrs=swap_tenors[i],
                market_normal_vol_ann=0.0,
                kappa=kappa,
                sigma_grid_yrs=np.array([30.0]),
                sigma_values_ann=np.array([true_sigma]),
                curve_yrs=curve_yrs,
                curve_dfs=curve_dfs,
                fixed_rate_ann=fixed_rates[i],
            )
            # The model price IS the "market" price for this test
            # Back-solve for normal vol:  price = A × σ_n × √T × φ(0)
            # We just use the model_normal_vol directly
            # Actually, let's just extract it from the model function
            # by setting σ(t) = true_sigma and reading model price
            pass

        # Simpler approach: directly use model vols as market vols
        # (the calibration should recover the same σ)
        synthetic_vols = []
        for i in range(3):
            mdl_p, _ = _lgm_swaption_price_normal(
                expiry_yrs=expiries[i],
                swap_tenor_yrs=swap_tenors[i],
                market_normal_vol_ann=0.005,  # dummy
                kappa=kappa,
                sigma_grid_yrs=np.array([30.0]),
                sigma_values_ann=np.array([true_sigma]),
                curve_yrs=curve_yrs,
                curve_dfs=curve_dfs,
                fixed_rate_ann=fixed_rates[i],
            )
            # Back out normal vol from price:
            #   price = annuity × σ_n × √T × φ(0)
            # We need annuity — recompute
            n_per = max(1, round(swap_tenors[i] / 0.5))
            freq = swap_tenors[i] / n_per
            pay_times = np.array([expiries[i] + freq * (k + 1) for k in range(n_per)])
            pay_dfs = _interpolate_discount_factor(pay_times, curve_yrs, curve_dfs)
            annuity = float(np.sum(freq * pay_dfs))
            sqrt_t = np.sqrt(expiries[i])
            phi_0 = 1.0 / np.sqrt(2.0 * np.pi)
            implied_vol = mdl_p / (annuity * sqrt_t * phi_0)
            synthetic_vols.append(implied_vol)

        market_vols = np.array(synthetic_vols)

        # Now calibrate
        result = calibrate_lgm_to_swaptions(
            swaption_expiries_yrs=expiries,
            swap_tenors_yrs=swap_tenors,
            market_normal_vols_ann=market_vols,
            curve_yrs=curve_yrs,
            curve_dfs=curve_dfs,
            fixed_rates_ann=fixed_rates,
            kappa_ann=kappa,
        )

        # All calibrated σ values should be close to the true σ
        for sig in result.sigma_values_ann:
            self.assertAlmostEqual(sig, true_sigma, delta=0.002)

    def test_calibration_returns_lgm_params(self) -> None:
        """calibrate_lgm_to_swaptions should return an LGMParams instance."""
        curve_yrs = np.array([0.0, 1.0, 5.0, 10.0, 30.0])
        curve_dfs = np.exp(-0.03 * curve_yrs)
        expiries = np.array([1.0])
        swap_tenors = np.array([5.0])
        fixed_rates = np.array([0.03])
        market_vols = np.array([0.005])

        result = calibrate_lgm_to_swaptions(
            swaption_expiries_yrs=expiries,
            swap_tenors_yrs=swap_tenors,
            market_normal_vols_ann=market_vols,
            curve_yrs=curve_yrs,
            curve_dfs=curve_dfs,
            fixed_rates_ann=fixed_rates,
        )
        self.assertIsInstance(result, LGMParams)
        self.assertEqual(len(result.sigma_values_ann), 1)
        self.assertGreater(result.sigma_values_ann[0], 0.0)

    def test_calibration_failure_raises_runtime_error(self) -> None:
        """Calibration with unphysical market vol should raise RuntimeError."""
        curve_yrs = np.array([0.0, 1.0, 5.0, 10.0, 30.0])
        curve_dfs = np.exp(-0.03 * curve_yrs)
        expiries = np.array([1.0])
        swap_tenors = np.array([5.0])
        fixed_rates = np.array([0.03])
        # A extremely high market vol (e.g. 10.0) cannot be matched by LGM [1e-6, 2.0] vol bounds.
        market_vols = np.array([10.0])

        with self.assertRaises(RuntimeError):
            calibrate_lgm_to_swaptions(
                swaption_expiries_yrs=expiries,
                swap_tenors_yrs=swap_tenors,
                market_normal_vols_ann=market_vols,
                curve_yrs=curve_yrs,
                curve_dfs=curve_dfs,
                fixed_rates_ann=fixed_rates,
            )


if __name__ == "__main__":
    unittest.main()
