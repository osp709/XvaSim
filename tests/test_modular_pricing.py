"""Tests for modular multi-currency pricing and CVA integration."""

import unittest

import numpy as np

from xvasim import (
    CIRHazardRateModel,
    CIRParams,
    FXLGMParams,
    HullWhite1FModel,
    LGMModel,
    LGMParams,
    OptionType,
    TwoCurrencyFXModel,
    compute_cva,
    compute_marginal_pd,
    price_foreign_exchange_forward,
    price_foreign_exchange_option,
    price_fx_forward,
    price_fx_option,
)


def _make_curves(
    dom_rate: float = 0.03,
    for_rate: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tenors = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 30.0])
    dom_dfs = np.exp(-dom_rate * tenors)
    for_dfs = np.exp(-for_rate * tenors)
    return tenors, dom_dfs, for_dfs


class TestModularFXPricing(unittest.TestCase):
    def setUp(self) -> None:
        tenors, dom_dfs, for_dfs = _make_curves()
        self.tenors = tenors
        self.dom_dfs = dom_dfs
        self.for_dfs = for_dfs
        self.corr = np.array([
            [1.0, 0.2, -0.1],
            [0.2, 1.0, 0.15],
            [-0.1, 0.15, 1.0],
        ])

    def test_lgm_two_currency_matches_legacy(self) -> None:
        """TwoCurrencyFXModel with LGM models should match FXLGMParams pricing."""
        dom_lgm_params = LGMParams(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([30.0]),
            sigma_values_ann=np.array([0.008]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )
        for_lgm_params = LGMParams(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([30.0]),
            sigma_values_ann=np.array([0.006]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.for_dfs,
        )
        legacy_params = FXLGMParams(
            domestic=dom_lgm_params,
            foreign=for_lgm_params,
            spot_fx=1.10,
            fx_vol_ann=0.10,
            correlation_matrix=self.corr,
        )
        modular_model = TwoCurrencyFXModel.from_lgm_params(
            domestic=dom_lgm_params,
            foreign=for_lgm_params,
            spot_fx=1.10,
            fx_vol_ann=0.10,
            correlation_matrix=self.corr,
        )

        res_legacy = price_fx_option(
            legacy_params,
            strike=1.12,
            maturity_yrs=1.0,
            notional=100_000.0,
            option_type=OptionType.CALL,
            n_paths=50_000,
            n_steps=50,
            seed=42,
        )
        res_modular = price_fx_option(
            modular_model,
            strike=1.12,
            maturity_yrs=1.0,
            notional=100_000.0,
            option_type=OptionType.CALL,
            n_paths=50_000,
            n_steps=50,
            seed=42,
        )

        self.assertAlmostEqual(res_legacy["price"], res_modular["price"], places=6)
        self.assertAlmostEqual(
            res_legacy["std_error"], res_modular["std_error"], places=6
        )

    def test_hull_white_fx_pricing(self) -> None:
        """Price FX forward and option with Hull-White domestic and foreign models."""
        dom_hw = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.008,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )
        for_hw = HullWhite1FModel(
            a_ann=0.02,
            sigma_ann=0.006,
            discount_curve_yrs=self.tenors,
            discount_factors=self.for_dfs,
        )
        fx_model = TwoCurrencyFXModel(
            domestic_ir_model=dom_hw,
            foreign_ir_model=for_hw,
            spot_fx=1.10,
            fx_vol_ann=0.10,
            correlation_matrix=self.corr,
        )

        fwd_res = price_foreign_exchange_forward(
            fx_model,
            strike=1.10,
            maturity_yrs=1.0,
            notional=100_000.0,
            n_paths=20_000,
            n_steps=20,
            seed=7,
        )
        self.assertIn("price", fwd_res)

        opt_res = price_foreign_exchange_option(
            fx_model,
            strike=1.12,
            maturity_yrs=1.0,
            notional=100_000.0,
            option_type=OptionType.CALL,
            n_paths=20_000,
            n_steps=20,
            seed=7,
        )
        self.assertGreater(opt_res["price"], 0.0)

        # Verify alias matches
        fwd_alias = price_fx_forward(
            fx_model,
            strike=1.10,
            maturity_yrs=1.0,
            notional=100_000.0,
            n_paths=20_000,
            n_steps=20,
            seed=7,
        )
        self.assertEqual(fwd_res["price"], fwd_alias["price"])

    def test_mixed_models_fx_pricing(self) -> None:
        """Price with mixed models: Hull-White domestic + LGM foreign."""
        dom_hw = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.008,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )
        for_lgm = LGMModel(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([30.0]),
            sigma_values_ann=np.array([0.006]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.for_dfs,
        )
        fx_model = TwoCurrencyFXModel(
            domestic_ir_model=dom_hw,
            foreign_ir_model=for_lgm,
            spot_fx=1.10,
            fx_vol_ann=0.10,
            correlation_matrix=self.corr,
        )

        call_res = price_fx_option(
            fx_model,
            strike=1.10,
            maturity_yrs=1.0,
            notional=1.0,
            option_type=OptionType.CALL,
            n_paths=10_000,
            n_steps=20,
            seed=12,
        )
        put_res = price_fx_option(
            fx_model,
            strike=1.10,
            maturity_yrs=1.0,
            notional=1.0,
            option_type=OptionType.PUT,
            n_paths=10_000,
            n_steps=20,
            seed=12,
        )
        self.assertGreater(call_res["price"], 0.0)
        self.assertGreater(put_res["price"], 0.0)

    def test_invalid_params_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            price_fx_forward(
                "invalid_params",  # type: ignore[arg-type]
                strike=1.10,
                maturity_yrs=1.0,
                notional=1.0,
            )


class TestModularCreditAndCVA(unittest.TestCase):
    def test_compute_marginal_pd_with_custom_model(self) -> None:
        """compute_marginal_pd should accept a custom CreditModel."""
        cir_model = CIRHazardRateModel(
            params=CIRParams(
                kappa_ann=0.5,
                theta_ann=0.03,
                sigma_ann=0.10,
                lambda_0_ann=0.02,
            )
        )
        tenors = np.array([0.5, 1.0, 2.0, 5.0])
        spreads_dummy = np.zeros_like(tenors)

        marginal_pd = compute_marginal_pd(spreads_dummy, tenors, model=cir_model)
        self.assertEqual(len(marginal_pd), len(tenors))
        self.assertTrue(np.all(marginal_pd >= 0.0))
        self.assertLessEqual(np.sum(marginal_pd), 1.0)

    def test_cva_with_modular_credit_pd(self) -> None:
        """Compute CVA using marginal PDs generated from modular CreditModel."""
        credit_model = CIRHazardRateModel.calibrate_from_spreads(
            credit_spreads_ann=np.array([0.015, 0.02, 0.025, 0.03]),
            tenors_yrs=np.array([1.0, 2.0, 3.0, 5.0]),
        )
        tenors = np.array([1.0, 2.0, 3.0, 5.0])
        marginal_pd_1d = credit_model.marginal_pd(tenors)

        n_paths = 100
        n_dates = len(tenors)
        exposure = np.ones((n_paths, n_dates)) * 50.0
        dfs = np.tile(np.exp(-0.03 * tenors), (n_paths, 1))
        marginal_pd = np.tile(marginal_pd_1d, (n_paths, 1))

        cva = compute_cva(exposure, marginal_pd, dfs, loss_given_default=0.6)
        self.assertGreater(cva, 0.0)


if __name__ == "__main__":
    unittest.main()
