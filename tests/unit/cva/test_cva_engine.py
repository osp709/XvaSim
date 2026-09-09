"""Tests for CVA aggregation and credit calculation engine."""

import typing
import unittest

import numpy as np

from xvasim.cva_engine import (
    CIRHazardRateParams,
    _calibrate_cir,
    _calibrate_credit_model,
    _credit_model_survival_probability,
    compute_cva,
    compute_cva_chunked,
    compute_dva,
    compute_exposure_profile,
    compute_fva,
    compute_kva,
    compute_marginal_pd,
    compute_mva,
    compute_total_xva,
)
from xvasim.models.credit.cir import CIRHazardRateModel


class TestCvaEngine(unittest.TestCase):
    """Unit tests for compute_cva and credit probability formulas."""

    def test_compute_cva_basic(self) -> None:
        """Verify compute_cva formula across multiple paths and dates."""
        exposure = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])
        marginal_pd = np.array([[0.01, 0.02, 0.03], [0.015, 0.025, 0.035]])
        discount_factor = np.array([[0.99, 0.98, 0.97], [0.985, 0.975, 0.965]])
        loss_given_default = 0.6

        cva = compute_cva(
            exposure=exposure,
            marginal_pd=marginal_pd,
            discount_factor=discount_factor,
            loss_given_default=loss_given_default,
        )
        expected_cva = 1.0131375
        self.assertAlmostEqual(cva, expected_cva, places=7)

    def test_compute_cva_single_value(self) -> None:
        """Verify compute_cva with single path and date."""
        exposure = np.array([[100.0]])
        marginal_pd = np.array([[0.05]])
        discount_factor = np.array([[0.95]])
        loss_given_default = 0.4

        cva = compute_cva(exposure, marginal_pd, discount_factor, loss_given_default)
        expected_cva = 100.0 * 0.05 * 0.95 * 0.4
        self.assertAlmostEqual(cva, expected_cva, places=7)

    def test_compute_cva_chunking_and_numexpr(self) -> None:
        """Verify chunked evaluation and Numexpr vs NumPy match exact values."""
        n_paths, n_dates = 1000, 10
        rng = np.random.default_rng(42)
        exposure = np.maximum(rng.standard_normal((n_paths, n_dates)) * 50.0, 0.0)
        marginal_pd = rng.uniform(0.001, 0.01, size=(n_paths, n_dates))
        discount_factor = np.exp(-0.03 * np.linspace(0.1, 5.0, n_dates))
        lgd = 0.60

        cva_numpy = compute_cva(
            exposure, marginal_pd, discount_factor, lgd, use_numexpr=False
        )
        cva_numexpr = compute_cva(
            exposure, marginal_pd, discount_factor, lgd, use_numexpr=True
        )
        cva_chunked = compute_cva(
            exposure, marginal_pd, discount_factor, lgd, chunk_size=100
        )

        self.assertAlmostEqual(cva_numpy, cva_numexpr, places=10)
        self.assertAlmostEqual(cva_numpy, cva_chunked, places=10)

    def test_compute_cva_chunked_generator(self) -> None:
        """Verify compute_cva_chunked on streaming generators."""
        n_paths, n_dates = 500, 5
        rng = np.random.default_rng(123)
        exposure = np.maximum(rng.standard_normal((n_paths, n_dates)) * 100.0, 0.0)
        marginal_pd = np.full(n_dates, 0.005)
        discount_factor = np.full(n_dates, 0.95)
        lgd = 0.40

        expected = compute_cva(exposure, marginal_pd, discount_factor, lgd)

        # Split into 5 chunks of 100 paths with one empty chunk in the middle
        def chunk_gen() -> typing.Iterator[np.ndarray]:
            for i in range(0, n_paths, 100):
                yield exposure[i : i + 100]
                if i == 200:
                    yield np.zeros((0, n_dates))

        stream_cva = compute_cva_chunked(
            chunk_gen(), marginal_pd, discount_factor, lgd, use_numexpr=True
        )
        self.assertAlmostEqual(expected, stream_cva, places=10)

        stream_cva_numpy = compute_cva_chunked(
            chunk_gen(), marginal_pd, discount_factor, lgd, use_numexpr=False
        )
        self.assertAlmostEqual(expected, stream_cva_numpy, places=10)

        cva_chunked_numpy = compute_cva(
            exposure, marginal_pd, discount_factor, lgd, chunk_size=100, use_numexpr=False
        )
        self.assertAlmostEqual(expected, cva_chunked_numpy, places=10)

    def test_compute_cva_empty_arrays(self) -> None:
        """Verify edge case handling for empty exposure inputs."""
        empty_exp = np.zeros((0, 5))
        cva = compute_cva(empty_exp, np.zeros(5), np.zeros(5), 0.6)
        self.assertEqual(cva, 0.0)

        cva_gen = compute_cva_chunked([], np.zeros(5), np.zeros(5), 0.6)
        self.assertEqual(cva_gen, 0.0)

    def test_compute_cva_chunked_2d_matrix(self) -> None:
        """Verify 2-D matrix streaming with chunk offset advancement."""
        n_paths, n_dates = 400, 4
        rng = np.random.default_rng(99)
        exposure = rng.uniform(10.0, 50.0, size=(n_paths, n_dates))
        marginal_pd_2d = rng.uniform(0.01, 0.03, size=(n_paths, n_dates))
        discount_factor_2d = rng.uniform(0.90, 0.99, size=(n_paths, n_dates))
        lgd = 0.40

        expected_cva = compute_cva(
            exposure=exposure,
            marginal_pd=marginal_pd_2d,
            discount_factor=discount_factor_2d,
            loss_given_default=lgd,
        )

        def chunk_gen() -> typing.Iterator[np.ndarray]:
            for i in range(0, n_paths, 100):
                yield exposure[i : i + 100]

        chunked_cva = compute_cva_chunked(
            exposure_chunks=chunk_gen(),
            marginal_pd=marginal_pd_2d,
            discount_factor=discount_factor_2d,
            loss_given_default=lgd,
        )
        self.assertAlmostEqual(expected_cva, chunked_cva, places=9)

    def test_compute_exposure_profile(self) -> None:
        """Verify Expected Exposure, EPE, NE, Max PFE, and percentile profiles."""
        rng = np.random.default_rng(42)
        n_paths, n_steps = 10_000, 5
        # Generate some synthetic exposure matrix with negative and positive values
        exposure = rng.normal(loc=10.0, scale=20.0, size=(n_paths, n_steps))

        profile = compute_exposure_profile(exposure, percentiles=[95.0, 99.0])
        self.assertIn("expected_exposure", profile)
        self.assertIn("expected_positive_exposure", profile)
        self.assertIn("expected_negative_exposure", profile)
        self.assertIn("ene", profile)
        self.assertIn("ene_scalar", profile)
        self.assertIn("funding_exposure", profile)
        self.assertIn("pfe_95.0", profile)
        self.assertIn("pfe_99.0", profile)
        self.assertIn("max_pfe", profile)

        ee = profile["expected_exposure"]
        epe = profile["expected_positive_exposure"]
        self.assertEqual(len(ee), n_steps)
        self.assertIsInstance(epe, float)
        self.assertAlmostEqual(epe, float(np.mean(ee)))

        # ENE is the average of negative exposure magnitudes
        ene = profile["expected_negative_exposure"]
        self.assertEqual(len(ene), n_steps)
        self.assertTrue(np.all(ene >= 0.0))
        self.assertAlmostEqual(
            profile["ene_scalar"], float(np.mean(ene)), places=10
        )

        # Funding exposure = EE - ENE
        np.testing.assert_allclose(
            profile["funding_exposure"], ee - ene, atol=1e-10
        )

        # Max PFE must match maximum across time of 99th percentile (highest percentile)
        self.assertAlmostEqual(profile["max_pfe"], float(np.max(profile["pfe_99.0"])))

        # Test empty exposure matrix edge case
        empty_prof = compute_exposure_profile(np.zeros((0, 5)))
        self.assertEqual(len(empty_prof["expected_exposure"]), 0)
        self.assertEqual(empty_prof["max_pfe"], 0.0)


class TestCirSurvivalProbability(unittest.TestCase):
    """Tests for the CIR survival probability closed-form solution."""

    _DEFAULT_PARAMS = CIRHazardRateParams(
        kappa_ann=0.5, theta_ann=0.03, sigma_ann=0.1, lambda_0_ann=0.02
    )

    def test_survival_probability_at_time_zero(self) -> None:
        """At t=0, survival probability should be 1.0."""
        tenors_yrs = np.array([0.0])
        surv = _credit_model_survival_probability(tenors_yrs, self._DEFAULT_PARAMS)
        np.testing.assert_allclose(surv, [1.0], atol=1e-10)

        # Test _credit_model_survival_probability with CIRHazardRateParams and Model
        surv_gn = _credit_model_survival_probability(tenors_yrs, self._DEFAULT_PARAMS)
        np.testing.assert_allclose(surv_gn, [1.0], atol=1e-10)

        model = CIRHazardRateModel(self._DEFAULT_PARAMS)
        surv_mdl = _credit_model_survival_probability(tenors_yrs, model)
        np.testing.assert_allclose(surv_mdl, [1.0], atol=1e-10)

    def test_survival_probability_decreasing(self) -> None:
        """Survival probability should decrease with tenor."""
        tenors_yrs = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        surv = _credit_model_survival_probability(tenors_yrs, self._DEFAULT_PARAMS)
        for i in range(len(surv) - 1):
            self.assertGreater(surv[i], surv[i + 1])

    def test_survival_probability_bounded(self) -> None:
        """Survival probabilities should be in (0, 1]."""
        tenors_yrs = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        surv = _credit_model_survival_probability(tenors_yrs, self._DEFAULT_PARAMS)
        self.assertTrue(np.all(surv > 0))
        self.assertTrue(np.all(surv <= 1.0))

    def test_calibrate_credit_model(self) -> None:
        """Verify _calibrate_credit_model and _calibrate_cir."""
        tenors = np.array([1.0, 2.0, 3.0, 5.0])
        spreads = np.array([0.015, 0.018, 0.020, 0.025])
        cal_params = _calibrate_credit_model(spreads, tenors, model_type="cir")
        self.assertIsInstance(cal_params, CIRHazardRateParams)

        cal_alias = _calibrate_cir(spreads, tenors)
        self.assertEqual(cal_params.kappa_ann, cal_alias.kappa_ann)

        with self.assertRaises(ValueError):
            _calibrate_credit_model(spreads, tenors, model_type="unknown")


class TestComputeMarginalPd(unittest.TestCase):
    """Tests for compute_marginal_pd calibration and monotonicity."""

    def test_output_shape(self) -> None:
        """Output should have the same length as input tenors."""
        tenors_yrs = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        credit_spreads_ann = np.array([0.01, 0.015, 0.02, 0.022, 0.025])
        marginal_pd = compute_marginal_pd(credit_spreads_ann, tenors_yrs)
        self.assertEqual(marginal_pd.shape, tenors_yrs.shape)

    def test_marginal_pd_non_negative(self) -> None:
        """All marginal PDs should be non-negative."""
        tenors_yrs = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        credit_spreads_ann = np.array([0.01, 0.015, 0.02, 0.022, 0.025])
        marginal_pd = compute_marginal_pd(credit_spreads_ann, tenors_yrs)
        self.assertTrue(np.all(marginal_pd >= -1e-10))

    def test_cumulative_pd_at_most_one(self) -> None:
        """Sum of marginal PDs should not exceed 1."""
        tenors_yrs = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        credit_spreads_ann = np.array([0.01, 0.015, 0.02, 0.022, 0.025])
        marginal_pd = compute_marginal_pd(credit_spreads_ann, tenors_yrs)
        self.assertLessEqual(np.sum(marginal_pd), 1.0 + 1e-10)

    def test_calibration_failure_raises_runtime_error(self) -> None:
        """CIR calibration should raise RuntimeError if optimization fails."""
        tenors_yrs = np.array([np.nan])
        credit_spreads_ann = np.array([0.02])
        with self.assertRaises(RuntimeError):
            compute_marginal_pd(credit_spreads_ann, tenors_yrs)

    def test_marginal_pd_with_credit_model(self) -> None:
        """Verify compute_marginal_pd dispatches to model.marginal_pd when model is provided."""
        from xvasim.models.credit.cir import CIRHazardRateModel

        credit_model = CIRHazardRateModel(
            kappa_ann=0.5,
            theta_ann=0.03,
            sigma_ann=0.1,
            lambda_0_ann=0.02,
        )
        tenors_yrs = np.array([0.5, 1.0, 2.0, 5.0])
        mpd = compute_marginal_pd(
            credit_spreads_ann=np.array([]), tenors_yrs=tenors_yrs, model=credit_model
        )
        self.assertEqual(len(mpd), 4)
        self.assertTrue(np.all(mpd >= 0.0))


class TestComputeDva(unittest.TestCase):
    """Unit tests for compute_dva."""

    def _dva_expected(self, ne, pd, df, lgd):
        """Manual DVA computation: LGD * mean(sum(ne * pd * df))."""
        return lgd * float(np.mean(np.sum(ne * pd * df, axis=1)))

    def test_compute_dva_basic(self) -> None:
        """Verify DVA formula against hand-computed expected values."""
        # All-negative exposure: NE = -E
        exposure = np.array([[-10.0, -20.0], [-15.0, -25.0]])
        marginal_pd = np.array([[0.01, 0.02], [0.015, 0.025]])
        discount_factor = np.array([[0.99, 0.98], [0.985, 0.975]])
        lgd = 0.6

        dva = compute_dva(exposure, marginal_pd, discount_factor, lgd)
        ne = np.maximum(-exposure, 0.0)
        expected = self._dva_expected(ne, marginal_pd, discount_factor, lgd)
        self.assertAlmostEqual(dva, expected, places=10)

    def test_dva_positive_exposure_is_zero(self) -> None:
        """DVA is 0.0 when all exposures are positive."""
        exposure = np.array([[10.0, 20.0], [15.0, 25.0]])
        marginal_pd = np.array([[0.01, 0.02], [0.015, 0.025]])
        discount_factor = np.array([[0.99, 0.98], [0.985, 0.975]])
        dva = compute_dva(exposure, marginal_pd, discount_factor, 0.6)
        self.assertEqual(dva, 0.0)

    def test_dva_with_1d_marginal_pd(self) -> None:
        """Verify 1-D marginal_pd broadcasting."""
        exposure = np.array([[-10.0, -20.0, -30.0], [-15.0, -25.0, -35.0]])
        marginal_pd = np.array([0.01, 0.02, 0.03])
        discount_factor = np.array([0.99, 0.98, 0.97])
        lgd = 0.5

        dva = compute_dva(exposure, marginal_pd, discount_factor, lgd)
        expected = lgd * float(
            np.mean(np.sum(np.maximum(-exposure, 0.0) * marginal_pd * discount_factor, axis=1))
        )
        self.assertAlmostEqual(dva, expected, places=10)

    def test_dva_chunking_matches(self) -> None:
        """Verify chunked vs full computation produces identical results."""
        rng = np.random.default_rng(7)
        exposure = -np.abs(rng.standard_normal((1000, 10))) * 50.0
        marginal_pd = rng.uniform(0.001, 0.01, size=(1000, 10))
        discount_factor = np.exp(-0.03 * np.arange(10))
        lgd = 0.6

        dva_full = compute_dva(exposure, marginal_pd, discount_factor, lgd, use_numexpr=False)
        dva_chunked = compute_dva(
            exposure, marginal_pd, discount_factor, lgd, chunk_size=100, use_numexpr=False
        )
        self.assertAlmostEqual(dva_full, dva_chunked, places=10)

    def test_dva_empty_arrays(self) -> None:
        """Empty inputs return 0.0."""
        dva = compute_dva(np.zeros((0, 5)), np.zeros(5), np.zeros(5), 0.6)
        self.assertEqual(dva, 0.0)

    def test_dva_uses_1d_discount(self) -> None:
        """Verify 1-D discount factor broadcasting works."""
        exposure = np.array([[-10.0], [-20.0], [-30.0]])
        marginal_pd = np.array([0.01])
        discount_factor = np.array([0.99])
        dva = compute_dva(exposure, marginal_pd, discount_factor, 0.6)
        expected = 0.6 * float(np.mean(np.array([10.0, 20.0, 30.0]) * 0.01 * 0.99))
        self.assertAlmostEqual(dva, expected, places=10)


class TestComputeFva(unittest.TestCase):
    """Unit tests for compute_fva."""

    def _fva_expected(self, exposure, dt, df, f_borrow, f_deposit):
        """Manual FVA leg computation alongside the net FVA."""
        ee = np.maximum(exposure, 0.0)
        ne = np.maximum(-exposure, 0.0)
        fca = float(np.mean(np.sum(ee * f_borrow * df * dt, axis=1)))
        fba = float(np.mean(np.sum(ne * f_deposit * df * dt, axis=1)))
        return fca, fba, fca + fba

    def test_fva_basic(self) -> None:
        """Verify FVA formula against hand-computed expected values."""
        exposure = np.array([[10.0, -20.0], [-15.0, 25.0]])
        dt = np.array([0.5, 0.5])
        df = np.array([0.99, 0.98])
        f_borrow = 0.005
        f_deposit = 0.002

        result = compute_fva(exposure, dt, df, f_borrow, f_deposit)
        fca_expected, fba_expected, fva_expected = self._fva_expected(
            exposure, dt, df, f_borrow, f_deposit
        )
        self.assertAlmostEqual(result["fca"], fca_expected, places=10)
        self.assertAlmostEqual(result["fba"], fba_expected, places=10)
        self.assertAlmostEqual(result["fva"], fva_expected, places=10)

    def test_fva_symmetric_legs(self) -> None:
        """FCA prices the positive-exposure leg, FBA the negative-exposure leg."""
        exposure = np.array([[10.0, -20.0], [-15.0, 25.0]])
        dt = np.array([0.5, 0.5])
        df = np.array([0.99, 0.98])
        f_borrow = 0.005
        f_deposit = 0.002

        result = compute_fva(exposure, dt, df, f_borrow, f_deposit)
        # FCA only involves positive exposure; FBA only negative exposure
        self.assertGreater(result["fca"], 0.0)
        self.assertGreater(result["fba"], 0.0)
        self.assertAlmostEqual(
            result["fva"], result["fca"] + result["fba"], places=10
        )

    def test_fva_default_deposit_spread(self) -> None:
        """Default deposit spread of 0.0 zeroes out the symmetric FBA leg."""
        exposure = np.array([[10.0], [-20.0]])
        dt = np.array([1.0])
        df = np.array([0.95])
        f_borrow = 0.01

        result = compute_fva(exposure, dt, df, f_borrow)
        # Negative-exposure leg vanishes: deposit spread is 0
        self.assertEqual(result["fba"], 0.0)
        self.assertAlmostEqual(
            result["fva"],
            0.01 * float(np.mean(np.maximum(exposure[:, 0], 0.0) * 0.95 * 1.0)),
            places=10,
        )

    def test_fva_zero_exposure(self) -> None:
        """FVA legs are 0.0 when exposure is zero."""
        exposure = np.zeros((5, 3))
        result = compute_fva(exposure, np.ones(3), np.ones(3), 0.005, 0.001)
        self.assertEqual(result["fca"], 0.0)
        self.assertEqual(result["fba"], 0.0)
        self.assertEqual(result["fva"], 0.0)

    def test_fva_chunking_matches(self) -> None:
        """Verify chunked vs full computation produces identical results."""
        rng = np.random.default_rng(11)
        exposure = rng.standard_normal((1000, 8)) * 100.0
        dt = np.full(8, 0.5)
        df = np.exp(-0.03 * np.arange(1, 9) * 0.5)
        f_borrow = 0.008
        f_deposit = 0.001

        fva_full = compute_fva(exposure, dt, df, f_borrow, f_deposit, use_numexpr=False)
        fva_chunked = compute_fva(
            exposure, dt, df, f_borrow, f_deposit, chunk_size=100, use_numexpr=False
        )
        self.assertAlmostEqual(fva_full["fca"], fva_chunked["fca"], places=10)
        self.assertAlmostEqual(fva_full["fba"], fva_chunked["fba"], places=10)
        self.assertAlmostEqual(fva_full["fva"], fva_chunked["fva"], places=10)

    def test_fva_empty_arrays(self) -> None:
        """Empty inputs return zero-valued legs."""
        result = compute_fva(np.zeros((0, 5)), np.ones(5), np.ones(5), 0.005)
        self.assertEqual(result, {"fca": 0.0, "fba": 0.0, "fva": 0.0})


class TestComputeKva(unittest.TestCase):
    """Unit tests for compute_kva."""

    def _kva_expected(self, exposure, dt, df, c_reg, lgd_reg):
        """Manual KVA computation: mean(sum(EPE * df * dt * c_reg * lgd_reg))."""
        epe = np.maximum(exposure, 0.0)
        return float(np.mean(np.sum(epe * df * dt * c_reg * lgd_reg, axis=1)))

    def test_kva_basic(self) -> None:
        """Verify KVA formula against hand-computed expected values."""
        exposure = np.array([[100.0, 200.0], [150.0, 250.0]])
        dt = np.array([0.5, 1.0])
        df = np.array([0.99, 0.98])
        c_reg = 0.08
        lgd_reg = 0.75

        kva = compute_kva(exposure, dt, df, c_reg, lgd_reg)
        expected = self._kva_expected(exposure, dt, df, c_reg, lgd_reg)
        self.assertAlmostEqual(kva, expected, places=10)

    def test_kva_default_regulatory_lgd(self) -> None:
        """Default regulatory LGD of 0.75 is applied."""
        exposure = np.array([[100.0]])
        kva = compute_kva(exposure, np.array([1.0]), np.array([0.95]), 0.08)
        expected = 100.0 * 0.95 * 1.0 * 0.08 * 0.75
        self.assertAlmostEqual(kva, expected, places=10)

    def test_kva_negative_exposure_ignored(self) -> None:
        """KVA is 0.0 when all exposures are negative."""
        exposure = np.array([[-100.0], [-200.0]])
        kva = compute_kva(exposure, np.array([1.0]), np.array([0.95]), 0.08)
        self.assertEqual(kva, 0.0)

    def test_kva_chunking_matches(self) -> None:
        """Verify chunked vs full computation produces identical results."""
        rng = np.random.default_rng(21)
        exposure = np.abs(rng.standard_normal((1000, 6))) * 50.0
        dt = np.full(6, 0.5)
        df = np.exp(-0.03 * np.arange(1, 7) * 0.5)

        kva_full = compute_kva(exposure, dt, df, 0.08, use_numexpr=False)
        kva_chunked = compute_kva(exposure, dt, df, 0.08, chunk_size=100, use_numexpr=False)
        self.assertAlmostEqual(kva_full, kva_chunked, places=10)

    def test_kva_empty_arrays(self) -> None:
        """Empty inputs return 0.0."""
        kva = compute_kva(np.zeros((0, 5)), np.ones(5), np.ones(5), 0.08)
        self.assertEqual(kva, 0.0)


class TestComputeMva(unittest.TestCase):
    """Unit tests for compute_mva."""

    def _im_profile(self, exposure, alpha, percentile):
        pfe = np.percentile(exposure, percentile, axis=0)
        return pfe * alpha

    def test_mva_basic(self) -> None:
        """Verify MVA uses PFE-based IM scaled by alpha."""
        rng = np.random.default_rng(31)
        exposure = np.abs(rng.standard_normal((1000, 5))) * 100.0
        dt = np.full(5, 0.5)
        df = np.exp(-0.03 * np.arange(1, 6) * 0.5)
        f_borrow = 0.01
        alpha = 0.5
        percentile = 95.0

        mva = compute_mva(
            exposure, dt, df, f_borrow, im_scaling=alpha, im_percentile=percentile
        )
        im = self._im_profile(exposure, alpha, percentile)
        im_matrix = np.tile(im, (1000, 1))
        expected = f_borrow * float(np.mean(np.sum(im_matrix * df * dt, axis=1)))
        self.assertAlmostEqual(mva, expected, places=10)

    def test_mva_alpha_one_default(self) -> None:
        """Default im_scaling=1.0."""
        exposure = np.array([[100.0, 200.0]])
        mva = compute_mva(exposure, np.array([0.5, 0.5]), np.array([0.99, 0.98]), 0.005)
        pfe = np.percentile(exposure, 99.0, axis=0)
        expected = 0.005 * float(np.sum(pfe * np.array([0.99, 0.98]) * np.array([0.5, 0.5])))
        self.assertAlmostEqual(mva, expected, places=10)

    def test_mva_scaling_effect(self) -> None:
        """Doubling im_scaling doubles MVA."""
        rng = np.random.default_rng(41)
        exposure = rng.standard_normal((500, 4)) * 50.0
        dt = np.full(4, 0.25)
        df = np.exp(-0.02 * np.arange(1, 5) * 0.25)

        mva_1x = compute_mva(exposure, dt, df, 0.005, im_scaling=1.0)
        mva_2x = compute_mva(exposure, dt, df, 0.005, im_scaling=2.0)
        self.assertAlmostEqual(mva_2x, 2.0 * mva_1x, places=10)

    def test_mva_zero_funding_spread(self) -> None:
        """MVA is 0.0 when funding spread is zero."""
        exposure = np.array([[100.0, 200.0]])
        mva = compute_mva(exposure, np.array([0.5, 0.5]), np.array([0.99, 0.98]), 0.0)
        self.assertEqual(mva, 0.0)

    def test_mva_empty_arrays(self) -> None:
        """Empty inputs return 0.0."""
        mva = compute_mva(np.zeros((0, 5)), np.ones(5), np.ones(5), 0.005)
        self.assertEqual(mva, 0.0)

    def test_mva_chunking_matches(self) -> None:
        """Verify chunked vs full computation produces identical results."""
        rng = np.random.default_rng(51)
        exposure = rng.standard_normal((1000, 6)) * 100.0
        dt = np.full(6, 0.5)
        df = np.exp(-0.03 * np.arange(1, 7) * 0.5)

        mva_full = compute_mva(exposure, dt, df, 0.008, use_numexpr=False)
        mva_chunked = compute_mva(exposure, dt, df, 0.008, chunk_size=100, use_numexpr=False)
        self.assertAlmostEqual(mva_full, mva_chunked, places=10)


class TestComputeTotalXva(unittest.TestCase):
    """Unit tests for compute_total_xva single-pass aggregator."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(61)
        self.n_paths, self.n_dates = 500, 6
        self.exposure = self.rng.standard_normal((self.n_paths, self.n_dates)) * 100.0
        self.dt = np.full(self.n_dates, 0.5)
        self.df = np.exp(-0.03 * np.arange(1, self.n_dates + 1) * 0.5)
        self.cp_pd = self.rng.uniform(0.001, 0.01, size=self.n_dates)
        self.own_pd = self.rng.uniform(0.001, 0.008, size=self.n_dates)

    def test_total_xva_matches_individual_components(self) -> None:
        """Single-pass total should match running each XVA independently."""
        total = compute_total_xva(
            self.exposure,
            self.dt,
            self.df,
            self.cp_pd,
            self.own_pd,
            counterparty_lgd=0.6,
            own_lgd=0.5,
            funding_spread_borrow_ann=0.006,
            funding_spread_deposit_ann=0.002,
            capital_charge_ann=0.08,
            regulatory_lgd=0.75,
            im_scaling=0.8,
            im_percentile=95.0,
            use_numexpr=False,
        )

        cva_expected = compute_cva(
            np.maximum(self.exposure, 0.0), self.cp_pd, self.df, 0.6, use_numexpr=False
        )
        dva_expected = compute_dva(
            self.exposure, self.own_pd, self.df, 0.5, use_numexpr=False
        )
        fva_expected = compute_fva(
            self.exposure, self.dt, self.df, 0.006, 0.002, use_numexpr=False
        )
        kva_expected = compute_kva(
            self.exposure, self.dt, self.df, 0.08, 0.75, use_numexpr=False
        )
        mva_expected = compute_mva(
            self.exposure, self.dt, self.df, 0.006,
            im_scaling=0.8, im_percentile=95.0, use_numexpr=False,
        )

        self.assertAlmostEqual(total["cva"], cva_expected, places=10)
        self.assertAlmostEqual(total["dva"], dva_expected, places=10)
        self.assertAlmostEqual(total["fca"], fva_expected["fca"], places=10)
        self.assertAlmostEqual(total["fba"], fva_expected["fba"], places=10)
        self.assertAlmostEqual(total["fva"], fva_expected["fva"], places=10)
        self.assertAlmostEqual(total["kva"], kva_expected, places=10)
        self.assertAlmostEqual(total["mva"], mva_expected, places=10)
        self.assertAlmostEqual(
            total["total_xva"],
            cva_expected - dva_expected + fva_expected["fva"] + kva_expected + mva_expected,
            places=10,
        )

    def test_total_xva_all_components_present(self) -> None:
        """Result contains all required keys."""
        total = compute_total_xva(
            self.exposure, self.dt, self.df, self.cp_pd, self.own_pd
        )
        for key in ("cva", "dva", "fca", "fba", "fva", "kva", "mva", "total_xva"):
            self.assertIn(key, total)
        self.assertIsInstance(total["total_xva"], float)
        # Symmetric FVA legs must sum to the net FVA
        self.assertAlmostEqual(total["fva"], total["fca"] + total["fba"], places=10)

    def test_total_xva_empty_arrays(self) -> None:
        """Empty inputs return all zeros."""
        total = compute_total_xva(
            np.zeros((0, 5)), np.ones(5), np.ones(5), np.zeros(5), np.zeros(5)
        )
        for key in ("cva", "dva", "fca", "fba", "fva", "kva", "mva", "total_xva"):
            self.assertEqual(total[key], 0.0)

    def test_total_xva_2d_pds(self) -> None:
        """2-D marginal PDs are supported."""
        cp_pd_2d = self.rng.uniform(0.001, 0.01, size=(self.n_paths, self.n_dates))
        own_pd_2d = self.rng.uniform(0.001, 0.008, size=(self.n_paths, self.n_dates))

        total_2d = compute_total_xva(
            self.exposure, self.dt, self.df, cp_pd_2d, own_pd_2d, use_numexpr=False
        )
        # Cross-check against individual computations with the same 2-D PDs
        cva_2d = compute_cva(
            np.maximum(self.exposure, 0.0), cp_pd_2d, self.df, 0.6, use_numexpr=False
        )
        dva_2d = compute_dva(
            self.exposure, own_pd_2d, self.df, 0.6, use_numexpr=False
        )
        self.assertAlmostEqual(total_2d["cva"], cva_2d, places=10)
        self.assertAlmostEqual(total_2d["dva"], dva_2d, places=10)

    def test_total_xva_negative_exposure_all_zero_credits(self) -> None:
        """All-negative exposure yields zero CVA/KVA but positive DVA contribution."""
        exposure = -np.abs(self.rng.standard_normal((200, 4))) * 100.0
        total = compute_total_xva(
            exposure,
            np.full(4, 0.5),
            np.exp(-0.03 * np.arange(1, 5) * 0.5),
            np.full(4, 0.01),
            np.full(4, 0.02),
            use_numexpr=False,
        )
        self.assertEqual(total["cva"], 0.0)
        self.assertEqual(total["kva"], 0.0)
        self.assertGreater(total["dva"], 0.0)


if __name__ == "__main__":
    unittest.main()
