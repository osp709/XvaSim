"""Tests for CVA aggregation and credit calculation engine."""

import typing
import unittest

import numpy as np

from xvasim.cva_engine import (
    CIRParams,
    _calibrate_cir,
    _calibrate_credit_model,
    _cir_survival_probability,
    _credit_model_survival_probability,
    compute_cva,
    compute_cva_chunked,
    compute_exposure_profile,
    compute_marginal_pd,
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
        """Verify Expected Exposure, EPE, Max PFE, and percentile profiles."""
        rng = np.random.default_rng(42)
        n_paths, n_steps = 10_000, 5
        # Generate some synthetic exposure matrix with negative and positive values
        exposure = rng.normal(loc=10.0, scale=20.0, size=(n_paths, n_steps))

        profile = compute_exposure_profile(exposure, percentiles=[95.0, 99.0])
        self.assertIn("expected_exposure", profile)
        self.assertIn("expected_positive_exposure", profile)
        self.assertIn("pfe_95.0", profile)
        self.assertIn("pfe_99.0", profile)
        self.assertIn("max_pfe", profile)

        ee = profile["expected_exposure"]
        epe = profile["expected_positive_exposure"]
        self.assertEqual(len(ee), n_steps)
        self.assertIsInstance(epe, float)
        self.assertAlmostEqual(epe, float(np.mean(ee)))

        # Max PFE must match maximum across time of 99th percentile (highest percentile)
        self.assertAlmostEqual(profile["max_pfe"], float(np.max(profile["pfe_99.0"])))

        # Test empty exposure matrix edge case
        empty_prof = compute_exposure_profile(np.zeros((0, 5)))
        self.assertEqual(len(empty_prof["expected_exposure"]), 0)
        self.assertEqual(empty_prof["max_pfe"], 0.0)


class TestCirSurvivalProbability(unittest.TestCase):
    """Tests for the CIR survival probability closed-form solution."""

    _DEFAULT_PARAMS = CIRParams(
        kappa_ann=0.5, theta_ann=0.03, sigma_ann=0.1, lambda_0_ann=0.02
    )

    def test_survival_probability_at_time_zero(self) -> None:
        """At t=0, survival probability should be 1.0."""
        tenors_yrs = np.array([0.0])
        surv = _cir_survival_probability(tenors_yrs, self._DEFAULT_PARAMS)
        np.testing.assert_allclose(surv, [1.0], atol=1e-10)

        # Test _credit_model_survival_probability with CIRParams and Model
        surv_gn = _credit_model_survival_probability(tenors_yrs, self._DEFAULT_PARAMS)
        np.testing.assert_allclose(surv_gn, [1.0], atol=1e-10)

        model = CIRHazardRateModel(self._DEFAULT_PARAMS)
        surv_mdl = _credit_model_survival_probability(tenors_yrs, model)
        np.testing.assert_allclose(surv_mdl, [1.0], atol=1e-10)

    def test_survival_probability_decreasing(self) -> None:
        """Survival probability should decrease with tenor."""
        tenors_yrs = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        surv = _cir_survival_probability(tenors_yrs, self._DEFAULT_PARAMS)
        for i in range(len(surv) - 1):
            self.assertGreater(surv[i], surv[i + 1])

    def test_survival_probability_bounded(self) -> None:
        """Survival probabilities should be in (0, 1]."""
        tenors_yrs = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        surv = _cir_survival_probability(tenors_yrs, self._DEFAULT_PARAMS)
        self.assertTrue(np.all(surv > 0))
        self.assertTrue(np.all(surv <= 1.0))

    def test_calibrate_credit_model(self) -> None:
        """Verify _calibrate_credit_model and _calibrate_cir."""
        tenors = np.array([1.0, 2.0, 3.0, 5.0])
        spreads = np.array([0.015, 0.018, 0.020, 0.025])
        cal_params = _calibrate_credit_model(spreads, tenors, model_type="cir")
        self.assertIsInstance(cal_params, CIRParams)

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


if __name__ == "__main__":
    unittest.main()
