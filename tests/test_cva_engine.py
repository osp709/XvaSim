import unittest

import numpy as np

from xvasim.cva_engine import (
    _cir_survival_probability,
    compute_cva,
    compute_marginal_pd,
)


class TestCvaEngine(unittest.TestCase):
    def test_compute_cva_basic(self) -> None:
        # Define test data for 2 paths and 3 dates
        exposure = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])

        marginal_pd = np.array([[0.01, 0.02, 0.03], [0.015, 0.025, 0.035]])

        discount_factor = np.array([[0.99, 0.98, 0.97], [0.985, 0.975, 0.965]])

        loss_given_default = 0.6

        # Execute CVA calculation
        cva = compute_cva(
            exposure=exposure,
            marginal_pd=marginal_pd,
            discount_factor=discount_factor,
            loss_given_default=loss_given_default,
        )

        # Expected calculation:
        # Path 0:
        #   t0: 10.0 * 0.01 * 0.99 * 0.6 = 0.0594
        #   t1: 20.0 * 0.02 * 0.98 * 0.6 = 0.2352
        #   t2: 30.0 * 0.03 * 0.97 * 0.6 = 0.5238
        #   Sum = 0.8184
        # Path 1:
        #   t0: 15.0 * 0.015 * 0.985 * 0.6 = 0.132975
        #   t1: 25.0 * 0.025 * 0.975 * 0.6 = 0.365625
        #   t2: 35.0 * 0.035 * 0.965 * 0.6 = 0.709275
        #   Sum = 1.207875
        # Average: (0.8184 + 1.207875) / 2 = 1.0131375
        expected_cva = 1.0131375

        self.assertAlmostEqual(cva, expected_cva, places=7)

    def test_compute_cva_single_value(self) -> None:
        # Test with 1 path and 1 date
        exposure = np.array([[100.0]])
        marginal_pd = np.array([[0.05]])
        discount_factor = np.array([[0.95]])
        loss_given_default = 0.4

        cva = compute_cva(exposure, marginal_pd, discount_factor, loss_given_default)
        expected_cva = 100.0 * 0.05 * 0.95 * 0.4
        self.assertAlmostEqual(cva, expected_cva, places=7)


class TestCirSurvivalProbability(unittest.TestCase):
    def test_survival_probability_at_time_zero(self) -> None:
        """At t=0, survival probability should be 1.0."""
        tenors_yrs = np.array([0.0])
        surv = _cir_survival_probability(
            tenors_yrs, kappa_ann=0.5, theta_ann=0.03, sigma_ann=0.1, lambda_0_ann=0.02
        )
        np.testing.assert_allclose(surv, [1.0], atol=1e-10)

    def test_survival_probability_decreasing(self) -> None:
        """Survival probability should decrease with tenor."""
        tenors_yrs = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        surv = _cir_survival_probability(
            tenors_yrs, kappa_ann=0.5, theta_ann=0.03, sigma_ann=0.1, lambda_0_ann=0.02
        )
        for i in range(len(surv) - 1):
            self.assertGreater(surv[i], surv[i + 1])

    def test_survival_probability_bounded(self) -> None:
        """Survival probabilities should be in (0, 1]."""
        tenors_yrs = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        surv = _cir_survival_probability(
            tenors_yrs, kappa_ann=0.5, theta_ann=0.03, sigma_ann=0.1, lambda_0_ann=0.02
        )
        self.assertTrue(np.all(surv > 0))
        self.assertTrue(np.all(surv <= 1.0))


class TestComputeMarginalPd(unittest.TestCase):
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
        self.assertTrue(
            np.all(marginal_pd >= -1e-10),
            f"Marginal PDs should be non-negative, got {marginal_pd}",
        )

    def test_cumulative_pd_at_most_one(self) -> None:
        """Sum of marginal PDs should not exceed 1."""
        tenors_yrs = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        credit_spreads_ann = np.array([0.01, 0.015, 0.02, 0.022, 0.025])
        marginal_pd = compute_marginal_pd(credit_spreads_ann, tenors_yrs)
        self.assertLessEqual(np.sum(marginal_pd), 1.0 + 1e-10)

    def test_flat_credit_spread_curve(self) -> None:
        """With a flat spread curve, marginal PDs should still be
        positive and increasing cumulative PD."""
        tenors_yrs = np.array([1.0, 2.0, 3.0, 5.0])
        credit_spreads_ann = np.array([0.02, 0.02, 0.02, 0.02])
        marginal_pd = compute_marginal_pd(credit_spreads_ann, tenors_yrs)
        cumulative_pd = np.cumsum(marginal_pd)

        # Cumulative PD should be monotonically increasing
        for i in range(len(cumulative_pd) - 1):
            self.assertLessEqual(cumulative_pd[i], cumulative_pd[i + 1])

    def test_upward_sloping_spread_curve(self) -> None:
        """With an upward-sloping spread curve, cumulative PD
        should increase more steeply at longer tenors."""
        tenors_yrs = np.array([1.0, 2.0, 3.0, 5.0])
        credit_spreads_ann = np.array([0.01, 0.02, 0.03, 0.05])
        marginal_pd = compute_marginal_pd(credit_spreads_ann, tenors_yrs)
        cumulative_pd = np.cumsum(marginal_pd)

        # All marginal PDs positive
        self.assertTrue(np.all(marginal_pd > -1e-10))
        # Cumulative PD increasing
        for i in range(len(cumulative_pd) - 1):
            self.assertLessEqual(cumulative_pd[i], cumulative_pd[i + 1])

    def test_single_tenor(self) -> None:
        """Should work with a single tenor point."""
        tenors_yrs = np.array([1.0])
        credit_spreads_ann = np.array([0.02])
        marginal_pd = compute_marginal_pd(credit_spreads_ann, tenors_yrs)
        self.assertEqual(marginal_pd.shape, (1,))
        self.assertGreater(marginal_pd[0], 0.0)


if __name__ == "__main__":
    unittest.main()
