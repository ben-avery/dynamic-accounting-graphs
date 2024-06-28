import unittest
import numpy as np

import utilities


class TestWeibullPdf(unittest.TestCase):
    """Tests ensuring that the Weibull Pdf
    is accurate
    """

    def test_zeros_and_ones(self):
        """Test x=0, alpha=1, beta=1
        """
        self.assertAlmostEqual(
            utilities.discrete_weibull_pmf(
                x=0, alpha=1, beta=1
            ),
            1 - 1/np.exp(1)
        )

    def test_others(self):
        """Test x=3, alpha=4, beta=5
        """
        x = 3
        alpha = 4
        beta = 5

        self.assertAlmostEqual(
            utilities.discrete_weibull_pmf(
                x, alpha, beta
            ),
            np.exp(-((x/alpha)**beta)) - np.exp(-(((x+1)/alpha)**beta))
        )

    def test_large_x(self):
        """Test x=500, alpha=4, beta=5
        """
        x = 500
        alpha = 4
        beta = 5

        self.assertAlmostEqual(
            utilities.discrete_weibull_pmf(
                x, alpha, beta
            ),
            0
        )

    def test_large_alpha(self):
        """Test x=0, alpha=1000, beta=500
        """
        x = 0
        alpha = 1000
        beta = 500

        self.assertAlmostEqual(
            utilities.discrete_weibull_pmf(
                x, alpha, beta
            ),
            0
        )

    def test_large_beta(self):
        """Test x=10, alpha=10.5, beta=50000
        """
        x = 10
        alpha = 10.5
        beta = 50000

        self.assertAlmostEqual(
            utilities.discrete_weibull_pmf(
                x, alpha, beta
            ),
            1
        )

    def test_non_integers(self):
        """Test x=8, alpha=10.5, beta=50.5
        """
        x = 8
        alpha = 10.5
        beta = 2.5

        self.assertAlmostEqual(
            utilities.discrete_weibull_pmf(
                x, alpha, beta
            ),
            np.exp(-((x/alpha)**beta)) - np.exp(-(((x+1)/alpha)**beta))
        )

    def test_less_than_one(self):
        """Test x=4, alpha=0.25, beta=0.25
        """
        x = 4
        alpha = 0.25
        beta = 0.25

        self.assertAlmostEqual(
            utilities.discrete_weibull_pmf(
                x, alpha, beta
            ),
            np.exp(-((x/alpha)**beta)) - np.exp(-(((x+1)/alpha)**beta))
        )

    def test_sum_to_1(self):
        """Test that the probabilities sum to one
        alpha=10.5, beta=50.5
        """
        alpha = 10.5
        beta = 2.5

        running_total = 0
        for time in range(250):
            probability = utilities.discrete_weibull_pmf(
                time, alpha, beta
            )

            # Check that the probability is non-negative
            self.assertGreaterEqual(
                probability, 0
            )

            running_total += probability

        self.assertAlmostEqual(
            running_total,
            1
        )

    def test_sum_to_1_small_params(self):
        """Test that the probabilities sum to one
        alpha=0.25, beta=0.25
        """
        alpha = 0.25
        beta = 0.25

        running_total = 0
        for time in range(25000):
            probability = utilities.discrete_weibull_pmf(
                time, alpha, beta
            )

            # Check that the probability is non-negative
            self.assertGreaterEqual(
                probability, 0
            )

            running_total += probability

        self.assertAlmostEqual(
            running_total,
            1
        )

    def test_sum_to_1_large_params(self):
        """Test that the probabilities sum to one
        alpha=5.5, beta=5000
        """
        alpha = 5.5
        beta = 5000

        running_total = 0
        for time in range(250):
            probability = utilities.discrete_weibull_pmf(
                time, alpha, beta
            )

            # Check that the probability is non-negative
            self.assertGreaterEqual(
                probability, 0
            )

            running_total += probability

        self.assertAlmostEqual(
            running_total,
            1
        )
