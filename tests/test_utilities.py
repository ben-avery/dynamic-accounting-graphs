import unittest
import numpy as np
from scipy.stats import poisson

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


class Test_delP_delIntensity(unittest.TestCase):
    """Test the derivative of the Poisson pmf relative
    to its mean
    """
    def derivative_helper(self, x, mu, epsilon=10**(-6)):
        """Calculate the derivative as a linear approximation
        over a small distance, and compare to the function

        Args:
            x (int): The count of edges
            mu (float): The mean of the Poisson (intensity)
            epsilon (float, optional): A very small interval
                over which to calculate the gradient.
                Defaults to 10**(-6).
        """
        base_prob = \
            poisson.pmf(
                k=x,
                mu=mu
            )

        next_prob = \
            poisson.pmf(
                k=x,
                mu=mu+epsilon
            )

        estimated_deriv = (next_prob-base_prob)/epsilon

        calc_deriv = utilities.calc_delP_delIntensity(
            count=x, sum_Intensity=mu
        )

        self.assertAlmostEqual(
            calc_deriv,
            estimated_deriv
        )

    def test_deriv(self):
        # Choose the derivate
        x = 2
        mu = 3

        # Test
        self.derivative_helper(
            x, mu
        )

    def test_deriv_at_zero(self):
        # Choose the derivate
        x = 0
        mu = 3

        # Test
        self.derivative_helper(
            x, mu
        )

    def test_deriv_non_integer(self):
        # Choose the derivate
        x = 6
        mu = 4.5

        # Test
        self.derivative_helper(
            x, mu
        )

    def test_deriv_small_mean(self):
        # Choose the derivate
        x = 1
        mu = 0.001

        # Test
        self.derivative_helper(
            x, mu, epsilon=10**(-8)
        )


class Test_delIntensity_delParams(unittest.TestCase):
    """Test the derivative of the weighted, discrete Weibull pmf
    relative to its parameters
    """
    def derivative_helper(self, time, alpha, beta, weight,
                          epsilon=10**(-8)):
        """Calculate the derivative as a linear approximation
        over a small distance, and compare to the function

        Args:
            time (int): The current time since the excitor which
                induced this Weibull distribution
            alpha (float): The alpha parameter of the Weibull
                distribution
            beta (float): The beta parameter of the Weibull
                distribution
            weight (float): The weight applied to the Weibull
                distribution
            epsilon (float, optional): A very small interval
                over which to calculate the gradient.
                Defaults to 10**(-6).
        """

        # Alpha parameter
        base_prob = \
            weight*utilities.discrete_weibull_pmf(
                x=time,
                alpha=alpha,
                beta=beta
            )

        next_prob = \
            weight*utilities.discrete_weibull_pmf(
                x=time,
                alpha=alpha+epsilon,
                beta=beta
            )

        estimated_deriv = (next_prob-base_prob)/epsilon

        calc_deriv = \
            utilities.calc_delIntensity_delAlpha(
                time, alpha, beta, weight)

        self.assertAlmostEqual(
            calc_deriv,
            estimated_deriv
        )

        # Beta parameter
        next_prob = \
            weight*utilities.discrete_weibull_pmf(
                x=time,
                alpha=alpha,
                beta=beta+epsilon
            )

        estimated_deriv = (next_prob-base_prob)/epsilon

        calc_deriv = \
            utilities.calc_delIntensity_delBeta(
                time, alpha, beta, weight)

        self.assertAlmostEqual(
            calc_deriv,
            estimated_deriv
        )

        # Weight parameter
        next_prob = \
            (weight+epsilon)*utilities.discrete_weibull_pmf(
                x=time,
                alpha=alpha,
                beta=beta
            )

        estimated_deriv = (next_prob-base_prob)/epsilon

        calc_deriv = \
            utilities.calc_delIntensity_delWeight(
                time, alpha, beta)

        self.assertAlmostEqual(
            calc_deriv,
            estimated_deriv
        )

    def test_deriv(self):
        # Choose the derivate
        time = 1
        alpha = 3
        beta = 3
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_at_zero_time(self):
        # Choose the derivate
        time = 0
        alpha = 3
        beta = 3
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_at_small_alpha(self):
        # Choose the derivate
        time = 1
        alpha = 0.001
        beta = 3
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_at_small_beta(self):
        # Choose the derivate
        time = 1
        alpha = 3
        beta = 0.001
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_at_large_alpha(self):
        # Choose the derivate
        time = 0
        alpha = 500
        beta = 3
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_at_large_beta(self):
        # Choose the derivate
        time = 0
        alpha = 3
        beta = 500
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_with_new_weight(self):
        # Choose the derivate
        time = 1
        alpha = 3
        beta = 3
        weight = 2

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )
