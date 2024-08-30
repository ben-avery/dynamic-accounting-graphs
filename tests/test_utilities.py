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


class Test_inverse_prob_delP_delIntensity(unittest.TestCase):
    """Test the derivative of the Poisson pmf relative
    to its mean
    """
    def derivative_helper(self, x, mu, epsilon=10**(-9)):
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

        calc_deriv_times_inv_prob = utilities.calc_inverse_probability_delP_delIntensity(
            count=x, sum_Intensity=mu
        )

        self.assertAlmostEqual(
            calc_deriv_times_inv_prob,
            estimated_deriv/base_prob,
            places=5
        )

    def test_deriv(self):
        # Choose the derivative
        x = 2
        mu = 3

        # Test
        self.derivative_helper(
            x, mu
        )

    def test_deriv_at_zero(self):
        # Choose the derivative
        x = 0
        mu = 3

        # Test
        self.derivative_helper(
            x, mu
        )

    def test_deriv_non_integer(self):
        # Choose the derivative
        x = 6
        mu = 4.5

        # Test
        self.derivative_helper(
            x, mu
        )

    def test_deriv_small_mean(self):
        # Choose the derivative
        x = 1
        mu = 0.001

        # Test
        self.derivative_helper(
            x, mu
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
        # Choose the derivative
        time = 1
        alpha = 3
        beta = 3
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_at_zero_time(self):
        # Choose the derivative
        time = 0
        alpha = 3
        beta = 3
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_at_small_alpha(self):
        # Choose the derivative
        time = 1
        alpha = 0.001
        beta = 3
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_at_small_beta(self):
        # Choose the derivative
        time = 1
        alpha = 3
        beta = 0.001
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_at_large_alpha(self):
        # Choose the derivative
        time = 0
        alpha = 500
        beta = 3
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_at_large_beta(self):
        # Choose the derivative
        time = 0
        alpha = 3
        beta = 500
        weight = 1

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )

    def test_deriv_with_new_weight(self):
        # Choose the derivative
        time = 1
        alpha = 3
        beta = 3
        weight = 2

        # Test
        self.derivative_helper(
            time, alpha, beta, weight
        )


class Test_delBaselineIntensity_delCoeffs(unittest.TestCase):
    """Test the derivative of the baseline intensity
    relative to the three coefficients
    """
    def linear_part(self, coef_0, coef_1, coef_2,
                    balance_i, balance_j):
        full_linear_output = \
            coef_0 + \
            balance_i * coef_1 + \
            balance_j * coef_2

        return full_linear_output

    def baseline_intensity(self, coef_0, coef_1, coef_2,
                           balance_i, balance_j,
                           f_shift):
        full_linear_output = self.linear_part(
            coef_0, coef_1, coef_2,
            balance_i, balance_j
        )

        return utilities.log_exp_function(full_linear_output, f_shift)

    def derivative_helper(self, coef_0, coef_1, coef_2,
                          balance_i, balance_j, f_shift,
                          epsilon=10**(-8)):
        # Calculate the base value for the estimated derivatives
        linear_value = self.linear_part(
            coef_0, coef_1, coef_2,
            balance_i, balance_j)
        base_value = \
            self.baseline_intensity(
                coef_0, coef_1, coef_2,
                balance_i, balance_j,
                f_shift)

        # Calculate the estimated derivative for coef_0
        next_value = \
            self.baseline_intensity(
                coef_0+epsilon, coef_1, coef_2,
                balance_i, balance_j, f_shift)
        estimated_deriv = (next_value-base_value)/epsilon

        # Calculate the actual derivative for coef_0
        calc_deriv_0 = \
            utilities.calc_delBaselineIntensity_delZero(
                linear_value, f_shift)

        self.assertAlmostEqual(
            calc_deriv_0,
            estimated_deriv,
            msg='Coefficient 0',
            places=6)

        # Calculate the estimated derivative for coef_1
        next_value = \
            self.baseline_intensity(
                coef_0, coef_1+epsilon, coef_2,
                balance_i, balance_j, f_shift)
        estimated_deriv = (next_value-base_value)/epsilon

        # Calculate the actual derivative for coef_1
        calc_deriv_1 = \
            utilities.calc_delBaselineIntensity_delOne(
                linear_value, balance_i, f_shift)

        self.assertAlmostEqual(
            calc_deriv_1,
            estimated_deriv,
            msg='Coefficient 1',
            places=6)

        # Calculate the estimated derivative for coef_2
        next_value = \
            self.baseline_intensity(
                coef_0, coef_1, coef_2+epsilon,
                balance_i, balance_j, f_shift)
        estimated_deriv = (next_value-base_value)/epsilon

        # Calculate the actual derivative for coef_2
        calc_deriv_2 = \
            utilities.calc_delBaselineIntensity_delTwo(
                linear_value, balance_j, f_shift)

        self.assertAlmostEqual(
            calc_deriv_2,
            estimated_deriv,
            msg='Coefficient 2',
            places=6)

    def test_deriv(self):
        # Choose the derivative
        coef_0 = 1.5
        coef_1 = 1.2
        coef_2 = 0.8
        balance_i = 10
        balance_j = -10
        f_shift = 1.5

        self.derivative_helper(
            coef_0, coef_1, coef_2,
            balance_i, balance_j,
            f_shift
        )

    def test_negative_coefs(self):
        # Choose the derivative
        coef_0 = -3
        coef_1 = -3
        coef_2 = -3
        balance_i = 1.5
        balance_j = 1.5
        f_shift = 1.5

        self.derivative_helper(
            coef_0, coef_1, coef_2,
            balance_i, balance_j,
            f_shift
        )

    def test_positive_coefs(self):
        # Choose the derivative
        coef_0 = 3
        coef_1 = 3
        coef_2 = 3
        balance_i = 1.5
        balance_j = 1.5
        f_shift = 1.5

        self.derivative_helper(
            coef_0, coef_1, coef_2,
            balance_i, balance_j,
            f_shift
        )

    def test_zero_logscale(self):
        # Choose the derivative
        coef_0 = 1.5
        coef_1 = 1.2
        coef_2 = 0.8
        balance_i = 10
        balance_j = -10
        f_shift = 0.0

        self.derivative_helper(
            coef_0, coef_1, coef_2,
            balance_i, balance_j,
            f_shift
        )

    def test_negative_logscale(self):
        # Choose the derivative
        coef_0 = 1.5
        coef_1 = 1.2
        coef_2 = 0.8
        balance_i = 10
        balance_j = -10
        f_shift = -1.5

        self.derivative_helper(
            coef_0, coef_1, coef_2,
            balance_i, balance_j,
            f_shift
        )


class Test_delBaselineDotproduct_delParam(unittest.TestCase):
    """Test the derivative of the dot product which gives the
    coefficients of the linear part of the baseline function
    relative to its parameters
    """
    def derivative_helper(self, s_i, s_j, epsilon=10**(-8)):
        # Calculate node dimension
        node_dimension = len(s_i)

        # Calculate the function value
        base_value = s_i.T @ s_j

        # Calculate the actual derivative for I
        calc_deriv_I = \
            utilities.calc_delBaselineDotproduct_delParam(s_j)

        # Calculate the estimated derivative in each dimension
        # in turn for node I embedding
        for a in range(node_dimension):
            increment = np.eye(1, node_dimension, a)[0]*epsilon
            next_value = (s_i + increment).T @ s_j

            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Node I, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_I[a],
                    estimated_deriv,
                    places=6
                )

        # Calculate the actual derivative for I
        calc_deriv_J = \
            utilities.calc_delBaselineDotproduct_delParam(s_i)

        # Calculate the estimated derivative in each dimension
        # in turn for node J embedding
        for a in range(node_dimension):
            increment = np.eye(1, node_dimension, a)[0]*epsilon
            next_value = s_i.T @ (s_j + increment)

            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Node J, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_J[a],
                    estimated_deriv,
                    places=6
                )

    def test_deriv(self):
        # Choose the derivative
        s_i = np.array(
            [1.0, -0.4, 1.7, 0.2]
        )
        s_j = np.array(
            [-1.0, -1.0, 1.0, 1.0]
        )

        # Test
        self.derivative_helper(
            s_i, s_j
        )

    def test_zeros(self):
        # Choose the derivative
        s_i = np.array(
            [0.0, 0.0, 0.0, 0.0]
        )
        s_j = np.array(
            [-1.0, -1.0, 1.0, 1.0]
        )

        # Test
        self.derivative_helper(
            s_i, s_j
        )

    def test_positive(self):
        # Choose the derivative
        s_i = np.array(
            [1.0, 2.0, 3.0, 4.0]
        )
        s_j = np.array(
            [-1.0, -2.0, -3.0, -4.0]
        )

        # Test
        self.derivative_helper(
            s_i, s_j
        )

    def test_negative(self):
        # Choose the derivative
        s_i = np.array(
            [1.0, 2.0, 3.0, 4.0]
        )
        s_j = np.array(
            [4.0, 3.0, 2.0, 1.0]
        )

        # Test
        self.derivative_helper(
            s_i, s_j
        )


class Test_delCausalDotproduct_delParam(unittest.TestCase):
    """Test the derivative of the dot product which gives the
    positive causal parameters of the Weibull distribution
    relative to its parameters
    """
    def derivative_helper(self, r_i, r_j, e_k, e_l, f_shift,
                          epsilon=10**(-8)):
        # Calculate node dimension
        node_dimension = len(r_i)

        # Calculate the function value
        linear_value = (r_i * r_j).T @ (e_k * e_l)
        base_value = utilities.log_exp_function(linear_value, f_shift)

        # Calculate the actual derivative for I
        calc_deriv_I = \
            utilities.calc_delCausalDotproduct_delParam(
                linear_value,
                node_embedding=r_j,
                edge_embedding=e_k*e_l,
                f_shift=f_shift)

        # Calculate the estimated derivative in each dimension
        # in turn for node I embedding
        for a in range(node_dimension):
            increment = np.eye(1, node_dimension, a)[0]*epsilon
            next_value = utilities.log_exp_function(
                ((r_i+increment) * r_j).T @ (e_k * e_l),
                f_shift)

            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Node I, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_I[a],
                    estimated_deriv,
                    places=6
                )

        # Calculate the actual derivative for J
        calc_deriv_J = \
            utilities.calc_delCausalDotproduct_delParam(
                linear_value,
                node_embedding=r_i,
                edge_embedding=e_k*e_l,
                f_shift=f_shift)

        # Calculate the estimated derivative in each dimension
        # in turn for node J embedding
        for a in range(node_dimension):
            increment = np.eye(1, node_dimension, a)[0]*epsilon
            next_value = utilities.log_exp_function(
                (r_i * (r_j+increment)).T @ (e_k * e_l),
                f_shift)

            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Node J, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_J[a],
                    estimated_deriv,
                    places=6
                )

        # Calculate the actual derivative for K
        calc_deriv_K = \
            utilities.calc_delCausalDotproduct_delParam(
                linear_value,
                node_embedding=e_l,
                edge_embedding=r_i*r_j,
                f_shift=f_shift)

        # Calculate the estimated derivative in each dimension
        # in turn for node K embedding
        for a in range(node_dimension):
            increment = np.eye(1, node_dimension, a)[0]*epsilon
            next_value = utilities.log_exp_function(
                (r_i * r_j).T @ ((e_k+increment) * e_l),
                f_shift)

            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Node K, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_K[a],
                    estimated_deriv,
                    places=6
                )

        # Calculate the actual derivative for L
        calc_deriv_L = \
            utilities.calc_delCausalDotproduct_delParam(
                linear_value,
                node_embedding=e_k,
                edge_embedding=r_i*r_j,
                f_shift=f_shift)

        # Calculate the estimated derivative in each dimension
        # in turn for node L embedding
        for a in range(node_dimension):
            increment = np.eye(1, node_dimension, a)[0]*epsilon
            next_value = utilities.log_exp_function(
                (r_i * r_j).T @ (e_k * (e_l+increment)),
                f_shift)

            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Node L, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_L[a],
                    estimated_deriv,
                    places=6
                )

    def test_deriv(self):
        # Choose the derivative
        r_i = np.array(
            [1.0, -0.4, 1.7, 0.2]
        )
        r_j = np.array(
            [-1.0, -1.0, 1.0, 1.0]
        )
        e_k = np.array(
            [0.1, 0.2, 0.3, 0.4]
        )
        e_l = np.array(
            [1.5, -0.1, 0.1, 0.7]
        )
        f_shift = 1.5

        # Test
        self.derivative_helper(
            r_i, r_j, e_k, e_l, f_shift
        )

    def test_zeros(self):
        # Choose the derivative
        r_i = np.array(
            [0.0, 0.0, 0.0, 0.0]
        )
        r_j = np.array(
            [0.0, 0.0, 0.0, 0.0]
        )
        e_k = np.array(
            [0.0, 0.0, 0.0, 0.0]
        )
        e_l = np.array(
            [0.0, 0.0, 0.0, 0.0]
        )
        f_shift = 1.5

        # Test
        self.derivative_helper(
            r_i, r_j, e_k, e_l, f_shift
        )

    def test_positive(self):
        # Choose the derivative
        r_i = np.array(
            [1.0, 2.0, 3.0, 4.0]
        )
        r_j = np.array(
            [1.1, 1.2, 1.3, 1.4]
        )
        e_k = np.array(
            [0.5, 1.0, 0.5, 1.0]
        )
        e_l = np.array(
            [1.0, 1.5, 1.75, 1.825]
        )
        f_shift = 1.5

        # Test
        self.derivative_helper(
            r_i, r_j, e_k, e_l, f_shift
        )

    def test_negative(self):
        # Choose the derivative
        r_i = np.array(
            [-1.0, -2.0, -3.0, -4.0]
        )
        r_j = np.array(
            [1.1, 1.2, 1.3, 1.4]
        )
        e_k = np.array(
            [0.5, 1.0, 0.5, 1.0]
        )
        e_l = np.array(
            [1.0, 1.5, 1.75, 1.825]
        )
        f_shift = 1.5

        # Test
        self.derivative_helper(
            r_i, r_j, e_k, e_l, f_shift
        )

    def test_zero_logscale(self):
        # Choose the derivative
        r_i = np.array(
            [1.0, -0.4, 1.7, 0.2]
        )
        r_j = np.array(
            [-1.0, -1.0, 1.0, 1.0]
        )
        e_k = np.array(
            [0.1, 0.2, 0.3, 0.4]
        )
        e_l = np.array(
            [1.5, -0.1, 0.1, 0.7]
        )
        f_shift = 0.0

        # Test
        self.derivative_helper(
            r_i, r_j, e_k, e_l, f_shift
        )

    def test_negative_logscale(self):
        # Choose the derivative
        r_i = np.array(
            [1.0, -0.4, 1.7, 0.2]
        )
        r_j = np.array(
            [-1.0, -1.0, 1.0, 1.0]
        )
        e_k = np.array(
            [0.1, 0.2, 0.3, 0.4]
        )
        e_l = np.array(
            [1.5, -0.1, 0.1, 0.7]
        )
        f_shift = -2.5

        # Test
        self.derivative_helper(
            r_i, r_j, e_k, e_l, f_shift
        )


class Test_delF(unittest.TestCase):
    """Test the derivative of the smooth, continuous
    function designed to make values positive
    """
    def derivative_helper(self, input_value, f_shift,
                          epsilon=10**(-8)):
        # Calculate the function value
        base_value = \
            utilities.log_exp_function(input_value, f_shift)

        # Estimate the derivative
        next_value = \
            utilities.log_exp_function(input_value+epsilon, f_shift)

        estimated_deriv = (next_value-base_value)/epsilon

        # Calculate the derivative
        calc_deriv = \
            utilities.log_exp_deriv_multiplier(input_value, f_shift)

        self.assertAlmostEqual(
            calc_deriv,
            estimated_deriv
        )

    def test_deriv(self):
        # Choose the derivative
        self.derivative_helper(
            1, 1.5
        )

    def test_deriv_high(self):
        # Choose the derivative
        self.derivative_helper(
            200, 1.5
        )

    def test_deriv_low(self):
        # Choose the derivative
        self.derivative_helper(
            -25, 1.5
        )

    def test_deriv_zero(self):
        # Choose the derivative
        self.derivative_helper(
            0, 1.5
        )

    def test_deriv_underflow(self):
        # Choose the derivative
        self.derivative_helper(
            -40, 1.5
        )

    def test_zero_logscale(self):
        # Choose the derivative
        self.derivative_helper(
            1, 0.0
        )

    def test_negative_logscale(self):
        # Choose the derivative
        self.derivative_helper(
            1, -4.5
        )


class Test_F(unittest.TestCase):
    """Test the smooth, continuous function designed
    to make values positive
    """
    def test_zero(self):
        self.assertEqual(
            1,
            utilities.log_exp_function(0, 0)
        )

    def test_minus_one(self):
        self.assertEqual(
            1/np.exp(1),
            utilities.log_exp_function(-1, 0)
        )

    def test_one(self):
        self.assertEqual(
            np.log(2)+1,
            utilities.log_exp_function(1, 0)
        )

    def test_zero_with_logscale(self):
        self.assertEqual(
            4.7,
            utilities.log_exp_function(0, utilities.find_log_exp_shift(4.7))
        )



class Test_InvF(unittest.TestCase):
    """Test the inverse of the smooth, continuous function
    """
    def inverse_helper(self, value, f_shift=1):
        self.assertAlmostEqual(
            value,
            utilities.log_exp_function(
                utilities.log_exp_inverse(value, f_shift),
                f_shift)
        )

    def test_zero(self):
        self.inverse_helper(0)

    def test_half(self):
        self.inverse_helper(0.5)

    def test_one(self):
        self.inverse_helper(1)

    def test_zero_with_logscale(self):
        self.inverse_helper(0, 1.5)

    def test_large(self):
        self.inverse_helper(100)

    def test_small(self):
        self.inverse_helper(0.0001)

    def test_large_shift1(self):
        self.inverse_helper(1.5, 100)

    def test_small_shift1(self):
        self.inverse_helper(1.5, 0.0001)

    def test_large_shift2(self):
        self.inverse_helper(0.5, 100)

    def test_small_shift2(self):
        self.inverse_helper(0.5, 0.0001)


class Test_findLogExpShift(unittest.TestCase):
    """Test the helper function that provides the requisite
    shift in the smooth, positive function's input in order
    to get the desired value at 0.
    """
    def test_positive(self):
        self.assertEqual(
            4.7,
            utilities.log_exp_function(0, utilities.find_log_exp_shift(4.7))
        )

    def test_one(self):
        self.assertEqual(
            1.0,
            utilities.log_exp_function(0, utilities.find_log_exp_shift(1.0))
        )

    def test_less_than_one(self):
        self.assertEqual(
            0.47,
            utilities.log_exp_function(0, utilities.find_log_exp_shift(0.47))
        )


class Test_Adam(unittest.TestCase):
    """Test the Adam optimiser
    """
    def setUp(self):
        self.true_gradient = 5
        self.true_intercept = -1

    def function_output(self, input_value):
        return input_value*self.true_gradient + self.true_intercept

    def partial_deriv_gradient(self, input_value, estimated_gradient, estimated_intercept):
        return -2*(
            input_value*(self.true_gradient - estimated_gradient)
            + (self.true_intercept - estimated_intercept)
        ) * input_value

    def partial_deriv_intercept(self, input_value, estimated_gradient, estimated_intercept):
        return -2*(
            input_value*(self.true_gradient - estimated_gradient)
            + (self.true_intercept - estimated_intercept)
        )

    def test_linear(self):
        # Fix random seed
        np.random.seed(628496)

        # Initialise guess
        estimated_gradient = 1
        estimated_intercept = 1

        # Set up variables
        last_gradient = 0
        last_intercept = 0
        prev_grad_first_moment = 0
        prev_intercept_first_moment = 0
        prev_grad_second_moment = 0
        prev_intercept_second_moment = 0
        time = 0

        # Iterate until convergence
        while abs(estimated_gradient-last_gradient) > 0.000000001 or \
                abs(estimated_intercept-last_intercept) > 0.000000001:
            # Increase time
            time += 1

            # Remember the previous parameters (to check for convergence)
            last_gradient = estimated_gradient
            last_intercept = estimated_intercept

            # Pick a new random point
            x = np.random.randint(-15, 15)

            # Get the next parameters from the Adam algorithm
            estimated_gradient, prev_grad_first_moment, prev_grad_second_moment = \
                utilities.adam_update(
                    time=time,
                    partial_deriv=self.partial_deriv_gradient(x, estimated_gradient, estimated_intercept),
                    prev_first_moment=prev_grad_first_moment,
                    prev_second_moment=prev_grad_second_moment,
                    prev_parameters=estimated_gradient,
                    regularisation_rate=0,
                    step_size=0.0001
                )
            estimated_intercept, prev_intercept_first_moment, prev_intercept_second_moment = \
                utilities.adam_update(
                    time=time,
                    partial_deriv=self.partial_deriv_intercept(x, estimated_gradient, estimated_intercept),
                    prev_first_moment=prev_intercept_first_moment,
                    prev_second_moment=prev_intercept_second_moment,
                    prev_parameters=estimated_intercept,
                    regularisation_rate=0,
                    step_size=0.0001
                )

        # Check that the true parameters were recovered
        self.assertAlmostEqual(
            self.true_gradient, estimated_gradient,
            places=5
        )

        self.assertAlmostEqual(
            self.true_intercept, estimated_intercept,
            places=5
        )
