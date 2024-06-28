import unittest

import excitement
import utilities


class TestIncrementTime(unittest.TestCase):
    """Tests ensuring that the correct procedures
    happen when time is incremented
    """
    def setUp(self):
        """Create an excitation
        """
        self.weibull_weight=2
        self.weibull_alpha=2
        self.weibull_beta=2
        self.lin_val_weight=1
        self.lin_val_alpha=2
        self.lin_val_beta=2
        self.excitor_nodes=(0,2)
        self.excitee_nodes=(2,3)
        self.alive_threshold=0

        self.excitement = excitement.Excitement(
            weibull_weight=self.weibull_weight,
            weibull_alpha=self.weibull_alpha,
            weibull_beta=self.weibull_beta,
            lin_val_weight=self.lin_val_weight,
            lin_val_alpha=self.lin_val_alpha,
            lin_val_beta=self.lin_val_beta,
            excitor_nodes=self.excitor_nodes,
            excitee_nodes=self.excitee_nodes,
            alive_threshold=self.alive_threshold
        )

    def test_time_increasing(self):
        """Test that the time attribute is increased
        """
        for time in range(25):
            self.assertEqual(
                self.excitement.time,
                time
            )

            self.excitement.increment_time()

    def test_sum_probability_matches(self):
        """Test that the probability parts are summed
        """
        for time in range(25):
            self.assertEqual(
                sum(self.excitement.prob_parts),
                self.excitement.probability
            )

            self.excitement.increment_time()


class TestProbabilities(unittest.TestCase):
    """Tests ensuring that the probabilities are
    behaving appropriately
    """
    def setUp(self):
        """Create an excitation
        """
        self.weibull_weight=2
        self.weibull_alpha=2
        self.weibull_beta=2
        self.lin_val_weight=1
        self.lin_val_alpha=2
        self.lin_val_beta=2
        self.excitor_nodes=(0,2)
        self.excitee_nodes=(2,3)
        self.alive_threshold=0

        self.excitement = excitement.Excitement(
            weibull_weight=self.weibull_weight,
            weibull_alpha=self.weibull_alpha,
            weibull_beta=self.weibull_beta,
            lin_val_weight=self.lin_val_weight,
            lin_val_alpha=self.lin_val_alpha,
            lin_val_beta=self.lin_val_beta,
            excitor_nodes=self.excitor_nodes,
            excitee_nodes=self.excitee_nodes,
            alive_threshold=self.alive_threshold
        )

    def test_probability_accurate(self):
        """Compare the recursive probability to a direct
        calculation from utilities.py
        """
        for time in range(25):
            self.assertEqual(
                self.excitement.probability,
                self.weibull_weight*utilities.discrete_weibull_pmf(
                    time, self.weibull_alpha, self.weibull_beta
                )
            )

            self.excitement.increment_time()

    def test_probability_sums_to_weight(self):
        """Verify that the probability sums to one (pre-weighting)
        """
        running_total = 0

        for time in range(25):
            running_total += self.excitement.probability

            self.excitement.increment_time()

        self.assertAlmostEqual(
            self.weibull_weight,
            running_total
        )


class TestAliveThreshold(unittest.TestCase):
    """Tests to verify that the excitation only
    stays alive while the remaining probability is
    above the alive threshold
    """
    def setUp(self):
        """Create an excitation
        """
        self.weibull_weight=2
        self.weibull_alpha=2
        self.weibull_beta=2
        self.lin_val_weight=1
        self.lin_val_alpha=2
        self.lin_val_beta=2
        self.excitor_nodes=(0,2)
        self.excitee_nodes=(2,3)
        # Set the alive threshold to be high enough to trigger early
        self.alive_threshold=0.5

        self.excitement = excitement.Excitement(
            weibull_weight=self.weibull_weight,
            weibull_alpha=self.weibull_alpha,
            weibull_beta=self.weibull_beta,
            lin_val_weight=self.lin_val_weight,
            lin_val_alpha=self.lin_val_alpha,
            lin_val_beta=self.lin_val_beta,
            excitor_nodes=self.excitor_nodes,
            excitee_nodes=self.excitee_nodes,
            alive_threshold=self.alive_threshold
        )

    def test_excitement_dies(self):
        time = 0
        running_total = self.excitement.probability
        alive_threshold = self.excitement.alive_threshold

        # Keep iterating until the excitation is no longer alive
        while time < 100 and self.excitement.alive:
            # While its alive, the remaining weight should be above
            # the alive threshold
            self.assertGreaterEqual(
                self.weibull_weight - running_total,
                alive_threshold
            )

            self.excitement.increment_time()
            running_total += self.excitement.probability

        # When it dies, the remaining weight should be below the
        # alive threshold
        self.assertLess(
            self.weibull_weight - running_total,
            alive_threshold
        )
