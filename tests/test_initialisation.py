"""Unit testing for initialise_params.py
"""

import unittest
import numpy as np

import initialise_params
from graphs import DynamicAccountingGraph


class TestDailyAverageBalance(unittest.TestCase):
    """Test that the daily average balance calculations
    are accurate
    """

    def test_values(self):
        # Create edges dictionary
        edges_by_day = {
            0: [
                (0, 2, 100),
                (0, 2, 100),
                (1, 2, 100)
            ],
            1: [
                (2, 3, 300),
                (0, 2, 100)
            ],
            2: [
                (2, 3, 100),
                (0, 2, 100),
                (1, 2, 100)
            ],
        }
        last_day = 2

        # Create the true average balances
        balances = {
            0: (0 - 200 - 300)/3,
            1: (0 - 100 - 100)/3,
            2: (0 + 300 + 100)/3,
            3: (0 + 0 + 300)/3
        }

        # Use the function to calculate the balances
        calc_balances = initialise_params.find_average_balances(
            [0,0,0,0], edges_by_day, last_day
        )

        # Verify
        for i, balance in balances.items():
            self.assertEqual(
                calc_balances[i],
                balance
            )

    def test_extra_day(self):
        # Create edges dictionary
        edges_by_day = {
            0: [
                (0, 2, 100),
                (0, 2, 100),
                (1, 2, 100)
            ],
            1: [
                (2, 3, 300),
                (0, 2, 100)
            ],
            2: [
                (2, 3, 100),
                (0, 2, 100),
                (1, 2, 100)
            ],
        }
        last_day = 3

        # Create the true average balances
        balances = {
            0: (0 - 200 - 300 - 400)/4,
            1: (0 - 100 - 100 - 200)/4,
            2: (0 + 300 + 100 + 200)/4,
            3: (0 + 0 + 300 + 400)/4
        }

        # Use the function to calculate the balances
        calc_balances = initialise_params.find_average_balances(
            [0,0,0,0], edges_by_day, last_day
        )

        # Verify
        for i, balance in balances.items():
            self.assertEqual(
                calc_balances[i],
                balance
            )

