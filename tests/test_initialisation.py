"""Unit testing for initialise_params.py
"""

import unittest
import numpy as np

import initialise_params


class TestIncidenceMatrix(unittest.TestCase):
    """Test that the incidence matrix is correctly
    created from a dictionary of edges
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

        # Create the matrix
        incidence_matrix = np.array([
            [0, 0, 0],  # Edge (0, 0)
            [0, 0, 0],  # Edge (0, 1)
            [2, 1, 1],  # Edge (0, 2)
            [0, 0, 0],  # Edge (0, 3)
            [0, 0, 0],  # Edge (1, 0)
            [0, 0, 0],  # Edge (1, 1)
            [1, 0, 1],  # Edge (1, 2)
            [0, 0, 0],  # Edge (1, 3)
            [0, 0, 0],  # Edge (2, 0)
            [0, 0, 0],  # Edge (2, 1)
            [0, 0, 0],  # Edge (2, 2)
            [0, 1, 1],  # Edge (2, 3)
            [0, 0, 0],  # Edge (3, 0)
            [0, 0, 0],  # Edge (3, 1)
            [0, 0, 0],  # Edge (3, 2)
            [0, 0, 0],  # Edge (3, 3)
        ])

        function_output = \
            initialise_params.create_incidence_matrix(
                edges_by_day=edges_by_day,
                last_day=2,
                count_nodes=4
            )

        # Verify that the function output matches
        # expectations
        for i, row in enumerate(incidence_matrix):
            self.assertListEqual(
                list(row),
                list(function_output[i])
            )

    def test_empty_days(self):
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

        # Create the matrix
        incidence_matrix = np.array([
            [0, 0, 0, 0, 0],  # Edge (0, 0)
            [0, 0, 0, 0, 0],  # Edge (0, 1)
            [2, 1, 1, 0, 0],  # Edge (0, 2)
            [0, 0, 0, 0, 0],  # Edge (0, 3)
            [0, 0, 0, 0, 0],  # Edge (1, 0)
            [0, 0, 0, 0, 0],  # Edge (1, 1)
            [1, 0, 1, 0, 0],  # Edge (1, 2)
            [0, 0, 0, 0, 0],  # Edge (1, 3)
            [0, 0, 0, 0, 0],  # Edge (2, 0)
            [0, 0, 0, 0, 0],  # Edge (2, 1)
            [0, 0, 0, 0, 0],  # Edge (2, 2)
            [0, 1, 1, 0, 0],  # Edge (2, 3)
            [0, 0, 0, 0, 0],  # Edge (3, 0)
            [0, 0, 0, 0, 0],  # Edge (3, 1)
            [0, 0, 0, 0, 0],  # Edge (3, 2)
            [0, 0, 0, 0, 0],  # Edge (3, 3)
        ])

        function_output = \
            initialise_params.create_incidence_matrix(
                edges_by_day=edges_by_day,
                last_day=4,
                count_nodes=4
            )

        # Verify that the function output matches
        # expectations
        for i, row in enumerate(incidence_matrix):
            self.assertListEqual(
                list(row),
                list(function_output[i])
            )

class TestFFT(unittest.TestCase):
    """Tests ensuring that functions turning
    an incidence matrix into a frequency decomposition
    are operating correctly
    """

    def test_combination(self):
        # Create three rows
        row_1 = np.array([0, 0, 1, 0, 0, 1])
        row_2 = np.array([1, 2, 1, 2, 1, 2])
        row_3 = np.array([0, 1, 5, 0, 0, 2])

        # Create the incidence matrix
        matrix = np.vstack((
            row_1, row_2, row_3
        ))

        # Calculate the FFT on each row manually
        manual_fft = np.vstack((
            np.fft.fft(row_1),
            np.fft.fft(row_2),
            np.fft.fft(row_3)
        ))

        # Use the function to convert the whole matrix
        function_fft = \
            initialise_params.matrix_to_freq_decomposition(
                matrix
            )

        # Verify that each row matches
        for i, row in enumerate(manual_fft):
            self.assertListEqual(
                list(row),
                list(function_fft[i])
            )
