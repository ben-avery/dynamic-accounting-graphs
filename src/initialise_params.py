"""Module for finding sensible initial values for the parameters
"""

import numpy as np


def create_incidence_matrix(edges_by_day, last_day, count_nodes):
    """Create a matrix with edges as rows and days as columns,
    with the count of such instances in each component

    Args:
        edges_by_day (dict): Days (integers) as keys, and edges
            as values, where edges are tuples (i, j, weight).
        last_day (int): The last day in the accounting period
            (since there may be days at the end of the period
            on which no edges occured)
        count_nodes (int): The total number of nodes in the graph
    """

    # Initialise the matrix
    matrix = np.zeros((count_nodes*count_nodes, last_day+1))

    # Add details from the edges
    for day, edges in edges_by_day.items():
        for i, j, weight in edges:
            matrix[i*count_nodes+j][day] += 1

    return matrix


def matrix_to_freq_decomposition(matrix):
    """Use discrete Fourier Transforms to get the frequency
    decomposition of an incidence matrix

    Args:
        matrix (numpy array): Incidence matrix with edges as
        rows, days as columns and the count of such instances
        in each component
    """

    return np.array([
        np.fft.fft(row)
        for row in matrix
    ])
