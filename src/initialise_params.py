"""Module for finding sensible initial values for the parameters
"""

import copy
import numpy as np

from utilities import discrete_weibull_pmf


def find_average_balances(opening_balances, edges_by_day, last_day):
    """Find the average daily opening balance for each account

    Args:
        opening_balances (list): The opening balances of each account
        edges_by_day (dict): Days (integers) as keys, and edges
            as values, where edges are tuples (i, j, weight).
        last_day (int): The last day in the accounting period
            (since there may be days at the end of the period
            on which no edges occured)
    """

    # Extract the opening balances
    summed_balances = {
        i: opening_balance
        for i, opening_balance in enumerate(opening_balances)}
    current_balances = copy.copy(summed_balances)

    for day in range(last_day):
        # Add any new edges
        if day in edges_by_day:
            for i, j, weight in edges_by_day[day]:
                current_balances[i] -= weight
                current_balances[j] += weight

        # Add the current balances to the running total
        for i, balance in current_balances.items():
            summed_balances[i] += balance

    # Divide by the number of days
    return {
        i: balance/(last_day+1)
        for i, balance in summed_balances.items()
    }


def find_average_initial_weight(edges_by_day, last_day,
                                initial_alpha, intial_beta):
    """Find the average weight per day from the initialised
    causal parameters

    Args:
        edges_by_day (dict): Days (integers) as keys, and edges
            as values, where edges are tuples (i, j, weight).
        last_day (int): The last day in the accounting period
            (since there may be days at the end of the period
            on which no edges occured)
        initial_alpha (float): The target alpha for parameter
            initialisation.
        initial_beta (float): The target beta for parameter
            initialisation.
    """

    # Get the excitation pattern for the initialised Weibull
    # distribution
    excitation_pattern = np.array([
        discrete_weibull_pmf(
            day, alpha=initial_alpha, beta=intial_beta
        )
        for day in range(last_day)
    ])

    # Calculate daily excitations
    daily_excitations = np.zeros(last_day)

    for day in range(last_day):
        if day in edges_by_day:
            if len(edges_by_day[day]) > 0:
                daily_excitations += \
                    np.hstack((
                        np.zeros(day+1),
                        len(edges_by_day[day])*excitation_pattern[:-day-1]
                    ))

    # Calculate the average
    return np.mean(daily_excitations)
