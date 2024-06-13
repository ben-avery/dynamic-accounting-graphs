import numpy as np


def soft_max(numbers):
    scaled_numbers = np.power(10, 9 * numbers)

    return scaled_numbers/np.sum(scaled_numbers)


def discrete_weibull_pmf(x, alpha, beta):
    return np.exp(-(x/alpha)**beta) - np.exp(-((x+1)/alpha)**beta)


def part_weibull(x, alpha, beta):
    return np.exp(-(x/alpha)**beta)
