import math
import numpy as np


def soft_max(numbers):
    scaled_numbers = np.power(10, 9 * numbers)

    return scaled_numbers/np.sum(scaled_numbers)


def discrete_weibull_pmf(x, alpha, beta):
    return \
        part_weibull(x, alpha, beta) \
        - part_weibull(x+1, alpha, beta)


def part_weibull(x, alpha, beta):
    return np.exp(-(x/alpha)**beta)


def calc_delP_delIntensity(count, sum_Intensity):
    return (1/math.factorial(count)) * (
        count*(sum_Intensity**(count-1)) - sum_Intensity**count
    )*np.exp(-sum_Intensity)


def calc_delIntensity_delAlpha(time, alpha, beta, weight):
    return weight * (
        beta *
        ((time**beta)/(alpha**(beta + 1))) *
        part_weibull(time, alpha, beta) - 
        beta *
        (((time + 1)**beta)/(alpha**(beta + 1))) *
        part_weibull(time + 1, alpha, beta)
    )


def calc_delIntensity_delBeta(time, alpha, beta, weight):
    if time == 0:
        return weight * (
            np.log((time+1)/alpha) *
            (((time+1)/alpha)**beta) *
            part_weibull(time+1, alpha, beta) 
        )
    else:
        return weight * (
            -np.log(time/alpha) *
            ((time/alpha)**beta) *
            part_weibull(time, alpha, beta) +
            np.log((time+1)/alpha) *
            (((time+1)/alpha)**beta) *
            part_weibull(time+1, alpha, beta) 
        )


def calc_delIntensity_delWeight(time, alpha, beta):
    return discrete_weibull_pmf(time, alpha, beta)


def calc_delComparer_delI(linear_value, matrix, e_kl, node_dimension):
    return \
        log_exp_multiplier(linear_value) * \
        (matrix[:node_dimension,:] @ e_kl)


def calc_delComparer_delJ(linear_value, matrix, e_kl, node_dimension):
    return \
        log_exp_multiplier(linear_value) * \
        (matrix[node_dimension:node_dimension*2,:] @ e_kl)


def calc_delComparer_delK(linear_value, matrix, e_ij, node_dimension):
    return \
        log_exp_multiplier(linear_value) * \
        (e_ij @ matrix[:,:node_dimension])


def calc_delComparer_delL(linear_value, matrix, e_ij, node_dimension):
    return \
        log_exp_multiplier(linear_value) * \
        (e_ij @ matrix[:,node_dimension:node_dimension*2])


def calc_delComparer_delMatrix(linear_value, e_ij, e_kl):
    return \
        log_exp_multiplier(linear_value) * \
        (e_ij * e_kl.reshape((e_kl.size, 1)))


def log_exp_multiplier(linear_value):
    if linear_value < 0:
        return np.exp(linear_value)
    else:
        return 1/(linear_value+1)
