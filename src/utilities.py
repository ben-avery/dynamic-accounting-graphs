"""A module with tools and helper functions
which are useful elsewhere in the code
"""
import math
import numpy as np


def discrete_weibull_pmf(x, alpha, beta):
    """Calculate the probability of a discrete
    Weibull distribution taking a value of x
    given parameters alpha and beta

    Args:
        x (int): The value taken by the distribution
        alpha (float): A positive parameter, broadly
            characterising the central tendency of the
            distribution (for large values of beta, the
            distribution mode is alpha-1)
        beta (float): A positive parameter, broadly
            characterising the variance of the distribution

    Returns:
        float: The probability of this distribution
            taking the value x
    """

    # Use the helper function, given the form
    # of the PMF
    return \
        part_weibull(x, alpha, beta) \
        - part_weibull(x+1, alpha, beta)


def part_weibull(x, alpha, beta):
    """A helper function giving the expression
    that occurs multiple times in the discrete
    Weibull distribution PMF and its derivative

    Args:
        x (int): The value taken by the distribution
        alpha (float): A positive parameter, broadly
            characterising the central tendency of the
            distribution (for large values of beta, the
            distribution mode is alpha-1)
        beta (float): A positive parameter, broadly
            characterising the variance of the distribution

    Returns:
        float: The expression for use in the Weibull
            distribution (or its derivative, when used
            in gradient-based algorithms)
    """
    return np.exp(-(x/alpha)**beta)


def calc_delP_delIntensity(count, sum_Intensity):
    """A partial derivative of the likelihood by
    the intensity

    Args:
        count (int): The number of a certain edge that
            have occurred on a particular day
        sum_Intensity (float): The total intensity for
            that edge on that day

    Returns:
        float: The partial derivative
    """
    if count == 0:
        return -np.exp(-sum_Intensity)
    else:
        return (1/math.factorial(count)) * (
            count*(sum_Intensity**(count-1)) - sum_Intensity**count
        )*np.exp(-sum_Intensity)


def calc_delIntensity_delAlpha(time, alpha, beta, weight):
    """A partial derivative of the intensity by
    the alpha parameter of a Weibull distribution

    Args:
        time (int): The current time since the excitor which
            induced this Weibull distribution
        alpha (float): The alpha parameter of the Weibull
            distribution
        beta (float): The beta parameter of the Weibull
            distribution
        weight (float): The weight applied to the Weibull
            distribution

    Returns:
        float: The partial derivative
    """
    return weight * (
        beta *
        ((time**beta)/(alpha**(beta + 1))) *
        part_weibull(time, alpha, beta) - 
        beta *
        (((time + 1)**beta)/(alpha**(beta + 1))) *
        part_weibull(time + 1, alpha, beta)
    )


def calc_delIntensity_delBeta(time, alpha, beta, weight):
    """A partial derivative of the intensity by
    the beta parameter of a Weibull distribution

    Args:
        time (int): The current time since the excitor which
            induced this Weibull distribution
        alpha (float): The alpha parameter of the Weibull
            distribution
        beta (float): The beta parameter of the Weibull
            distribution
        weight (float): The weight applied to the Weibull
            distribution

    Returns:
        float: The partial derivative
    """
    if time == 0:
        # If time is zero, there is no change in
        # the first term of the Weibull PMF when beta
        # varies, and so that term can be dropped from
        # the partial derivative
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
    """A partial derivative of the intensity by
    the weight applied to the Weibull distribution

    Args:
        time (int): The current time since the excitor which
            induced this Weibull distribution
        alpha (float): The alpha parameter of the Weibull
            distribution
        beta (float): The beta parameter of the Weibull
            distribution

    Returns:
        float: The partial derivative
    """

    # The partial derivative is just the Weibull PMF
    return discrete_weibull_pmf(time, alpha, beta)


def calc_delComparer_delI(linear_value, matrix, e_kl, node_dimension):
    """A partial derivative of the Weibull parameters by
    the components of the source node of the excitor edge

    Args:
        linear_value (float): The value of the parameter before
            being passed through the smooth, continuous function
            to ensure it is positive
        matrix (np.array): The matrix from the linear function
        e_kl (np.array): The edge embedding of the excitee edge
        node_dimension (int): The dimension of the node embeddings
            (half of the edge embedding dimension)

    Returns:
        float: The partial derivative
    """

    return \
        log_exp_multiplier(linear_value) * \
        (matrix[:node_dimension,:] @ e_kl)


def calc_delComparer_delJ(linear_value, matrix, e_kl, node_dimension):
    """A partial derivative of the Weibull parameters by
    the components of the destination node of the excitor edge

    Args:
        linear_value (float): The value of the parameter before
            being passed through the smooth, continuous function
            to ensure it is positive
        matrix (np.array): The matrix from the linear function
        e_kl (np.array): The edge embedding of the excitee edge
        node_dimension (int): The dimension of the node embeddings
            (half of the edge embedding dimension)

    Returns:
        float: The partial derivative
    """

    return \
        log_exp_multiplier(linear_value) * \
        (matrix[node_dimension:node_dimension*2,:] @ e_kl)


def calc_delComparer_delK(linear_value, matrix, e_ij, node_dimension):
    """A partial derivative of the Weibull parameters by
    the components of the source node of the excitee edge

    Args:
        linear_value (float): The value of the parameter before
            being passed through the smooth, continuous function
            to ensure it is positive
        matrix (np.array): The matrix from the linear function
        e_ij (np.array): The edge embedding of the excitor edge
        node_dimension (int): The dimension of the node embeddings
            (half of the edge embedding dimension)

    Returns:
        float: The partial derivative
    """

    return \
        log_exp_multiplier(linear_value) * \
        (e_ij @ matrix[:,:node_dimension])


def calc_delComparer_delL(linear_value, matrix, e_ij, node_dimension):
    """A partial derivative of the Weibull parameters by
    the components of the destination node of the excitee edge

    Args:
        linear_value (float): The value of the parameter before
            being passed through the smooth, continuous function
            to ensure it is positive
        matrix (np.array): The matrix from the linear function
        e_ij (np.array): The edge embedding of the excitor edge
        node_dimension (int): The dimension of the node embeddings
            (half of the edge embedding dimension)

    Returns:
        float: The partial derivative
    """

    return \
        log_exp_multiplier(linear_value) * \
        (e_ij @ matrix[:,node_dimension:node_dimension*2])


def calc_delComparer_delMatrix(linear_value, e_ij, e_kl):
    """A partial derivative of the Weibull parameters by
    the components of the matrix from the linear function

    Args:
        linear_value (float): The value of the parameter before
            being passed through the smooth, continuous function
            to ensure it is positive
        e_ij (np.array): The edge embedding of the excitor edge
        e_kl (np.array): The edge embedding of the excitee edge

    Returns:
        float: The partial derivative
    """

    return \
        log_exp_multiplier(linear_value) * \
        (e_ij * e_kl.reshape((e_kl.size, 1)))


def calc_delBaselineIntensity_delZero(linear_value):
    """A partial derivative of the baseline intensity by
    the first linear parameter

    Args:
        linear_value (float): The linear part of the function

    Returns:
        float: The partial derivative
    """
    return log_exp_multiplier(linear_value)


def calc_delBaselineIntensity_delOne(linear_value, source_balance):
    """A partial derivative of the baseline intensity by
    the second linear parameter

    Args:
        linear_value (float): The linear part of the function
        source_balance (float): The balance on the source node

    Returns:
        float: The partial derivative
    """
    return log_exp_multiplier(linear_value) * source_balance


def calc_delBaselineIntensity_delTwo(linear_value, dest_balance):
    """A partial derivative of the baseline intensity by
    the third linear parameter

    Args:
        linear_value (float): The linear part of the function
        dest_balance (float): The balance on the source node

    Returns:
        float: The partial derivative
    """
    return log_exp_multiplier(linear_value) * dest_balance


def calc_delBaselineComparer_delK(matrix, y_l):
    """A partial derivative of the baseline linear coefficients
    by the source node embedding

    Args:
        matrix (np.array): The matrix from the linear function
        y_l (np.array): The node embedding of the destination node

    Returns:
        float: The partial derivative
    """

    return matrix @ y_l


def calc_delBaselineComparer_delL(matrix, y_k):
    """A partial derivative of the baseline linear coefficients
    by the destination node embedding

    Args:
        matrix (np.array): The matrix from the linear function
        y_k (np.array): The node embedding of the source node

    Returns:
        float: The partial derivative
    """

    return y_k @ matrix


def calc_delBaselineComparer_delMatrix(y_k, y_l):
    """A partial derivative of the baseline linear coefficients
    by the matrix components

    Args:
        y_k (np.array): The node embedding of the source node
        y_l (np.array): The node embedding of the destination node

    Returns:
        float: The partial derivative
    """

    return \
        y_k * y_l.reshape((y_l.size, 1))


def log_exp_multiplier(linear_value):
    """A helper function which gives the partial derivative
    from the smooth, continuous function that ensures the
    Weibull parameters and weight are positive.

    Args:
        linear_value (float): The value of the parameter before
            being passed through the smooth, continuous function
            to ensure it is positive

    Returns:
        float: The partial derivative of the smooth, continuous
            function with respect to the linear function
    """

    # The smooth function is defined piecewise, and therefore
    # its gradient has a different expression for positive and
    # negative values
    if linear_value < 0:
        # Exponential portion
        return np.exp(linear_value)
    else:
        # Logarithmic portion
        return 1/(linear_value+1)
