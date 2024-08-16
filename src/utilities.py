"""A module with tools and helper functions
which are useful elsewhere in the code
"""
import math
import numpy as np
import functools


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


@functools.lru_cache(maxsize=128, typed=False)
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

    if x == 0:
        # Shortcut if x == 0
        return 1
    else:
        # Evaluate log of exponent to check for
        # underflows or overflows
        pre_exponent = beta * np.log(x/alpha)

        if pre_exponent > 20:
            # Handle underflows
            return 0
        elif pre_exponent < -20:
            # Handle overflows
            return 1
        else:
            # Calculate exponent
            exponent = -np.exp(pre_exponent)

    # Handle underflows
    if exponent < -20:
        return 0

    # Return the part-Weibull evaluation
    return np.exp(exponent)


def calc_inverse_probability_delP_delIntensity(count, sum_Intensity):
    """A partial derivative of the likelihood by
    the intensity, multiplied by the inverse of the likelihood
    (since the terms are always multiplied by each other)

    Args:
        count (int): The number of a certain edge that
            have occurred on a particular day
        sum_Intensity (float): The total intensity for
            that edge on that day

    Returns:
        float: The partial derivative
    """
    return count/sum_Intensity - 1


#@profile
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


def calc_delBaselineIntensity_delZero(linear_value, f_shift):
    """A partial derivative of the baseline intensity by
    the first linear parameter

    Args:
        linear_value (float): The linear part of the function
        f_shift (float): The f_shift for the log_exp function.
        
    Returns:
        float: The partial derivative
    """
    return log_exp_deriv_multiplier(linear_value, f_shift)


def calc_delBaselineIntensity_delOne(linear_value, source_balance, f_shift):
    """A partial derivative of the baseline intensity by
    the second linear parameter

    Args:
        linear_value (float): The linear part of the function
        source_balance (float): The balance on the source node
        f_shift (float): The f_shift for the log_exp function.

    Returns:
        float: The partial derivative
    """
    return log_exp_deriv_multiplier(linear_value, f_shift) * source_balance


def calc_delBaselineIntensity_delTwo(linear_value, dest_balance, f_shift):
    """A partial derivative of the baseline intensity by
    the third linear parameter

    Args:
        linear_value (float): The linear part of the function
        dest_balance (float): The balance on the source node
        f_shift (float): The f_shift for the log_exp function.

    Returns:
        float: The partial derivative
    """
    return log_exp_deriv_multiplier(linear_value, f_shift) * dest_balance


def calc_delBaselineDotproduct_delParam(parameter):
    """A partial derivative of the baseline linear coefficients
    by the relevant embeddings

    Args:
        parameter (np.array): The node embedding

    Returns:
        np.array: The partial derivative
    """

    return parameter


def calc_delCausalDotproduct_delParam(
        linear_value, node_embedding, edge_embedding, f_shift=None, g_shift=None):
    """A partial derivative of the causal parameters
    by the relevant embeddings

    Args:
        linear_value (float): The linear part of the function
        node_embedding (np.array): The node embedding
        edge_embedding (np.array): The edge embedding
        f_shift (float): The f_shift for the log_exp function.
        f_shift (float): The g_shift for the lin_exp function.

    Returns:
        np.array: The partial derivative
    """

    if f_shift is None:       
        return \
            lin_exp_deriv_multiplier(linear_value, g_shift) * (node_embedding*edge_embedding)
    else:
        return \
            log_exp_deriv_multiplier(linear_value, f_shift) * (node_embedding*edge_embedding)


@functools.lru_cache(maxsize=128, typed=False)
def log_exp_function(linear_value, f_shift):
    """A helper function which gives the smooth, continuous
    function that ensures parameters are positive.

    Args:
        linear_value (float): The value of the parameter before
            being passed through the smooth, continuous function
            to ensure it is positive
        f_shift (float): Shift the function by f_shift so
            that f(0) is a fixed value.

    Returns:
        float: The output of the smooth, continuous function
    """

    # Shift the linear value
    shifted_linear_value = linear_value + f_shift

    # The smooth function is defined piecewise
    if shifted_linear_value <= -30:
        # Lower limit for exponential function to prevent underflow
        return 0
    elif shifted_linear_value < 0:
        # Exponential portion
        return np.exp(shifted_linear_value)
    else:
        # Logarithmic portion
        return np.log(shifted_linear_value + 1) + 1


def log_exp_inverse(output_value, f_shift):
    """An inverse of log_exp_function

    Args:
        output_value (float): The output of the log_exp_function
        f_shift (float): Shift the original function by f_shift so
            that f(0) is a fixed value.

    Returns:
        float: The inverse of the smooth, continuous function
    """

    # The smooth function is defined piecewise
    if output_value == 0:
        # Lower limit for exponential function to prevent underflow
        return -30 - f_shift
    elif output_value < 1:
        # Exponential portion
        return np.log(output_value) - f_shift
    else:
        # Logarithmic portion
        return np.exp(output_value - 1) - 1 - f_shift


def find_log_exp_shift(value_at_zero):
    """Return the f_shift required to get
    log_exp_function(0, f_shift)=value_at_zero.

    Args:
        value_at_zero (float): The value that log_exp_function
            should take at zero, with f_shift given by the return
            of this function

    Returns:
        f_shift: The required shift
    """
    if value_at_zero > 1:
        return np.exp(value_at_zero - 1) - 1
    else:
        return np.log(value_at_zero)


@functools.lru_cache(maxsize=128, typed=False)
def log_exp_deriv_multiplier(linear_value, f_shift):
    """A helper function which gives the partial derivative
    from the smooth, continuous function that ensures the
    Weibull parameters and weight are positive.

    Args:
        linear_value (float): The value of the parameter before
            being passed through the smooth, continuous function
            to ensure it is positive
        f_shift (float): Shift the function by f_shift so
            that f(0) is a fixed value.

    Returns:
        float: The partial derivative of the smooth, continuous
            function with respect to the linear function
    """

    # Shift the linear value
    shifted_linear_value = linear_value + f_shift

    # The smooth function is defined piecewise, and therefore
    # its gradient has a different expression for positive and
    # negative values
    if shifted_linear_value < -30:
        # Lower limit for exponential function to prevent underflow
        # (but don't set to zero, otherwise gradient-based optimisers
        # will get stuck)
        return log_exp_deriv_multiplier(-30, 0)
    elif shifted_linear_value < 0:
        # Exponential portion
        return np.exp(shifted_linear_value)
    else:
        # Logarithmic portion
        return 1 / (shifted_linear_value + 1)


@functools.lru_cache(maxsize=128, typed=False)
def lin_exp_function(linear_value, g_shift):
    """A helper function which gives the smooth, continuous
    function that ensures parameters are positive.

    Args:
        linear_value (float): The value of the parameter before
            being passed through the smooth, continuous function
            to ensure it is positive
        g_shift (float): Shift the function by g_shift so
            that g(0) is a fixed value.

    Returns:
        float: The output of the smooth, continuous function
    """

    # Shift the linear value
    shifted_linear_value = linear_value + g_shift

    # The smooth function is defined piecewise
    if shifted_linear_value <= -30:
        # Lower limit for exponential function to prevent underflow
        return 0
    elif shifted_linear_value < 0:
        # Exponential portion
        return np.exp(shifted_linear_value)
    else:
        # Linear portion
        return shifted_linear_value + 1


def lin_exp_inverse(output_value, g_shift):
    """An inverse of log_exp_function

    Args:
        output_value (float): The output of the log_exp_function
        g_shift (float): Shift the original function by g_shift so
            that f(0) is a fixed value.

    Returns:
        float: The inverse of the smooth, continuous function
    """

    # The smooth function is defined piecewise
    if output_value == 0:
        # Lower limit for exponential function to prevent underflow
        return -30 - g_shift
    elif output_value < 1:
        # Exponential portion
        return np.log(output_value) - g_shift
    else:
        # Logarithmic portion
        return output_value - 1 - g_shift


def find_lin_exp_shift(value_at_zero):
    """Return the g_shift required to get
    lin_exp_function(0, g_shift)=value_at_zero.

    Args:
        value_at_zero (float): The value that lin_exp_function
            should take at zero, with g_shift given by the return
            of this function

    Returns:
        g_shift: The required shift
    """
    if value_at_zero > 1:
        return value_at_zero - 1
    else:
        return np.log(value_at_zero)


@functools.lru_cache(maxsize=128, typed=False)
def lin_exp_deriv_multiplier(linear_value, g_shift):
    """A helper function which gives the partial derivative
    from the smooth, continuous function that ensures the
    Weibull parameters and weight are positive.

    Args:
        linear_value (float): The value of the parameter before
            being passed through the smooth, continuous function
            to ensure it is positive
        g_shift (float): Shift the function by g_shift so
            that g(0) is a fixed value.

    Returns:
        float: The partial derivative of the smooth, continuous
            function with respect to the linear function
    """

    # Shift the linear value
    shifted_linear_value = linear_value + g_shift

    # The smooth function is defined piecewise, and therefore
    # its gradient has a different expression for positive and
    # negative values
    if shifted_linear_value < -30:
        # Lower limit for exponential function to prevent underflow
        # (but don't set to zero, otherwise gradient-based optimisers
        # will get stuck)
        return lin_exp_deriv_multiplier(-30, 0)
    elif shifted_linear_value < 0:
        # Exponential portion
        return np.exp(shifted_linear_value)
    else:
        # Logarithmic portion
        return 1


def adam_update(
        time, partial_deriv,
        prev_first_moment, prev_second_moment, prev_parameters,
        step_size=0.001, decay_one=0.9, decay_two=0.999,
        regularisation_rate=10**(-8), epsilon=10**(-8)):
    """Implementation of Adam optimisation algorithm
    Ref - Kingma, Ba, 'Adam: A Method for Stochastic Optimisation', 2014,
    https://doi.org/10.48550/arXiv.1412.6980

    Args:
        time (int): The number of learning steps taken
        partial_deriv (float/np.array): The partial derivative of
            the objective with respect to the parameters
        prev_first_moment (float/np.array): The estimate of the first
            moment from the previous calculation
        prev_second_moment (float/np.array): The estimate of the second
            moment from the previous calculation
        prev_parameters (float/np.array): The current value for the
            parameters before this update
        step_size (float, optional): Step size hyperparameter.
            Defaults to 0.001.
        decay_one (float, optional): Hyperparameter controlling the
            exponential decay of the first moment estimate.
            Defaults to 0.9.
        decay_two (float, optional): Hyperparameter controlling the
            exponential decay of the second moment estimate.
             Defaults to 0.999.
        epsilon (float, optional): Small value preventing
            division by zero.
            Defaults to 10**(-8).
    """

    # Update the moment estimates
    first_moment = decay_one*prev_first_moment + (1-decay_one)*(partial_deriv)
    second_moment = decay_two*prev_second_moment + (1-decay_two)*(partial_deriv*partial_deriv)

    # Get the adapted step size
    adapted_step_size = step_size*np.sqrt(1-decay_two**time)/(1-decay_one**time)

    # Update the parameters
    parameters = \
        prev_parameters*(1-2*regularisation_rate) \
        - adapted_step_size*first_moment/(np.sqrt(second_moment)+epsilon)

    return parameters, first_moment, second_moment
