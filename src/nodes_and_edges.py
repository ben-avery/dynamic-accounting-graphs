"""Module for handling lower-level classes
"""
import numpy as np

from utilities import (
    log_exp_function, log_exp_inverse, find_log_exp_shift,
    lin_exp_function, lin_exp_inverse, find_lin_exp_shift,
    adam_update
)


class Node():
    """A class for a node, containing embeddings and gradient-based
    learning functions
    """
    def __init__(self, name, opening_balance, dimension,
                 average_balance, average_weight,
                 epsilon=0.0001,
                 causal_learning_rate=0.001,
                 causal_learning_boost=1,
                 alpha_regularisation_rate=10**(-7),
                 beta_regularisation_rate=10**(-7),
                 weight_regularisation_rate=5*(10**(-4)),
                 spontaneous_learning_rate=0.00001,
                 spontaneous_regularisation_rate=10**(-7),
                 meta_data=None):
        """Initialise the class

        Args:
            name (str): The name of the node
            opening_balance (float): The monetary value accumulated in the
                associated account at the start of the period
            dimension (int): The dimension of the node embeddings
            average_balance (float): The daily average opening monetary value
                accumulated in the associated account for the whole period
            average_weight (float): The daily average excitation from the
                initialised Weibull distributions across the whole period
            epsilon (float, optional): The value below which intensity is trivial.
                Defaults to 0.0001.
            causal_learning_rate (float, optional): The learning rate for the
                optimisation of causal parameters. Defaults to 0.001.
            causal_learning_boost (float, optional): Multiple to boost the causal
                learning rate by during the training of only the causal part
                of the model (i.e. when the spontaneous part of the model is
                deactivated). Defaults to 1.
            alpha_regularisation_rate (float, optional): The weight towards the
                L2 regularisation penalty of Weibull alpha parameters. Defaults to 10**(-7).
            beta_regularisation_rate (float, optional): The weight towards the
                L2 regularisation penalty of Weibull beta parameters. Defaults to 10**(-7).
            weight_regularisation_rate  (float, optional): The weight towards the
                L2 regularisation penalty of Weibull weight parameters. Defaults to 5*(10**(-4)).
            spontaneous_learning_rate (float, optional): The learning rate for the
                optimisation of spontaneous parameters. Defaults to 0.00001.
            spontaneous_regularisation_rate (float, optional): The weight towards the
                L2 regularisation penalty of spontaneous parameters. Defaults to 10**(-7).
            meta_data (dict, optional): A dictionary of other information
                about a node which is irrelevant to the training. For example,
                it may be useful for downstream applications to know the
                mapping of the node to an associated accounting concept.
                Defaults to None.
        """

        # Store node parameters
        self.name = name
        self.opening_balance = opening_balance
        self.balance = opening_balance
        self.pending_balance_changes = 0
        self.dimension = dimension
        self.meta_data = meta_data

        self.causal_learning_rate = causal_learning_rate
        self.causal_learning_boost = causal_learning_boost

        self.alpha_regularisation_rate = alpha_regularisation_rate
        self.beta_regularisation_rate = beta_regularisation_rate
        self.weight_regularisation_rate = weight_regularisation_rate

        self.spontaneous_learning_rate = spontaneous_learning_rate
        self.spontaneous_regularisation_rate = spontaneous_regularisation_rate

        # Create the embeddings of the node
        f_shift = find_log_exp_shift(epsilon)
        if log_exp_inverse(0.001, f_shift) < 0:
            spontaneous_base_scaling_source = np.sqrt((1/(3*dimension))*(-log_exp_inverse(0.001, f_shift)))
            spontaneous_base_scaling_dest = -spontaneous_base_scaling_source
        else:
            spontaneous_base_scaling_source = np.sqrt((1/(3*dimension))*(log_exp_inverse(0.001, f_shift)))
            spontaneous_base_scaling_dest = spontaneous_base_scaling_source

        self.spontaneous_source_0 = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=spontaneous_base_scaling_source,
                learning_rate=self.spontaneous_learning_rate,
                regularisation_rate=self.spontaneous_regularisation_rate
            )
        self.spontaneous_source_1 = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=spontaneous_base_scaling_source/average_balance,
                learning_rate=self.spontaneous_learning_rate,
                regularisation_rate=self.spontaneous_regularisation_rate
            )
        self.spontaneous_source_2 = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=spontaneous_base_scaling_source,
                learning_rate=self.spontaneous_learning_rate,
                regularisation_rate=self.spontaneous_regularisation_rate
            )

        self.spontaneous_dest_0 = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=spontaneous_base_scaling_dest,
                learning_rate=self.spontaneous_learning_rate,
                regularisation_rate=self.spontaneous_regularisation_rate
            )
        self.spontaneous_dest_1 = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=spontaneous_base_scaling_dest,
                learning_rate=self.spontaneous_learning_rate,
                regularisation_rate=self.spontaneous_regularisation_rate
            )
        self.spontaneous_dest_2 = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=spontaneous_base_scaling_dest/average_balance,
                learning_rate=self.spontaneous_learning_rate,
                regularisation_rate=self.spontaneous_regularisation_rate
            )


        g_shift = find_lin_exp_shift(13/2)
        if lin_exp_inverse(15/2, g_shift) < 0:
            alpha_scaling_source = (-lin_exp_inverse(15/2, g_shift)/dimension)**(0.25)
            alpha_scaling_dest = -alpha_scaling_source
        else:
            alpha_scaling_source = (lin_exp_inverse(15/2, g_shift)/dimension)**(0.25)
            alpha_scaling_dest = alpha_scaling_source

        g_shift = find_lin_exp_shift(2)
        if lin_exp_inverse(3, g_shift) < 0:
            beta_scaling_source = (-lin_exp_inverse(3, g_shift)/dimension)**(0.25)
            beta_scaling_dest = -beta_scaling_source
        else:
            beta_scaling_source = (lin_exp_inverse(3, g_shift)/dimension)**(0.25)
            beta_scaling_dest = beta_scaling_source

        f_shift = find_log_exp_shift(epsilon)
        if lin_exp_inverse(1/average_weight, f_shift) < 0:
            weight_scaling_source = (-lin_exp_inverse(1/average_weight, f_shift)/dimension)**(0.25)
            weight_scaling_dest = -weight_scaling_source
        else:
            weight_scaling_source = (lin_exp_inverse(1/average_weight, f_shift)/dimension)**(0.25)
            weight_scaling_dest = weight_scaling_source


        self.causal_excitor_source_alpha = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=alpha_scaling_source,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.alpha_regularisation_rate
            )
        self.causal_excitor_source_beta = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=beta_scaling_source,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.beta_regularisation_rate
            )
        self.causal_excitor_source_weight = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=weight_scaling_source,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.weight_regularisation_rate
            )

        self.causal_excitor_dest_alpha = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=alpha_scaling_dest,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.alpha_regularisation_rate
            )
        self.causal_excitor_dest_beta = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=beta_scaling_dest,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.beta_regularisation_rate
            )
        self.causal_excitor_dest_weight = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=weight_scaling_dest,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.weight_regularisation_rate
            )

        self.causal_excitee_source_alpha = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=alpha_scaling_source,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.alpha_regularisation_rate
            )
        self.causal_excitee_source_beta = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=beta_scaling_source,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.beta_regularisation_rate
            )
        self.causal_excitee_source_weight = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=weight_scaling_source,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.weight_regularisation_rate
            )

        self.causal_excitee_dest_alpha = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=alpha_scaling_dest,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.alpha_regularisation_rate
            )
        self.causal_excitee_dest_beta = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=beta_scaling_dest,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.beta_regularisation_rate
            )
        self.causal_excitee_dest_weight = \
            NodeEmbedding(
                dimension=dimension,
                initialisation_scaling=weight_scaling_dest,
                learning_rate=self.causal_learning_rate,
                regularisation_rate=self.weight_regularisation_rate
            )

    def increment_time(self):
        """Advance to the next day
        """
        self.balance += self.pending_balance_changes
        self.pending_balance_changes = 0

    def update_balance(self, change):
        """Add a pending change to the balance, to be applied
        at the end of the day

        Args:
            change (float): The value to change by
        """
        self.pending_balance_changes += change

    def reset(self, discard_gradient_updates=False,
              spontaneous_on=True):
        """Reset balance and apply or discard gradient updates

        Args:
            discard_gradient_updates (bool, optional):
                Choose whether to discard or apply the pending gradient
                updates. Defaults to False.
            spontaneous_on (bool, optional): Whether the spontaneous part
                of the model is enabled. Recommended to have off for the
                early part of the model learning, to allow the causal part
                of the model to dominate.
                Defaults to True.
        """

        if discard_gradient_updates:
            self.clear_gradient_updates()
        else:
            self.apply_gradient_updates(
                spontaneous_on=spontaneous_on
            )

        self.reset_balance()

    def apply_gradient_updates(self, spontaneous_on=True):
        """At the end of an epoch, update the node embeddings
        based on the cached updates

        Args:
            spontaneous_on (bool, optional): Whether the spontaneous part
                of the model is enabled. Recommended to have off for the
                early part of the model learning, to allow the causal part
                of the model to dominate.
                Defaults to True.
        """

        # Update the embeddings
        if spontaneous_on:
            causal_boost = 1

            self.spontaneous_source_0.apply_gradient_updates()
            self.spontaneous_source_1.apply_gradient_updates()
            self.spontaneous_source_2.apply_gradient_updates()

            self.spontaneous_dest_0.apply_gradient_updates()
            self.spontaneous_dest_1.apply_gradient_updates()
            self.spontaneous_dest_2.apply_gradient_updates()
        else:
            causal_boost = self.causal_learning_boost

            self.spontaneous_source_0.clear_gradient_updates()
            self.spontaneous_source_1.clear_gradient_updates()
            self.spontaneous_source_2.clear_gradient_updates()

            self.spontaneous_dest_0.clear_gradient_updates()
            self.spontaneous_dest_1.clear_gradient_updates()
            self.spontaneous_dest_2.clear_gradient_updates()

        self.causal_excitor_source_alpha.apply_gradient_updates(
            learning_boost=causal_boost
        )
        self.causal_excitor_source_beta.apply_gradient_updates(
            learning_boost=causal_boost
        )
        self.causal_excitor_source_weight.apply_gradient_updates(
            learning_boost=causal_boost
        )

        self.causal_excitor_dest_alpha.apply_gradient_updates(
            learning_boost=causal_boost
        )
        self.causal_excitor_dest_beta.apply_gradient_updates(
            learning_boost=causal_boost
        )
        self.causal_excitor_dest_weight.apply_gradient_updates(
            learning_boost=causal_boost
        )

        self.causal_excitee_source_alpha.apply_gradient_updates(
            learning_boost=causal_boost
        )
        self.causal_excitee_source_beta.apply_gradient_updates(
            learning_boost=causal_boost
        )
        self.causal_excitee_source_weight.apply_gradient_updates(
            learning_boost=causal_boost
        )

        self.causal_excitee_dest_alpha.apply_gradient_updates(
            learning_boost=causal_boost
        )
        self.causal_excitee_dest_beta.apply_gradient_updates(
            learning_boost=causal_boost
        )
        self.causal_excitee_dest_weight.apply_gradient_updates(
            learning_boost=causal_boost
        )

    def clear_gradient_updates(self):
        """Remove any pending gradient updates
        """

        # Remove pending updates
        self.spontaneous_source_0.clear_gradient_updates()
        self.spontaneous_source_1.clear_gradient_updates()
        self.spontaneous_source_2.clear_gradient_updates()

        self.spontaneous_dest_0.clear_gradient_updates()
        self.spontaneous_dest_1.clear_gradient_updates()
        self.spontaneous_dest_2.clear_gradient_updates()

        self.causal_excitor_source_alpha.clear_gradient_updates()
        self.causal_excitor_source_beta.clear_gradient_updates()
        self.causal_excitor_source_weight.clear_gradient_updates()

        self.causal_excitor_dest_alpha.clear_gradient_updates()
        self.causal_excitor_dest_beta.clear_gradient_updates()
        self.causal_excitor_dest_weight.clear_gradient_updates()

        self.causal_excitee_source_alpha.clear_gradient_updates()
        self.causal_excitee_source_beta.clear_gradient_updates()
        self.causal_excitee_source_weight.clear_gradient_updates()

        self.causal_excitee_dest_alpha.clear_gradient_updates()
        self.causal_excitee_dest_beta.clear_gradient_updates()
        self.causal_excitee_dest_weight.clear_gradient_updates()

    def reset_balance(self):
        """Reset balance back to opening balance
        """
        self.balance = self.opening_balance

class NodeEmbedding():
    """A class to contain the source and destination node
    embedding for a particular node
    """
    def __init__(self, dimension, initialisation_scaling=1,
                 learning_rate=0.001, regularisation_rate=0.01):
        """Initialise the class

        Args:
            dimension (int): The dimension of the embeddings.
            initialisation_scaling (float, optional): Scale the random
                initialisation so that the expectation of each component
                is equal to this value.
                Defaults to 1.
            learning_rate (float, optional): The learning rate for the
                gradient ascent algorithm. Defaults to 0.001.
            regularisation_rate (float, optional): The weight towards the
                L2 regularisation penalty. Defaults to 0.01.
        """

        # Save the dimension and learning rates
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.regularisation_rate = regularisation_rate

        # Initialise the embeddings randomly
        self.value = np.random.uniform(0, 2*initialisation_scaling, self.dimension)

        # Create attributes to track gradient updates
        self.pending_updates = np.zeros(self.dimension)

        # Track the number of gradient steps
        self.learning_steps = 0

        # Initialise moment estimations
        self.prev_first_moment = 0
        self.prev_second_moment = 0

    def add_gradient_update(self, gradient_update):
        """Store a gradient update, to be applied to the node
        embedding at the end of the epoch.

        Args:
            gradient_update (np.array): The change in the source
                node embedding, as given by the gradient ascent
                algorithm for a single excitee
        """

        # Record the cumulative change to be applied later
        self.pending_updates += gradient_update

    def apply_gradient_updates(self, learning_boost=1):
        """At the end of an epoch, update the node embedding
        based on the cached updates

        Args:
            learning_boost (float, optional): Multiply the learning
                rate by this, to allow the rate of learning to be
                boosted or supressed.
                Defaults to 1.

        """

        # Increase the number of learning steps
        self.learning_steps += 1

        # Extract gradient
        gradient = -self.pending_updates

        # Adam update (with weight-decay regularisation)
        self.value, self.prev_first_moment, self.prev_second_moment = \
            adam_update(
                time=self.learning_steps,
                partial_deriv=gradient,
                prev_first_moment=self.prev_first_moment,
                prev_second_moment=self.prev_second_moment,
                prev_parameters=self.value,
                step_size=self.learning_rate*learning_boost,
                regularisation_rate=self.regularisation_rate
            )

        # Reset the cache
        self.pending_updates = np.zeros(self.dimension)

    def clear_gradient_updates(self):
        """Reset the pending updates without applying them
        """

        # Reset the cache
        self.pending_updates = np.zeros(self.dimension)

class EdgeEmbedder():
    """A class to combine two node embeddings into an edge embedding
    """
    def __init__(self, input_dimension):
        """Initialise the class

        Args:
            input_dimension (int): The dimension of the node embeddings
        """

        # Store the input dimension
        self.input_dimension = input_dimension

    def embed_edge(self, x_i, x_j):
        """Create an edge embedding by taking the hadamard
        product of the two node embeddings.

        Args:
            x_i (np.array): The node embedding for the source
            x_j (np.array): The node emebedding for the destination

        Returns:
            np.array: The edge embedding
        """
        return x_i * x_j


class EdgeComparer():
    """A class for generating positive, real numbers from two
    embeddings, with gradient-based learning functions
    """
    def __init__(self, dimension, f_shift=None, g_shift=None, positive_output=True, min_at=0):
        """Initialise the class

        Args:
            dimension (int): The dimension of the embeddings
            f_shift (float): Shift the function by to achieve a certain
                value at zero. (Only for positive_output=True).
                Defaults to None.
            g_shift (float): Shift the function by to achieve a certain
                value at zero. (Only for positive_output=True).
                Defaults to None.
            positive_output (bool, optional): Whether a function should
                be applied to the output to ensure it is positive.
                Defaults to True.
            min_at (float, optional): The minimum value for the function.
                Note that the regularisation will push the output to
                min_at + log_exp_scale if positive_output is True.
                Defaults to 0.0.
        """

        # Record the dimension
        self.dimension = dimension

        # Record if the positive function should be applied
        self.positive_output = positive_output
        self.min_at = min_at

        if self.positive_output:
            # To ensure that the output is positive, the result
            # of the linear function is passed through a smooth,
            # continuous function that is always positive. For
            # the gradient-ascent algorithm, the value of the
            # linear function is required.
            self.last_linear_value = None

            # Record the scaling to be applied to the smooth,
            # positive function
            self.f_shift = f_shift
            self.g_shift = g_shift
            if f_shift is None and g_shift is None:
                raise ValueError(
                    'One of f_shift and g_shift must be '
                    'provided if positive_output is True'
                )
            if f_shift is not None and g_shift is not None:
                raise ValueError(
                    'At most one of f_shift and g_shift must be '
                    'provided if positive_output is True'
                )

    #@profile
    def compare_embeddings(self, e_i, e_j):
        """Return the dot product of the two embeddings. If
        self.positive_output is set, this is then passed through
        the smooth, continuous, positive function.

        Args:
            e_i (np.array): First embedding
            e_j (np.array): Second embedding

        Returns:
            float: A positive real number
        """

        # Calculate the linear part of the function
        linear_value = e_i @ e_j

        if self.positive_output:
            # Cache this value for the gradient-ascent algorithm
            self.last_linear_value = linear_value

            # Apply the piecewise function to make the output
            # certainly positive
            if self.f_shift is None:
                return lin_exp_function(linear_value, self.g_shift) + self.min_at
            else:
                return log_exp_function(linear_value, self.f_shift) + self.min_at
        else:
            return linear_value
