"""Module for handling lower-level classes
"""
import numpy as np

from utilities import log_exp_function


class Node():
    """A class for a node, containing embeddings and gradient-based
    learning functions
    """
    def __init__(self, name, opening_balance, dimension,
                 learning_rate=0.001, regularisation_rate=0.01,
                 meta_data=None):
        """Initialise the class

        Args:
            name (str): The name of the node
            opening_balance (float): The monetary value accumulated in the
                associated account at the start of the period
            dimension (int): The dimension of the node embeddings
            learning_rate (float, optional): The learning rate for the
                gradient ascent algorithm. Defaults to 0.001.
            regularisation_rate (float, optional): The weight towards the
                L2 regularisation penalty. Defaults to 0.01.
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
        self.learning_rate = learning_rate
        self.regularisation_rate = regularisation_rate

        # Create the embeddings of the node (as a source and
        # destination separately)
        self.spontaneous_embeddings = \
            NodeEmbedding(dimension=dimension,
                          learning_rate=self.learning_rate,
                          regularisation_rate=self.regularisation_rate)
        self.causal_embeddings = \
            NodeEmbedding(dimension=dimension,
                          learning_rate=self.learning_rate,
                          regularisation_rate=self.regularisation_rate)

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

    def add_spontaneous_gradient_update(
            self, gradient_update, node_type):
        """Store a gradient update, to be applied to the 
        spontaneous node embedding at the end of the epoch.

        Args:
            gradient_update (np.array): The change in the source
                node embedding, as given by the gradient ascent
                algorithm for a single excitee
            node_type (str): Either 'source' or 'dest'
        """

        # Add the update
        if node_type == 'source':
            self.spontaneous_embeddings.add_source_gradient_update(
                gradient_update
            )
        elif node_type == 'dest':
            self.spontaneous_embeddings.add_dest_gradient_update(
                gradient_update
            )
        else:
            raise ValueError('Node type must be "source" or "dest"')

    def add_causal_gradient_update(
            self, gradient_update, node_type):
        """Store a gradient update, to be applied to the 
        causal node embedding at the end of the epoch.

        Args:
            gradient_update (np.array): The change in the source
                node embedding, as given by the gradient ascent
                algorithm for a single excitee
            node_type (str): Either 'source' or 'dest'
        """

        # Add the update
        if node_type == 'source':
            self.causal_embeddings.add_source_gradient_update(
                gradient_update
            )
        elif node_type == 'dest':
            self.causal_embeddings.add_dest_gradient_update(
                gradient_update
            )
        else:
            raise ValueError('Node type must be "source" or "dest"')

    def reset(self, discard_gradient_updates=False):
        """Reset balance and apply or discard gradient updates

        Args:
            discard_gradient_updates (bool, optional):
            Choose whether to discard or apply the pending gradient
            updates. Defaults to False.
        """

        if discard_gradient_updates:
            self.clear_gradient_updates()
        else:
            self.apply_gradient_updates()

        self.reset_balance()

    def apply_gradient_updates(self):
        """At the end of an epoch, update the node embeddings
        based on the cached updates
        """

        # Update the embeddings
        self.spontaneous_embeddings.apply_gradient_updates()
        self.causal_embeddings.apply_gradient_updates()

    def clear_gradient_updates(self):
        """Remove any pending gradient updates
        """

        # Remove pending updates
        self.spontaneous_embeddings.clear_gradient_updates()
        self.causal_embeddings.clear_gradient_updates()

    def reset_balance(self):
        """Reset balance back to opening balance
        """
        self.balance = self.opening_balance

class NodeEmbedding():
    """A class to contain the source and destination node
    embedding for a particular node
    """
    def __init__(self, dimension,
                 learning_rate=0.001, regularisation_rate=0.01):
        """Initialise the class

        Args:
            dimension (int): The dimension of the embeddings.
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
        max_value = 2*np.sqrt(1/self.dimension)
        self.source_value = np.random.uniform(0, max_value, self.dimension)
        self.dest_value = np.random.uniform(0, max_value, self.dimension)

        # Create attributes to track gradient updates
        self.source_pending_updates = np.zeros(self.dimension)
        self.dest_pending_updates = np.zeros(self.dimension)

    def add_source_gradient_update(self, gradient_update):
        """Store a gradient update, to be applied to the node
        embedding (as a source) at the end of the epoch.

        Args:
            gradient_update (np.array): The change in the source
                node embedding, as given by the gradient ascent
                algorithm for a single excitee
        """

        # Multiply the given change by the learning rate, and
        # record the cumulative change to be applied later
        self.source_pending_updates += self.learning_rate*gradient_update

    def add_dest_gradient_update(self, gradient_update):
        """Store a gradient update, to be applied to the node
        embedding (as a destination) at the end of the epoch.

        Args:
            gradient_update (np.array): The change in the destination
                node embedding, as given by the gradient ascent
                algorithm for a single excitee
        """

        # Multiply the given change by the learning rate, and
        # record the cumulative change to be applied later
        self.dest_pending_updates += self.learning_rate*gradient_update

    def apply_gradient_updates(self):
        """At the end of an epoch, update the node embeddings
        based on the cached updates
        """

        # Update the embeddings
        # Regularisation penalty
        self.source_value -= 2*self.regularisation_rate*self.source_value
        self.dest_value -= 2*self.regularisation_rate*self.dest_value
        # Gradient ascent
        self.source_value += self.source_pending_updates
        self.dest_value += self.dest_pending_updates

        # Reset the cache
        self.source_pending_updates = np.zeros(self.dimension)
        self.dest_pending_updates = np.zeros(self.dimension)

    def clear_gradient_updates(self):
        """Reset the pending updates without applying them
        """

        # Reset the cache
        self.source_pending_updates = np.zeros(self.dimension)
        self.dest_pending_updates = np.zeros(self.dimension)

class EdgeEmbedder():
    """A class to combine two node embeddings into an edge embedding
    """
    def __init__(self, input_dimension, mode='concatenate'):
        """Initialise the class

        Args:
            input_dimension (int): The dimension of the node embeddings
            mode (str, optional): The method used to combine the node embeddings.
                Only concatenate is implemented currently.
                Defaults to 'concatenate'.

        Raises:
            ValueError: A valid mode must be provided.
        """

        # Store the input dimension
        self.input_dimension = input_dimension

        if mode=='concatenate':
            # Overwrite the embed_edge method (to save checking
            # the mode attribute every time it's called)
            self.embed_edge = self.concatenator

            # Concatenating the embeddings will double the
            # dimension
            self.output_dimension = input_dimension * 2
        else:
            raise ValueError(
                f'Edge embedder mode {mode} is not recognised'
            )

    def concatenator(self, x_i, x_j):
        """The edge embedding method for 'concatenate', which
        creates an edge embedding by stacking the node embeddings
        on top of each other.

        Args:
            x_i (np.array): The node embedding for the source
            x_j (np.array): The node emebedding for the destination

        Returns:
            np.array: The edge embedding
        """
        return np.hstack((x_i, x_j))


class EdgeComparer():
    """A class for generating positive, real numbers from two
    embeddings, with gradient-based learning functions
    """
    def __init__(self, dimension,
                 learning_rate=0.001, regularisation_rate=0.01,
                 mode='matrix', positive_output=True,
                 min_at=0):
        """Initialise the class

        Args:
            dimension (int): The dimension of the embeddings
            learning_rate (float, optional): The learning rate to use
                for the gradient-ascent algorithm. Defaults to 0.001.
            regularisation_rate (float, optional): The weight towards the
                L2 regularisation penalty. Defaults to 0.01.
            mode (str, optional): The method to use for generating a
                real-valued output. The 'vector' option (dot-product)
                is removed, and therefore only 'matrix' is implemented.
                Defaults to 'matrix'.
            positive_output (bool, optional): Whether a function should
                be applied to the output to ensure it is positive.
                Defaults to True.
            min_at (float, optional): The minimum value for the function.
                Defaults to 0.0.

        Raises:
            ValueError: A valid mode must be provided
        """

        # Record the dimension
        self.dimension = dimension

        # Record if the positive function should be applied
        self.positive_output = positive_output
        self.min_at = min_at

        if mode == 'matrix':
            # Initialise a random matrix
            max_value = 2/self.dimension
            self.matrix = \
                np.random.uniform(
                    0, max_value, (self.dimension, self.dimension)
                )

            # Overwrite the compare_embeddings method (to save
            # checking the mode attribute every time it is
            # called)
            self.compare_embeddings = self.matrix_form
        else:
            raise ValueError(
                f'Edge comparer mode {mode} is not recognised'
            )

        # To ensure that the output is positive, the result
        # of the linear function is passed through a smooth,
        # continuous function that is always positive. For
        # the gradient-ascent algorithm, the value of the
        # linear function is required.
        self.last_linear_value = None

        # Save the learning rates
        self.learning_rate = learning_rate
        self.regularisation_rate = regularisation_rate

        # Set up a variable to store pending gradient updates
        # (to be applied at the end of each epoch)
        self.pending_updates = np.zeros(
            (self.dimension, self.dimension)
            )

    def matrix_form(self, e_i, e_j):
        """The function used by the 'matrix' mode. A
        matrix is included inside a dot-product of the two
        embeddings, to allow interactions between embeddings
        and to allow features to be amplified, reversed or
        turned off (depending on the matrix coefficients)

        Args:
            e_i (np.array): First embedding
            e_j (np.array): Second embedding

        Returns:
            float: A positive real number
        """

        # Calculate the linear part of the function
        linear_value = e_i.T @ self.matrix @ e_j

        if self.positive_output:
            # Cache this value for the gradient-ascent algorithm
            self.last_linear_value = linear_value

            # Apply the piecewise function to make the output
            # certainly positive
            return log_exp_function(linear_value) + self.min_at
        else:
            return linear_value

    def add_gradient_update(self, gradient_update):
        """Store a gradient update, to be applied to the
        matrix at the end of the epoch.

        Args:
            gradient_update (np.array): The change in the matrix,
                as given by the gradient ascent algorithm for a
                single excitee
        """

        # Multiply the given change by the learning rate, and
        # record the cumulative change to be applied later
        self.pending_updates += self.learning_rate*gradient_update

    def reset(self, discard_gradient_updates=False):
        """Apply or discard gradient updates

        Args:
            discard_gradient_updates (bool, optional):
            Choose whether to discard or apply the pending gradient
            updates. Defaults to False.
        """

        if discard_gradient_updates:
            self.clear_gradient_updates()
        else:
            self.apply_gradient_updates()

    def apply_gradient_updates(self):
        """At the end of an epoch, update the matrix
        based on the cached updates
        """

        # Update the matrix
        # Regularisation penalty
        self.matrix -= 2*self.regularisation_rate*self.matrix
        # Gradient ascent
        self.matrix += self.pending_updates

        # Reset the cache
        self.pending_updates = np.zeros(
            (self.dimension, self.dimension)
            )

    def clear_gradient_updates(self):
        """Reset the pending updates without applying them
        """

        # Reset the cache
        self.pending_updates = np.zeros(
            (self.dimension, self.dimension)
            )
