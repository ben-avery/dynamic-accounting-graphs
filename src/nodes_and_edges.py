"""Module for handling lower-level classes
"""
import numpy as np


class Node():
    """A class for a node, containing embeddings and gradient-based
    learning functions
    """
    def __init__(self, name, opening_balance, dimension,
                 learning_rate=0.0001, meta_data=None):
        """Initialise the class

        Args:
            name (str): The name of the node
            opening_balance (float): The monetary value accumulated in the
                associated account at the start of the period
            dimension (int): The dimension of the node embeddings
            learning_rate (float, optional): The learning rate for the
                gradient ascent algorithm. Defaults to 0.0001.
            meta_data (dict, optional): A dictionary of other information
                about a node which is irrelevant to the training. For example,
                it may be useful for downstream applications to know the
                mapping of the node to an associated accounting concept.
                Defaults to None.
        """

        # Store node parameters
        self.name = name
        self.opening_balance = opening_balance
        self.dimension = dimension
        self.meta_data = meta_data
        self.learning_rate = learning_rate

        # Create the embeddings of the node (as a source and
        # destination separately)
        self.embeddings = NodeEmbedding(dimension=dimension)

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
        """At the end of an epoch, update the two node embeddings
        based on the cached updates
        """

        # Update the embeddings
        self.embeddings.source_value += self.source_pending_updates
        self.embeddings.dest_value += self.dest_pending_updates

        # Reset the cache
        self.source_pending_updates = np.zeros(self.dimension)
        self.dest_pending_updates = np.zeros(self.dimension)

class NodeEmbedding():
    """A class to contain the source and destination node
    embedding for a particular node
    """
    def __init__(self, dimension):
        """Initialise the class

        Args:
            dimension (int): The dimension of the embeddings
        """

        # Initialise the embeddings randomly
        self.source_value = np.random.uniform(-1, 1, dimension)
        self.dest_value = np.random.uniform(-1, 1, dimension)


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
    edge embeddings, with gradient-based learning functions
    """
    def __init__(self, dimension, learning_rate=0.0001,
                 mode='matrix'):
        """Initialise the class

        Args:
            dimension (int): The dimension of the edge embeddings
            learning_rate (float, optional): The learning rate to use
                for the gradient-ascent algorithm. Defaults to 0.0001.
            mode (str, optional): The method to use for generating a
                real-valued output. The 'vector' option (dot-product)
                is removed, and therefore only 'matrix' is implemented.
                Defaults to 'matrix'.

        Raises:
            ValueError: A valid mode must be provided
        """

        # Record the dimension
        self.dimension = dimension

        if mode == 'matrix':
            # Initialise a random matrix
            self.matrix = \
                np.random.uniform(
                    -1, 1, (self.dimension, self.dimension)
                )

            # Overwrite the compare_edges method (to save
            # checking the mode attribute every time it is
            # called)
            self.compare_edges = self.matrix_form
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

        # Save the learning rate
        self.learning_rate = learning_rate

        # Set up a variable to store pending gradient updates
        # (to be applied at the end of each epoch)
        self.pending_updates = np.zeros(
            (self.dimension, self.dimension)
            )

    def matrix_form(self, e_i, e_j):
        """The function used by the 'matrix' mode. A
        matrix is included inside a dot-product of the two
        edge embeddings, to allow interactions between
        source and destination node embeddings and to
        allow features to be amplified, reversed  or
        turned off (depending on the matrix coefficients)

        Args:
            e_i (np.array): Excitor node embedding
            e_j (np.array): Excitee node embedding

        Returns:
            float: A positive real number
        """

        # Calculate the linear part of the function
        linear_value = e_i.T @ self.matrix @ e_j

        # Cache this value for the gradient-ascent algorithm
        self.last_linear_value = linear_value

        # Apply the piecewise function to make the output
        # certainly positive
        if linear_value < 0:
            # The smooth, continuous function is exponential
            # for negative inputs...
            return np.exp(linear_value)
        else:
            # ... and logarithmic for positive inputs.
            return np.log(linear_value + 1) + 1

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

    def apply_gradient_updates(self):
        """At the end of an epoch, update the matrix
        based on the cached updates
        """

        # Update the matrix
        self.matrix += self.pending_updates

        # Reset the cache
        self.pending_updates = np.zeros(
            (self.dimension, self.dimension)
            )
