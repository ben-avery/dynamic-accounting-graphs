import numpy as np


class Node():
    def __init__(self, name, opening_balance, dimension,
                 learning_rate=0.0001, meta_data=None):
        self.name = name
        self.opening_balance = opening_balance
        self.dimension = dimension
        self.embeddings = NodeEmbedding(dimension=dimension)
        self.meta_data = meta_data

        self.learning_rate = learning_rate

        self.source_pending_updates = np.zeros(self.dimension)
        self.dest_pending_updates = np.zeros(self.dimension)

    def add_source_gradient_update(self, gradient_update):
        self.source_pending_updates += self.learning_rate*gradient_update

    def add_dest_gradient_update(self, gradient_update):
        self.dest_pending_updates += self.learning_rate*gradient_update

    def apply_gradient_updates(self):
        self.embeddings.source_value += self.source_pending_updates
        self.embeddings.dest_value += self.dest_pending_updates

        self.source_pending_updates = np.zeros(self.dimension)
        self.dest_pending_updates = np.zeros(self.dimension)


class NodeEmbedding():
    def __init__(self, dimension):
        self.source_value = np.random.uniform(-1, 1, dimension)
        self.dest_value = np.random.uniform(-1, 1, dimension)


class EdgeEmbedder():
    def __init__(self, input_dimension, mode='concatenate'):
        self.input_dimension = input_dimension

        if mode=='concatenate':
            self.embed_edge = self.concatenator
            self.output_dimension = input_dimension * 2
        else:
            raise ValueError(
                f'Edge embedder mode {mode} is not recognised'
            )

    def concatenator(self, x_i, x_j):
        return np.hstack((x_i, x_j))


class EdgeComparer():
    def __init__(self, dimension, learning_rate=0.0001,
                 mode='matrix'):
        self.dimension = dimension

        if mode == 'matrix':
            self.matrix = \
                np.random.uniform(
                    -1, 1, (self.dimension, self.dimension)
                )
            self.compare_edges = self.matrix_form
        else:
            raise ValueError(
                f'Edge comparer mode {mode} is not recognised'
            )

        self.learning_rate = learning_rate
        self.pending_updates = np.zeros(
            (self.dimension, self.dimension)
            )

    def matrix_form(self, e_i, e_j):
        return max(0.0001, e_i.T @ self.matrix @ e_j)

    def add_gradient_update(self, gradient_update):
        self.pending_updates += self.learning_rate*gradient_update

    def apply_gradient_updates(self):
        self.matrix += self.pending_updates

        self.pending_updates = np.zeros(
            (self.dimension, self.dimension)
            )
