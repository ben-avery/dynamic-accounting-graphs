import numpy as np


class Node():
    def __init__(self, name, opening_balance, dimension, meta_data=None):
        self.name = name
        self.opening_balance = opening_balance
        self.layer_balances = []
        self.dimension = dimension
        self.embeddings = NodeEmbedding(dimension=dimension)
        self.meta_data = meta_data


class NodeEmbedding():
    def __init__(self, dimension):
        self.source_value = np.random.uniform(0,1,dimension)
        self.dest_value = np.random.uniform(0,1,dimension)


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
    def __init__(self, dimension, mode='matrix'):
        if mode == 'matrix':
            self.matrix = np.random.uniform(-1,1,(dimension, dimension))
            self.compare_edges = self.matrix_form
        elif mode == 'vector':
            self.vector = np.random.uniform(-1,1,dimension)
            self.compare_edges = self.vector_form
        else:
            raise ValueError(
                f'Edge comparer mode {mode} is not recognised'
            )

    def matrix_form(self, e_i, e_j):
        return e_i.T @ self.matrix @ e_j

    def vector_form(self, e_i, e_j):
        return e_i.T @ np.multiply(self.vector, e_j)
