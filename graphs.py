from nodes_and_edges import Node, EdgeEmbedder
from layers import Layer
from utilities import soft_max
import numpy as np


class DynamicAccountingGraph():
    def __init__(self, accounts, node_dimension,
                 edge_embedder_mode='concatenate',
                 edge_similarity_mode='matrix'):
        self.nodes = []
        for account in accounts:
            node = Node(
                name=account.name,
                opening_balance=account.balance,
                dimension=node_dimension,
                meta_data={
                    'account_number': account.number,
                    'mapping': account.mapping
                }
            )

            self.nodes.append(node)

        self.edge_embedder = EdgeEmbedder(
            input_dimension=node_dimension,
            mode=edge_embedder_mode
        )

        self.edge_similarity_mode = edge_similarity_mode
        self.layers = []

    def add_new_layer(self):
        self.layers.append(Layer(
            nodes=self.nodes,
            edge_embedder=self.edge_embedder,
            edge_similarity_mode=self.edge_similarity_mode
        ))

    def seed_intensity(self, i, j, edge_time):
        return 0.01

    def add_edge(self, i, j, edge_time, edge_weight):
        # Find the intensities of the edge in each layer
        intensities = [
            layer.edge_intensity(
                i, j, edge_time
            )
            for layer in self.layers
        ]
        
        # Find the intensity of the edge in the seed layer
        seed_intensity = self.seed_intensity(
            i, j, edge_time
        )

        # Soft-max all the intensities
        softmax_intensities = \
            soft_max(np.array(intensities+[seed_intensity]))

        # Allocate the edge accordingly (where the value is equal to
        # more than a penny)
        threshold = 0.01/edge_weight
        remaining_weight = edge_weight
        for layer_index, intensity in enumerate(softmax_intensities[:-1]):
            if intensity > threshold:
                layer_weight = min(
                    np.round(edge_weight*intensity,2),
                    remaining_weight
                )
                if layer_weight >= 0.01:
                    self.layers[layer_index].add_edge(
                        i, j, edge_time, layer_weight
                    )
                    remaining_weight -= layer_weight

        # If there is any weight remaining, it belongs to the seed
        # layer. A new layer is formed with this edge.
        if remaining_weight >= 0.01:
            self.add_new_layer()

            self.layers[-1].add_edge(
                        i, j, edge_time, remaining_weight
                    )
