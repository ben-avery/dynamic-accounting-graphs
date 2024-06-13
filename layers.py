from nodes_and_edges import EdgeSimilarity
import numpy as np

from utilities import discrete_weibull_pmf


class Layer():
    def __init__(self, nodes, edge_embedder,
                 edge_similarity_mode='matrix'):
        self.nodes = nodes
        self.edge_embedder = edge_embedder
        self.edge_similarity = EdgeSimilarity(
            dimension=self.edge_embedder.output_dimension,
            mode=edge_similarity_mode
        )
        self.edge_log = []
        self.edge_intensity_log = []

    def add_edge(self, i, j, edge_time, edge_weight, edge_intensity=None):
        # Get the node embeddings
        x_i = self.nodes[i].embeddings.source_value
        x_j = self.nodes[i].embeddings.dest_value

        # Get the edge embedding
        edge_embedding = self.edge_embedder.embed_edge(x_i, x_j)
        self.edge_log.append(
            (i, j, edge_time, edge_weight, edge_embedding))

        # Record the edge intensity, when learning
        if edge_intensity is not None:
            self.edge_intensity_log = edge_intensity

    def edge_intensity(self, i, j, edge_time):
        # Get the node embeddings
        x_i = self.nodes[i].embeddings.source_value
        x_j = self.nodes[j].embeddings.dest_value

        # Get the edge embedding
        edge_embedding = self.edge_embedder.embed_edge(x_i, x_j)

        intensity = 0
        for edge_details in self.edge_log:
            curr_i, curr_j, curr_edge_time, \
                curr_edge_weight, curr_edge_embedding = edge_details

            if curr_edge_embedding < edge_time:
                edge_similarity = self.edge_similarity.return_similarity(
                    edge_embedding, curr_edge_embedding
                )

                intensity += \
                    edge_similarity * discrete_weibull_pmf(
                        edge_time-curr_edge_time,
                        alpha=1, beta=1
                    )

        return intensity

    def edge_probability(self, i, j, edge_time):
        pass
