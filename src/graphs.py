from nodes_and_edges import Node, EdgeEmbedder, EdgeComparer
from excitement import Excitement
from scipy.stats import poisson
from numpy import log


class DynamicAccountingGraph():
    def __init__(self, accounts, node_dimension,
                 edge_embedder_mode='concatenate',
                 weibull_weight_generator_mode='matrix',
                 weibull_alpha_generator_mode='matrix',
                 weibull_beta_generator_mode='matrix'):
        self.time = 0

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
        self.count_nodes = len(self.nodes)

        self.edge_embedder = EdgeEmbedder(
            input_dimension=node_dimension,
            mode=edge_embedder_mode
        )

        self.weibull_weight_generator = EdgeComparer(
            dimension=self.edge_embedder.output_dimension,
            mode=weibull_weight_generator_mode
        )
        self.weibull_alpha_generator = EdgeComparer(
            dimension=self.edge_embedder.output_dimension,
            mode=weibull_alpha_generator_mode
        )
        self.weibull_beta_generator = EdgeComparer(
            dimension=self.edge_embedder.output_dimension,
            mode=weibull_beta_generator_mode
        )

        self.edge_log = []

        self.new_edges = dict()

        self.possible_excitees = dict()
        self.excitment_threshold = 0.01

        self.current_excitees = dict()
        self.find_excitors()

    def find_excitors(self):
        # Clear the current excitors
        self.possible_excitees = dict()

        for excitor_i in range(self.count_nodes):
            # Get the node embedding
            excitor_x_i = self.nodes[excitor_i].embeddings.source_value

            for excitor_j in range(self.count_nodes):
                # Get the node embedding
                excitor_x_j = self.nodes[excitor_j].embeddings.dest_value

                # Get the edge embedding
                excitor_edge_embedding = self.edge_embedder.embed_edge(excitor_x_i, excitor_x_j)

                # Create an entry for this excitor
                self.possible_excitees[(excitor_i, excitor_j)] = dict()

                for excitee_i in range(self.count_nodes):
                    # Get the node embedding
                    excitee_x_i = self.nodes[excitee_i].embeddings.source_value

                    for excitee_j in range(self.count_nodes):
                        # Get the node embedding
                        excitee_x_j = self.nodes[excitee_j].embeddings.dest_value

                        # Get the edge embedding
                        excitee_edge_embedding = \
                            self.edge_embedder.embed_edge(excitee_x_i, excitee_x_j)

                        # Get the Weibull weight
                        weibull_weight = \
                            self.weibull_weight_generator.compare_edges(
                                excitor_edge_embedding, excitee_edge_embedding
                            )

                        # If there is sufficient connection, add as an excitee
                        if weibull_weight > self.excitment_threshold:
                            weibull_alpha = \
                                self.weibull_alpha_generator.compare_edges(
                                    excitor_edge_embedding, excitee_edge_embedding
                                )
                            weibull_beta = \
                                self.weibull_beta_generator.compare_edges(
                                    excitor_edge_embedding, excitee_edge_embedding
                                )

                            self.possible_excitees[
                                (excitor_i, excitor_j)
                                ][
                                    (excitee_i, excitee_j)
                                    ] = \
                                        (weibull_weight, weibull_alpha, weibull_beta)

    def increment_time(self):
        self.time += 1

        # Increment time for the excitees
        for excites in self.current_excitees.values():
            for excite in excites:
                excite.increment_time()

        # Remove any dead excitees
        self.current_excitees = {
            excitee: [
                excite
                for excite in excites
                if excite.alive
            ]
            for excitee, excites in self.current_excitees.items()
        }

        # Remove any empty excitees
        self.current_excitees = {
            excitee: excites
            for excitee, excites in self.current_excitees.items()
            if len(excites) > 0
        }

        # Reset the new edges counter
        self.new_edges = dict()

    def add_edge(self, i, j, edge_time, edge_weight):
        # Get the node embeddings
        x_i = self.nodes[i].embeddings.source_value
        x_j = self.nodes[i].embeddings.dest_value

        # Save the edge
        self.edge_log.append(
            (i, j, edge_time, edge_weight))

        # Add the new edge to today's counter
        if (i, j) in self.new_edges:
            self.new_edges[(i, j)] += 1
        else:
            self.new_edges[(i, j)] = 1

        # Record any excitees
        for excitee_nodes, excitee_parameters in self.possible_excitees[(i,j)].items():
            weibull_weight, weibull_alpha, weibull_beta = excitee_parameters

            if excitee_nodes not in self.current_excitees:
                self.current_excitees[excitee_nodes] = []

            self.current_excitees[excitee_nodes].append(
                Excitement(
                    weibull_weight=weibull_weight,
                    weibull_alpha=weibull_alpha,
                    weibull_beta=weibull_beta,
                    alive_threshold=self.excitment_threshold
                )
            )

    def edge_baseline(self, i, j):
        return 0.01

    def edge_intensity(self, i, j):
        intensity = self.edge_baseline(i, j)

        nodes = (i,j)
        if nodes in self.current_excitees:
            intensity += sum(
                excite.probability
                for excite in self.current_excitees[(i,j)]
                )

        return intensity

    def edge_probability(self, i, j, count):
        return poisson.pmf(
            k=count,
            mu=self.edge_intensity(i, j)
        )

    def day_log_probability(self):
        log_probability = 0
        for i in range(self.count_nodes):
            for j in range(self.count_nodes):
                if (i, j) in self.new_edges:
                    count = self.new_edges[(i, j)]
                else:
                    count = 0
                
                log_probability += log(
                    self.edge_probability(
                        i, j, count
                    )
                    )

        return log_probability
