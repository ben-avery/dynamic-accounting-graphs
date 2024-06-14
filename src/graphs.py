from scipy.stats import poisson
from numpy import log

from nodes_and_edges import Node, EdgeEmbedder, EdgeComparer
from excitement import Excitement
from utilities import (
    calc_delP_delIntensity, calc_delIntensity_delAlpha,
    calc_delIntensity_delBeta, calc_delIntensity_delWeight,
    calc_delComparer_delI, calc_delComparer_delJ,
    calc_delComparer_delK, calc_delComparer_delL,
    calc_delComparer_delMatrix
)


class DynamicAccountingGraph():
    def __init__(self, accounts, node_dimension,
                 edge_embedder_mode='concatenate',
                 weibull_weight_generator_mode='matrix',
                 weibull_alpha_generator_mode='matrix',
                 weibull_beta_generator_mode='matrix'):
        self.time = 0

        self.nodes = []
        self.node_dimension = node_dimension
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

        self.gradient_log = dict()

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
                        lin_val_weight = self.weibull_weight_generator.last_linear_value

                        # If there is sufficient connection, add as an excitee
                        if weibull_weight > self.excitment_threshold:
                            weibull_alpha = \
                                self.weibull_alpha_generator.compare_edges(
                                    excitor_edge_embedding, excitee_edge_embedding
                                )
                            lin_val_alpha = self.weibull_alpha_generator.last_linear_value
                            weibull_beta = \
                                self.weibull_beta_generator.compare_edges(
                                    excitor_edge_embedding, excitee_edge_embedding
                                )
                            lin_val_beta = self.weibull_beta_generator.last_linear_value

                            self.possible_excitees[
                                (excitor_i, excitor_j)
                                ][
                                    (excitee_i, excitee_j)
                                    ] = \
                                        ((weibull_weight, weibull_alpha, weibull_beta),
                                         (lin_val_weight, lin_val_alpha, lin_val_beta))

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
            weibull_params, lin_val_params = excitee_parameters
            weibull_weight, weibull_alpha, weibull_beta = weibull_params
            lin_val_weight, lin_val_alpha, lin_val_beta = lin_val_params

            if excitee_nodes not in self.current_excitees:
                self.current_excitees[excitee_nodes] = []

            self.current_excitees[excitee_nodes].append(
                Excitement(
                    weibull_weight=weibull_weight,
                    weibull_alpha=weibull_alpha,
                    weibull_beta=weibull_beta,
                    lin_val_weight=lin_val_weight,
                    lin_val_alpha=lin_val_alpha,
                    lin_val_beta=lin_val_beta,
                    source_nodes=(i,j),
                    dest_nodes=excitee_nodes,
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

            self.gradient_log['alphas'] = \
                [excite.weibull_alpha
                 for excite in self.current_excitees[(i,j)]]
            self.gradient_log['betas'] = \
                [excite.weibull_beta
                 for excite in self.current_excitees[(i,j)]]
            self.gradient_log['weights'] = \
                [excite.weibull_weight
                 for excite in self.current_excitees[(i,j)]]
            self.gradient_log['lin_alphas'] = \
                [excite.lin_val_alpha
                 for excite in self.current_excitees[(i,j)]]
            self.gradient_log['lin_betas'] = \
                [excite.lin_val_beta
                 for excite in self.current_excitees[(i,j)]]
            self.gradient_log['lin_weights'] = \
                [excite.lin_val_weight
                 for excite in self.current_excitees[(i,j)]]
            self.gradient_log['times'] = \
                [excite.time
                 for excite in self.current_excitees[(i,j)]]
            self.gradient_log['source_nodes'] = \
                [excite.source_nodes
                 for excite in self.current_excitees[(i,j)]]
        else:
            self.gradient_log['alphas'] = []
            self.gradient_log['betas'] = []
            self.gradient_log['weights'] = []
            self.gradient_log['times'] = []
            self.gradient_log['source_nodes'] = []

        self.gradient_log['sum_Intensity'] = intensity

        return intensity

    def edge_probability(self, i, j, count):
        self.gradient_log['k'] = i
        self.gradient_log['l'] = j
        self.gradient_log['count'] = count


        probability = poisson.pmf(
            k=count,
            mu=self.edge_intensity(i, j)
        )

        self.gradient_ascent()

        return probability

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

    def gradient_ascent(self):
        # Partials
        delP_delIntensity= \
            calc_delP_delIntensity(
                self.gradient_log['count'],
                self.gradient_log['sum_Intensity'])
        delIntensity_delAlpha = [
            calc_delIntensity_delAlpha(
                time,
                self.gradient_log['alphas'][excite_index],
                self.gradient_log['betas'][excite_index],
                self.gradient_log['weights'][excite_index]
            )
            for excite_index, time in enumerate(self.gradient_log['times'])
        ]
        delIntensity_delBeta = [
            calc_delIntensity_delBeta(
                time,
                self.gradient_log['alphas'][excite_index],
                self.gradient_log['betas'][excite_index],
                self.gradient_log['weights'][excite_index]
            )
            for excite_index, time in enumerate(self.gradient_log['times'])
        ]
        delIntensity_delWeight = [
            calc_delIntensity_delWeight(
                time,
                self.gradient_log['alphas'][excite_index],
                self.gradient_log['betas'][excite_index]
            )
            for excite_index, time in enumerate(self.gradient_log['times'])
        ]

        k = self.gradient_log['k']
        l = self.gradient_log['l']

        # Get the node embeddings
        node_k = self.nodes[k]
        x_k = node_k.embeddings.dest_value
        node_l = self.nodes[l]
        x_l = node_l.embeddings.dest_value

        # Get the edge embedding
        e_kl = self.edge_embedder.embed_edge(x_k, x_l)

        for excite_index, (i, j) in enumerate(self.gradient_log['source_nodes']):
            # Get linear values
            lin_val_alpha = self.gradient_log['lin_alphas'][excite_index]
            lin_val_beta = self.gradient_log['lin_betas'][excite_index]
            lin_val_weight = self.gradient_log['lin_weights'][excite_index]

            # Get the node embeddings
            node_i = self.nodes[i]
            x_i = node_i.embeddings.dest_value
            node_j = self.nodes[j]
            x_j = node_j.embeddings.dest_value

            # Get the edge embedding
            e_ij = self.edge_embedder.embed_edge(x_i, x_j)

            delAlpha_delI = \
                calc_delComparer_delI(
                    linear_value=lin_val_alpha,
                    matrix=self.weibull_alpha_generator.matrix,
                    e_kl=e_kl,
                    node_dimension=self.node_dimension
                )
            delBeta_delI = \
                calc_delComparer_delI(
                    linear_value=lin_val_beta,
                    matrix=self.weibull_beta_generator.matrix,
                    e_kl=e_kl,
                    node_dimension=self.node_dimension
                )
            delWeight_delI = \
                calc_delComparer_delI(
                    linear_value=lin_val_weight,
                    matrix=self.weibull_weight_generator.matrix,
                    e_kl=e_kl,
                    node_dimension=self.node_dimension
                )

            delAlpha_delJ = \
                calc_delComparer_delJ(
                    linear_value=lin_val_alpha,
                    matrix=self.weibull_alpha_generator.matrix,
                    e_kl=e_kl,
                    node_dimension=self.node_dimension
                )
            delBeta_delJ = \
                calc_delComparer_delJ(
                    linear_value=lin_val_beta,
                    matrix=self.weibull_beta_generator.matrix,
                    e_kl=e_kl,
                    node_dimension=self.node_dimension
                )
            delWeight_delJ = \
                calc_delComparer_delJ(
                    linear_value=lin_val_weight,
                    matrix=self.weibull_weight_generator.matrix,
                    e_kl=e_kl,
                    node_dimension=self.node_dimension
                )

            delAlpha_delK = \
                calc_delComparer_delK(
                    linear_value=lin_val_alpha,
                    matrix=self.weibull_alpha_generator.matrix,
                    e_ij=e_ij,
                    node_dimension=self.node_dimension
                )
            delBeta_delK = \
                calc_delComparer_delK(
                    linear_value=lin_val_beta,
                    matrix=self.weibull_beta_generator.matrix,
                    e_ij=e_ij,
                    node_dimension=self.node_dimension
                )
            delWeight_delK = \
                calc_delComparer_delK(
                    linear_value=lin_val_weight,
                    matrix=self.weibull_weight_generator.matrix,
                    e_ij=e_ij,
                    node_dimension=self.node_dimension
                )

            delAlpha_delL = \
                calc_delComparer_delL(
                    linear_value=lin_val_alpha,
                    matrix=self.weibull_alpha_generator.matrix,
                    e_ij=e_ij,
                    node_dimension=self.node_dimension
                )
            delBeta_delL = \
                calc_delComparer_delL(
                    linear_value=lin_val_beta,
                    matrix=self.weibull_beta_generator.matrix,
                    e_ij=e_ij,
                    node_dimension=self.node_dimension
                )
            delWeight_delL = \
                calc_delComparer_delL(
                    linear_value=lin_val_weight,
                    matrix=self.weibull_weight_generator.matrix,
                    e_ij=e_ij,
                    node_dimension=self.node_dimension
                )

            delAlphaComparerdelMatrix = \
                calc_delComparer_delMatrix(
                    linear_value=lin_val_alpha,
                    e_ij=e_ij,
                    e_kl=e_kl,
                )

            delBetaComparerdelMatrix = \
                calc_delComparer_delMatrix(
                    linear_value=lin_val_beta,
                    e_ij=e_ij,
                    e_kl=e_kl,
                )

            delWeightComparerdelMatrix = \
                calc_delComparer_delMatrix(
                    linear_value=lin_val_weight,
                    e_ij=e_ij,
                    e_kl=e_kl,
                )

            # Node i
            node_i.add_source_gradient_update(
                delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delI +
                    delIntensity_delBeta[excite_index] *
                    delBeta_delI +
                    delIntensity_delWeight[excite_index] *
                    delWeight_delI
                )
            )

            # Node j
            node_j.add_source_gradient_update(
                delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delJ +
                    delIntensity_delBeta[excite_index] *
                    delBeta_delJ +
                    delIntensity_delWeight[excite_index] *
                    delWeight_delJ
                )
            )

            # Node k
            node_k.add_dest_gradient_update(
                delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delK +
                    delIntensity_delBeta[excite_index] *
                    delBeta_delK +
                    delIntensity_delWeight[excite_index] *
                    delWeight_delK
                )
            )

            # Node l
            node_l.add_dest_gradient_update(
                delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delL +
                    delIntensity_delBeta[excite_index] *
                    delBeta_delL +
                    delIntensity_delWeight[excite_index] *
                    delWeight_delL
                )
            )

            # Weight matrix
            self.weibull_weight_generator.add_gradient_update(
                delP_delIntensity *
                delIntensity_delAlpha[excite_index] *
                delWeightComparerdelMatrix
            )

            # Alpha matrix
            self.weibull_alpha_generator.add_gradient_update(
                delP_delIntensity *
                delIntensity_delBeta[excite_index] *
                delAlphaComparerdelMatrix
            )

            # Beta matrix
            self.weibull_beta_generator.add_gradient_update(
                delP_delIntensity *
                delIntensity_delWeight[excite_index] *
                delBetaComparerdelMatrix
            )

        self.gradient_log = dict()

    def apply_gradient_updates(self):
        for node in self.nodes:
            node.apply_gradient_updates()

        self.weibull_alpha_generator.apply_gradient_updates()
        self.weibull_beta_generator.apply_gradient_updates()
        self.weibull_weight_generator.apply_gradient_updates()

    def reset(self):
        self.time = 0
        self.edge_log = []

        self.new_edges = dict()

        self.possible_excitees = dict()
        self.current_excitees = dict()

        # Apply gradient updates
        self.apply_gradient_updates()

        # Update excitors
        self.find_excitors()
