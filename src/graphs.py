"""Main module for the Dynamic Accounting Graph class
"""
import csv
from scipy.stats import poisson
from numpy import log

from nodes_and_edges import Node, EdgeEmbedder, EdgeComparer
from excitement import Excitement
from utilities import (
    calc_delP_delIntensity, calc_delIntensity_delAlpha,
    calc_delIntensity_delBeta, calc_delIntensity_delWeight,
    calc_delComparer_delI, calc_delComparer_delJ,
    calc_delComparer_delK, calc_delComparer_delL,
    calc_delComparer_delMatrix,
    calc_delBaselineIntensity_delZero,
    calc_delBaselineIntensity_delOne,
    calc_delBaselineIntensity_delTwo,
    calc_delBaselineComparer_delK,
    calc_delBaselineComparer_delL,
    calc_delBaselineComparer_delMatrix,
    calc_delCausalDotproduct_delParam,
    calc_delBaselineDotproduct_delParam,
    log_exp_function
)


class DynamicAccountingGraph():
    """Class for a Dynamic Accounting Graph
    """
    def __init__(self, accounts, node_dimension,
                 mode='dot', learning_rate=0.001, regularisation_rate=0.01,
                 debug_mode=False, debug_edges_to_monitor=[],
                 debug_monitor_excitement=False):
        """Initialise the class

        Args:
            accounts (list): A list of Account object (with attributes 
                for name, balance, number and mapping)
            node_dimension (int): The dimension for the node embeddings
            mode (str, optional): The method for generating parameter
                values from embeddings. Can either be 'matrix' or 'dot'.
                Defaults to 'matrix'.
            learning_rate (float, optional): The learning rate for the
                gradient ascent algorithm. Defaults to 0.001.
            regularisation_rate (float, optional): The weight towards the
                L2 regularisation penalty. Defaults to 0.01.
            debug_mode (bool, optional): Whether to output details of the
                calculations to a debug file. Defaults to False.
        """
        self.time = 0
        self.epoch = 0

        self.learning_rate = learning_rate
        self.regularisation_rate = regularisation_rate

        # Get the modes
        self.mode = mode
        if self.mode == 'matrix':
            node_mode = 'matrix'
            embedder_mode = 'concatenate'
            comparer_mode = 'matrix'

            self.calculate_derivatives_mode = self.calculate_derivatives_matrix
        elif self.mode == 'dot':
            node_mode = 'dot'
            embedder_mode = 'hadamard'
            comparer_mode = 'dot'

            self.calculate_derivatives_mode = self.calculate_derivatives_dot
        else:
            raise ValueError(
                f'Graph mode {self.mode} is not recognised'
            )

        # Create the nodes with random embeddings
        self.nodes = []
        self.node_dimension = node_dimension
        for account in accounts:
            node = Node(
                name=account.name,
                opening_balance=account.balance,
                dimension=node_dimension,
                learning_rate=self.learning_rate,
                regularisation_rate=self.regularisation_rate,
                mode=node_mode,
                meta_data={
                    'account_number': account.number,
                    'mapping': account.mapping
                }
            )

            self.nodes.append(node)
        self.count_nodes = len(self.nodes)

        # Create an edge embedder which turns two
        # node embeddings into an embedding for an
        # edge between the nodes
        self.edge_embedder = EdgeEmbedder(
            input_dimension=node_dimension,
            mode=embedder_mode
        )

        if self.mode == 'matrix':
            # Create generators for the Weibull distribution
            # parameters
            self.weibull_weight_generator = EdgeComparer(
                dimension=self.edge_embedder.output_dimension,
                learning_rate=self.learning_rate,
                regularisation_rate=self.regularisation_rate,
                mode=comparer_mode
            )
            self.weibull_alpha_generator = EdgeComparer(
                dimension=self.edge_embedder.output_dimension,
                learning_rate=self.learning_rate,
                regularisation_rate=self.regularisation_rate,
                mode=comparer_mode,
                min_at=0.5
            )
            self.weibull_beta_generator = EdgeComparer(
                dimension=self.edge_embedder.output_dimension,
                learning_rate=self.learning_rate,
                regularisation_rate=self.regularisation_rate,
                mode=comparer_mode,
                min_at=1
            )

            # Create generators for linear parameters of
            # baseline intensity
            self.base_param_0 = EdgeComparer(
                dimension=self.node_dimension,
                learning_rate=self.learning_rate,
                regularisation_rate=self.regularisation_rate,
                mode=comparer_mode, positive_output=False
            )
            self.base_param_1 = EdgeComparer(
                dimension=self.node_dimension,
                learning_rate=self.learning_rate,
                regularisation_rate=self.regularisation_rate,
                mode=comparer_mode, positive_output=False
            )
            self.base_param_2 = EdgeComparer(
                dimension=self.node_dimension,
                learning_rate=self.learning_rate,
                regularisation_rate=self.regularisation_rate,
                mode=comparer_mode, positive_output=False
            )
        elif self.mode == 'dot':
            self.causal_comparer_weight = EdgeComparer(
                dimension=self.node_dimension,
                learning_rate=self.learning_rate,
                regularisation_rate=self.regularisation_rate,
                mode=comparer_mode, positive_output=True
            )
            self.causal_comparer_alpha = EdgeComparer(
                dimension=self.node_dimension,
                learning_rate=self.learning_rate,
                regularisation_rate=self.regularisation_rate,
                mode=comparer_mode, positive_output=True,
                min_at=0.5
            )
            self.causal_comparer_beta = EdgeComparer(
                dimension=self.node_dimension,
                learning_rate=self.learning_rate,
                regularisation_rate=self.regularisation_rate,
                mode=comparer_mode, positive_output=True,
                min_at=1.0
            )

            self.spontaneous_comparer = EdgeComparer(
                dimension=self.node_dimension,
                learning_rate=self.learning_rate,
                regularisation_rate=self.regularisation_rate,
                mode=comparer_mode, positive_output=False
            )

        # Create attributes to record all edges and any
        # new edges for this day
        self.edge_log = []
        self.new_edges = dict()

        # Store any pairs of edges which will excite each
        # other based on the weight of the corresponding
        # Weibull distribution
        self.possible_excitees = dict()
        self.excitement_threshold = 0.0001
        self.find_excitors()

        # Create an attribute to store any edges that are
        # currently excited by previous edges
        self.current_excitees = dict()

        # Create an attribute to store gradients and values
        # for the gradient ascent algorithm
        self.gradient_log = dict()

        # Set up debug mode
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.debug_log = {}
            self.debug_edges_to_monitor = debug_edges_to_monitor
            self.debug_monitor_excitement = debug_monitor_excitement

    def add_to_debug_log(self, attribute, value, edges):
        """Add an entry in the debug log

        Args:
            attribute (str): The name of the information being saved
            value (any): The information being saved
            edges (list): List of tuples (i,j) for relevant edges
        """

        if any(edge in self.debug_edges_to_monitor for edge in edges) or edges == []:
            time_identifier = (self.epoch, self.time)

            if time_identifier not in self.debug_log:
                self.debug_log[time_identifier] = {}

            if attribute in self.debug_log[time_identifier]:
                raise ValueError(f'Attribute {attribute} is already recorded '
                                f'for epoch {self.epoch}, time {self.time}.')
            else:
                self.debug_log[time_identifier][attribute] = value

    def print_debug_log(self, file_path):
        """Save the debug log to a CSV file

        Args:
            file_path (str): File path for the CSV file
        """

        # Find all the attribute across all time
        all_attributes = set()
        for time_identifier, datapoints in self.debug_log.items():
            for attribute in datapoints:
                all_attributes.add(attribute)

        # Back fill missing attributes with 'None'
        new_datapoints = {}
        for time_identifier, datapoints in self.debug_log.items():
            new_datapoints[time_identifier] = set()
            for attribute in all_attributes:
                if attribute not in datapoints:
                    new_datapoints[time_identifier].add(attribute)
        for time_identifier, missing_attributes in new_datapoints.items():
            for attribute in missing_attributes:
                self.debug_log[time_identifier][attribute] = None

        # Output to a CSV file
        with open(file_path, 'w', newline='') as csvfile:
            # Create the CSV writer
            writer = csv.writer(csvfile)

            # Write the headings
            writer.writerow(['Epoch', 'Time']+list(all_attributes))

            # Add one row for each time step
            for time_identifier, datapoints in self.debug_log.items():
                # Get the time details from the identifier
                epoch, time = time_identifier

                # Add the logged data itself
                new_row = [epoch, time] + \
                    [datapoints[attribute] for attribute in all_attributes]

                # Save it to the CSV file
                writer.writerow(new_row)

    def find_excitors(self):
        """Find any pairs of edges which will excite each
        other based on the weight of the corresponding
        Weibull distribution
        """
        # Clear the current excitors
        self.possible_excitees = dict()

        for excitor_i in range(self.count_nodes):
            # Get the node embedding
            if self.mode == 'matrix':
                r_i = self.nodes[excitor_i].causal_source.value
            elif self.mode == 'dot':
                r_i = self.nodes[excitor_i].causal_excitor_source_weight.value

            for excitor_j in range(self.count_nodes):
                # Get the node embedding
                if self.mode == 'matrix':
                    r_j = self.nodes[excitor_j].causal_dest.value
                elif self.mode == 'dot':
                    r_j = self.nodes[excitor_j].causal_excitor_dest_weight.value

                # Get the edge embedding
                excitor_edge_embedding = self.edge_embedder.embed_edge(r_i, r_j)

                # Create an entry for this excitor
                self.possible_excitees[(excitor_i, excitor_j)] = dict()

                for excitee_k in range(self.count_nodes):
                    # Get the node embedding
                    if self.mode == 'matrix':
                        e_k = self.nodes[excitee_k].causal_source.value
                    elif self.mode == 'dot':
                        e_k = self.nodes[excitee_k].causal_excitee_source_weight.value

                    for excitee_l in range(self.count_nodes):
                        # Get the node embedding
                        if self.mode == 'matrix':
                            e_l = self.nodes[excitee_l].causal_dest.value
                        elif self.mode == 'dot':
                            e_l = self.nodes[excitee_l].causal_excitee_dest_weight.value

                        # Get the edge embedding
                        excitee_edge_embedding = \
                            self.edge_embedder.embed_edge(e_k, e_l)

                        # Get the Weibull weight (the total probability of the excitor
                        # edge exciting the excitee edge)
                        if self.mode == 'matrix':
                            weibull_weight = \
                                self.weibull_weight_generator.compare_embeddings(
                                    excitor_edge_embedding, excitee_edge_embedding
                                )

                            # Extract the pre-scaled value from the calculation, for use
                            # in the gradient ascent algorithm
                            lin_val_weight = self.weibull_weight_generator.last_linear_value
                        elif self.mode == 'dot':
                            weibull_weight = \
                                self.causal_comparer_weight.compare_embeddings(
                                    excitor_edge_embedding, excitee_edge_embedding
                                )

                            # Extract the pre-scaled value from the calculation, for use
                            # in the gradient ascent algorithm
                            lin_val_weight = self.causal_comparer_weight.last_linear_value

                        # If there is sufficient probability, add as a possible excitee
                        if weibull_weight > self.excitement_threshold:
                            # Calculate the Weibull parameters (characterising the average
                            # time to excitation, and the spread of likely times to
                            # excitation)
                            if self.mode == 'matrix':
                                weibull_alpha = \
                                    self.weibull_alpha_generator.compare_embeddings(
                                        excitor_edge_embedding, excitee_edge_embedding
                                    )
                                weibull_beta = \
                                    self.weibull_beta_generator.compare_embeddings(
                                        excitor_edge_embedding, excitee_edge_embedding
                                    )
                                # Extract the pre-scaled value from the calculation, for use
                                # in the gradient ascent algorithm
                                lin_val_alpha = self.weibull_alpha_generator.last_linear_value
                                lin_val_beta = self.weibull_beta_generator.last_linear_value
                            elif self.mode == 'dot':
                                # Alpha
                                # Get the node embeddings
                                r_i_alpha = self.nodes[excitor_i].causal_excitor_source_alpha.value
                                r_j_alpha = self.nodes[excitor_j].causal_excitor_dest_alpha.value
                                e_k_alpha = self.nodes[excitee_k].causal_excitee_source_alpha.value
                                e_l_alpha = self.nodes[excitee_l].causal_excitee_dest_alpha.value

                                # Get the edge embeddings
                                excitor_edge_embedding_alpha = \
                                    self.edge_embedder.embed_edge(r_i_alpha, r_j_alpha)
                                excitee_edge_embedding_alpha = \
                                    self.edge_embedder.embed_edge(e_k_alpha, e_l_alpha)

                                # Compare the embeddings
                                weibull_alpha = \
                                    self.causal_comparer_alpha.compare_embeddings(
                                        excitor_edge_embedding_alpha, excitee_edge_embedding_alpha
                                    )

                                # Extract the pre-scaled value from the calculation, for use
                                # in the gradient ascent algorithm
                                lin_val_alpha = self.causal_comparer_alpha.last_linear_value

                                # Beta
                                # Get the node embeddings
                                r_i_beta = self.nodes[excitor_i].causal_excitor_source_beta.value
                                r_j_beta = self.nodes[excitor_j].causal_excitor_dest_beta.value
                                e_k_beta = self.nodes[excitee_k].causal_excitee_source_beta.value
                                e_l_beta = self.nodes[excitee_l].causal_excitee_dest_beta.value

                                # Get the edge embeddings
                                excitor_edge_embedding_beta = \
                                    self.edge_embedder.embed_edge(r_i_beta, r_j_beta)
                                excitee_edge_embedding_beta = \
                                    self.edge_embedder.embed_edge(e_k_beta, e_l_beta)

                                # Compare the embeddings
                                weibull_beta = \
                                    self.causal_comparer_beta.compare_embeddings(
                                        excitor_edge_embedding_beta, excitee_edge_embedding_beta
                                    )

                                # Extract the pre-scaled value from the calculation, for use
                                # in the gradient ascent algorithm
                                lin_val_beta = self.causal_comparer_beta.last_linear_value

                            # Store the potential pairs, and the requisite parameters
                            self.possible_excitees[
                                (excitor_i, excitor_j)
                                ][
                                    (excitee_k, excitee_l)
                                    ] = \
                                        ((weibull_weight, weibull_alpha, weibull_beta),
                                         (lin_val_weight, lin_val_alpha, lin_val_beta))

    def increment_time(self):
        """Advance to the next day
        """

        # Move on the day counter
        self.time += 1

        # Increment time for the excitees
        for excites in self.current_excitees.values():
            for excite in excites:
                excite.increment_time()

        # Remove any dead excitees (where the remaining probability
        # of an excitement is below the threshold)
        self.current_excitees = {
            excitee: [
                excite
                for excite in excites
                if excite.alive
            ]
            for excitee, excites in self.current_excitees.items()
        }

        # Remove any empty excitees (i.e. any pairs of edges where
        # there are no more alive 'excites')
        self.current_excitees = {
            excitee: excites
            for excitee, excites in self.current_excitees.items()
            if len(excites) > 0
        }

        # Reset the new edges counter
        self.new_edges = dict()

        # Increment the time for the nodes
        for node in self.nodes:
            node.increment_time()

    def add_edge(self, i, j, edge_weight):
        """Add a new edge to the graph

        Args:
            i (int): The index of the source node
            j (int): The index of the destination node
            edge_weight (float): The monetary value of the new edge
        """
        # Save the edge
        self.edge_log.append(
            (i, j, self.time, edge_weight))

        # Add the new edge to today's counter
        if (i, j) in self.new_edges:
            self.new_edges[(i, j)] += 1
        else:
            self.new_edges[(i, j)] = 1

        # Update balances
        self.nodes[i].update_balance(-edge_weight)
        self.nodes[j].update_balance(edge_weight)

        # Record any edges that are excited by this edge occuring
        for excitee_nodes, excitee_parameters in self.possible_excitees[(i,j)].items():
            # Get the key parameters for this pair of edges
            weibull_params, lin_val_params = excitee_parameters
            weibull_weight, weibull_alpha, weibull_beta = weibull_params
            lin_val_weight, lin_val_alpha, lin_val_beta = lin_val_params

            # Add a new slot for this pair of edges, if required
            if excitee_nodes not in self.current_excitees:
                self.current_excitees[excitee_nodes] = []

            # Add the excitement to the list of excitements for
            # this pair of edges
            self.current_excitees[excitee_nodes].append(
                Excitement(
                    weibull_weight=weibull_weight,
                    weibull_alpha=weibull_alpha,
                    weibull_beta=weibull_beta,
                    lin_val_weight=lin_val_weight,
                    lin_val_alpha=lin_val_alpha,
                    lin_val_beta=lin_val_beta,
                    excitor_nodes=(i,j),
                    excitee_nodes=excitee_nodes,
                    alive_threshold=self.excitement_threshold
                )
            )

    def edge_baseline(self, i, j, min_intensity=0.000001):
        """The baseline intensity for edges i->j based
        on the nodes' relationship and the respective
        cumulative balances.

        Args:
            i (int): The index of the source node
            j (int): The index of the destination node
            min_intensity (str, optional): Prevents the
                intensity being zero. Default 0.000001

        Returns:
            float: The baseline intensity
        """
        # Get the balances at the start of the day
        balance_i = self.nodes[i].balance
        balance_j = self.nodes[j].balance

        # Add to debug log
        if self.debug_mode:
            self.add_to_debug_log(
                f'Balance of node {i} (for edge {i},{j})',
                balance_i,
                edges=[(i,j)])
            if i != j:
                self.add_to_debug_log(
                    f'Balance of node {j} (for edge {i},{j})',
                    balance_i,
                    edges=[(i,j)])

        # Store the balances for the gradient algorithm
        self.gradient_log['source_balance'] = balance_i
        self.gradient_log['dest_balance'] = balance_j

        # Calculate the linear output
        if self.mode == 'matrix':
            # Get the embeddings
            s_i = self.nodes[i].spontaneous_source.value
            s_j = self.nodes[j].spontaneous_dest.value

            linear_output_0 = \
                self.base_param_0.compare_embeddings(
                    s_i, s_j
                )
            linear_output_1 = \
                self.base_param_1.compare_embeddings(
                    s_i, s_j
                )
            linear_output_2 = \
                self.base_param_2.compare_embeddings(
                    s_i, s_j
                )
        elif self.mode == 'dot':
            # Get the embeddings
            s_0_i = self.nodes[i].spontaneous_source_0.value
            s_0_j = self.nodes[j].spontaneous_dest_0.value
            s_1_i = self.nodes[i].spontaneous_source_1.value
            s_1_j = self.nodes[j].spontaneous_dest_1.value
            s_2_i = self.nodes[i].spontaneous_source_2.value
            s_2_j = self.nodes[j].spontaneous_dest_2.value

            linear_output_0 = \
                self.spontaneous_comparer.compare_embeddings(
                    s_0_i, s_0_j
                )
            linear_output_1 = \
                self.spontaneous_comparer.compare_embeddings(
                    s_1_i, s_1_j
                )
            linear_output_2 = \
                self.spontaneous_comparer.compare_embeddings(
                    s_2_i, s_2_j
                )

        full_linear_output = \
            linear_output_0 + \
            balance_i * linear_output_1 + \
            balance_j * linear_output_2

        # Add to debug log
        if self.debug_mode:
            self.add_to_debug_log(
                f'Linear output 0 (for edge {i},{j})',
                linear_output_0,
                edges=[(i, j)])
            self.add_to_debug_log(
                f'Linear output 1 (for edge {i},{j})',
                linear_output_1,
                edges=[(i, j)])
            self.add_to_debug_log(
                f'Linear output 2 (for edge {i},{j})',
                linear_output_2,
                edges=[(i, j)])
            self.add_to_debug_log(
                f'Full linear output (for edge {i},{j})',
                full_linear_output,
                edges=[(i, j)])

        # Store the linear output for the gradient algorithm
        self.gradient_log['baseline_linear_value'] = \
            full_linear_output

        # Make it positive
        return log_exp_function(full_linear_output) + min_intensity

    def edge_intensity(self, i, j, spontaneous_on=True):
        """Get the total intensity for a particular edge

        Args:
            i (int): Index of the source node
            j (int): Index of the destination node
            spontaneous_on (bool, optional): Whether the spontaneous part
                of the model is enabled. Recommended to have off for the
                early part of the model learning, to allow the causal part
                of the model to dominate.
                Defaults to True.

        Returns:
            float: The total intensity of the edge
        """

        # Record whether the spontaneous part of the model
        # is active
        self.gradient_log['spontaneous_on'] = spontaneous_on

        # Start with the baseline intensity of the edge
        if spontaneous_on:
            intensity = self.edge_baseline(i, j)
        else:
            # If the spontaneous part of the model is disabled,
            # set a low constant probability across all edges
            intensity = 0.00001

        # Add to debug log
        if self.debug_mode:
            self.add_to_debug_log(
                f'Baseline intensity of edge {i}, {j}',
                intensity,
                edges=[(i, j)])

        # For each 'excite' that exists for this edge,
        # increase the intensity accordingly
        nodes = (i, j)
        if nodes in self.current_excitees:
            # Increase the intensity by the weighted
            # probability mass function of the generated,
            # discrete Weibull distribution
            intensity += sum(
                excite.probability
                for excite in self.current_excitees[nodes]
                if not excite.dormant
                )

            # Add to debug log
            if self.debug_mode and self.debug_monitor_excitement:
                self.add_to_debug_log(
                    f'Excitors of edge {i}, {j}',
                    [excite.excitor_nodes
                    for excite in self.current_excitees[nodes]
                    if not excite.dormant],
                    edges=[(i,j)]
                )

            # Record all the necessary values for the gradient
            # ascent algorithm

            # Record the Weibull parameters
            self.gradient_log['alphas'] = \
                [excite.weibull_alpha
                 for excite in self.current_excitees[nodes]
                 if not excite.dormant]
            self.gradient_log['betas'] = \
                [excite.weibull_beta
                 for excite in self.current_excitees[nodes]
                 if not excite.dormant]
            # Record the weighting of the Weibull distribution
            self.gradient_log['weights'] = \
                [excite.weibull_weight
                 for excite in self.current_excitees[nodes]
                 if not excite.dormant]

            # Record the linear value from the parameter
            # calculations (before passed through the smooth,
            # continuous function mapping R->R+)
            self.gradient_log['lin_alphas'] = \
                [excite.lin_val_alpha
                 for excite in self.current_excitees[nodes]
                 if not excite.dormant]
            self.gradient_log['lin_betas'] = \
                [excite.lin_val_beta
                 for excite in self.current_excitees[nodes]
                 if not excite.dormant]
            self.gradient_log['lin_weights'] = \
                [excite.lin_val_weight
                 for excite in self.current_excitees[nodes]
                 if not excite.dormant]

            # Record the original time that the excitor edge
            # occurred
            self.gradient_log['times'] = \
                [excite.time
                 for excite in self.current_excitees[nodes]
                 if not excite.dormant]

            # Record the nodes in the excitor edge
            self.gradient_log['excitor_nodes'] = \
                [excite.excitor_nodes
                 for excite in self.current_excitees[nodes]
                 if not excite.dormant]
        else:
            # Record empty lists if there are no 'excites'
            self.gradient_log['alphas'] = []
            self.gradient_log['betas'] = []
            self.gradient_log['weights'] = []
            self.gradient_log['times'] = []
            self.gradient_log['excitor_nodes'] = []

        # Record the total intensity for the gradient ascent
        # algorithm
        self.gradient_log['sum_Intensity'] = intensity

        if self.debug_mode:
            self.add_to_debug_log(
                f'Total intensity for edge {i}, {j}',
                intensity,
                edges=[(i,j)]
            )

        return intensity

    def edge_probability(self, i, j, count,
                         spontaneous_on=True):
        """The probability that a particular edge occured a
        given number of times on this day.

        Args:
            i (int): Index of the source node
            j (int): Index of the destination node
            count (int): The number of times that this edge
                occurred on this day
            spontaneous_on (bool, optional): Whether the spontaneous part
                of the model is enabled. Recommended to have off for the
                early part of the model learning, to allow the causal part
                of the model to dominate.
                Defaults to True.

        Returns:
            float: A probability
        """

        # Record the parameters for gradient ascent
        self.gradient_log['k'] = i
        self.gradient_log['l'] = j
        self.gradient_log['count'] = count

        # Calculate the probability using a Poisson
        # distribution with the intensity as the mean
        probability = poisson.pmf(
            k=count,
            mu=self.edge_intensity(
                i, j, spontaneous_on=spontaneous_on)
        )

        # Add to debug log
        if self.debug_mode:
            self.add_to_debug_log(
                f'Probability of edge {i}, {j}',
                probability,
                edges=[(i,j)])

        # Record the probability for gradient ascent
        self.gradient_log['P'] = probability

        # Calculate partial derivatives to create
        # pending parameter updates (to be applied when
        # the entire period has been added to the graph)
        self.calculate_derivatives()

        return probability

    def day_log_probability(self, spontaneous_on=True):
        """The log probability of all the edges that have
        occurred on this day

        Args:
            spontaneous_on (bool, optional): Whether the spontaneous part
                of the model is enabled. Recommended to have off for the
                early part of the model learning, to allow the causal part
                of the model to dominate.
                Defaults to True.

        Returns:
            float: The log probability
        """

        # Iterate through all possible edges
        total_log_probability = 0
        for i in range(self.count_nodes):
            for j in range(self.count_nodes):
                # Count how many times (if any) that
                # edge has occurred
                if (i, j) in self.new_edges:
                    count = self.new_edges[(i, j)]
                else:
                    count = 0

                # Add to debug log
                if self.debug_mode:
                    self.add_to_debug_log(
                        f'Count of edge {i}, {j}',
                        count,
                        edges=[(i,j)])

                # Take the log of the probability of
                # that edge occurring that number of
                # times, and add to the running total
                log_probability = log(
                    max(
                        0.00001,
                        self.edge_probability(
                            i, j, count,
                            spontaneous_on=spontaneous_on
                        )
                    )
                )
                total_log_probability += log_probability

                # Add to debug log
                if self.debug_mode:
                    self.add_to_debug_log(
                        f'Log probability of edge {i}, {j}',
                        log_probability,
                        edges=[(i,j)])

        # Add to debug log
        if self.debug_mode:
            self.add_to_debug_log(
                'Total log probability',
                total_log_probability,
                edges=[])

        return total_log_probability

    def calculate_derivatives(self):
        """Calculate an element of the derivatives of the log likelihood
        of the edges occuring in the frequencies that they did on each
        day in the period (with the current values of all the parameters),
        with respect to each parameter
        """
        return self.calculate_derivatives_mode()

    def calculate_derivatives_matrix(self):
        """The calculate_derivatives function for the 'matrix' mode
        """

        # Calculate the inverse of the probability
        inverse_probability = 1 / max(0.00001, self.gradient_log['P'])

        # Calculate partial derivatives that are
        # independent of the excitor edge.
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

        # Get the indices of the nodes in the excitee edge
        k = self.gradient_log['k']
        l = self.gradient_log['l']

        # Get the node embeddings
        node_k = self.nodes[k]
        e_k = node_k.causal_source.value
        node_l = self.nodes[l]
        e_l = node_l.causal_dest.value

        # Get the edge embedding
        excitee_kl = self.edge_embedder.embed_edge(e_k, e_l)

        if self.gradient_log['spontaneous_on']:
            # Calculate the spontaneous intensity gradient updates
            # - Calculate the partial derivative
            baseline_linear_value = \
                self.gradient_log['baseline_linear_value']
            delBaselineIntensity_delZero = \
                calc_delBaselineIntensity_delZero(
                    baseline_linear_value
                )

            source_balance = \
                self.gradient_log['source_balance']
            delBaselineIntensity_delOne = \
                calc_delBaselineIntensity_delOne(
                    baseline_linear_value,
                    source_balance
                )

            dest_balance = \
                self.gradient_log['dest_balance']
            delBaselineIntensity_delTwo = \
                calc_delBaselineIntensity_delTwo(
                    baseline_linear_value,
                    dest_balance
                )

            s_k = node_k.spontaneous_source.value
            s_l = node_l.spontaneous_dest.value
            delZero_delK = \
                calc_delBaselineComparer_delK(
                    self.base_param_0.matrix,
                    s_l
                )
            delZero_delL = \
                calc_delBaselineComparer_delL(
                    self.base_param_0.matrix,
                    s_k
                )
            delOne_delK = \
                calc_delBaselineComparer_delK(
                    self.base_param_1.matrix,
                    s_l
                )
            delOne_delL = \
                calc_delBaselineComparer_delL(
                    self.base_param_1.matrix,
                    s_k
                )
            delTwo_delK = \
                calc_delBaselineComparer_delK(
                    self.base_param_2.matrix,
                    s_l
                )
            delTwo_delL = \
                calc_delBaselineComparer_delL(
                    self.base_param_2.matrix,
                    s_k
                )
            delBaselineComparer_delMatrix = \
                calc_delBaselineComparer_delMatrix(
                    s_k, s_l
                )

            # - Apply the updates
            node_k.spontaneous_source.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delZero *
                    delZero_delK +
                    delBaselineIntensity_delOne *
                    delOne_delK +
                    delBaselineIntensity_delTwo *
                    delTwo_delK
                )
            )

            node_l.spontaneous_dest.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delZero *
                    delZero_delL +
                    delBaselineIntensity_delOne *
                    delOne_delL +
                    delBaselineIntensity_delTwo *
                    delTwo_delL
                )
            )

            self.base_param_0.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delZero *
                    delBaselineComparer_delMatrix
                )
            )

            self.base_param_1.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delOne *
                    delBaselineComparer_delMatrix
                )
            )

            self.base_param_2.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delTwo *
                    delBaselineComparer_delMatrix
                )
            )

        # Add to debug log
        if self.debug_mode:
            self.add_to_debug_log(
                f'Inverse probability (edge {k}, {l})',
                inverse_probability,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delP_delIntensity (edge {k}, {l})',
                delP_delIntensity,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delIntensity_delAlpha (edge {k}, {l})',
                delIntensity_delAlpha,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delIntensity_delBeta (edge {k}, {l})',
                delIntensity_delBeta,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delIntensity_delWeight (edge {k}, {l})',
                delIntensity_delWeight,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'baseline_linear_value (edge {k}, {l})',
                baseline_linear_value,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delBaselineIntensity_delZero (edge {k}, {l})',
                delBaselineIntensity_delZero,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'source_balance (edge {k}, {l})',
                source_balance,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delBaselineIntensity_delOne (edge {k}, {l})',
                delBaselineIntensity_delOne,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'dest_balance (edge {k}, {l})',
                dest_balance,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delBaselineIntensity_delTwo (edge {k}, {l})',
                delBaselineIntensity_delTwo,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delZero_delK (edge {k}, {l})',
                delZero_delK,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delZero_delL (edge {k}, {l})',
                delZero_delL,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delOne_delK (edge {k}, {l})',
                delOne_delK,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delOne_delL (edge {k}, {l})',
                delOne_delL,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delTwo_delK (edge {k}, {l})',
                delZero_delL,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delTwo_delL (edge {k}, {l})',
                delOne_delK,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delBaselineComparer_delMatrix (edge {k}, {l})',
                delBaselineComparer_delMatrix,
                edges=[(k,l)])

        for excite_index, (i, j) in enumerate(self.gradient_log['excitor_nodes']):
            # Get linear values from the calculations of the
            # Comparer objects
            lin_val_alpha = self.gradient_log['lin_alphas'][excite_index]
            lin_val_beta = self.gradient_log['lin_betas'][excite_index]
            lin_val_weight = self.gradient_log['lin_weights'][excite_index]

            # Get the node embeddings for the nodes in
            # the excitor edge
            node_i = self.nodes[i]
            r_i = node_i.causal_source.value
            node_j = self.nodes[j]
            r_j = node_j.causal_dest.value

            # Get the edge embedding
            excitor_ij = self.edge_embedder.embed_edge(r_i, r_j)

            # Calculate the partial derivates that depend
            # on the excitor edge
            delAlpha_delI = \
                calc_delComparer_delI(
                    linear_value=lin_val_alpha,
                    matrix=self.weibull_alpha_generator.matrix,
                    e_kl=excitee_kl,
                    node_dimension=self.node_dimension
                )
            delBeta_delI = \
                calc_delComparer_delI(
                    linear_value=lin_val_beta,
                    matrix=self.weibull_beta_generator.matrix,
                    e_kl=excitee_kl,
                    node_dimension=self.node_dimension
                )
            delWeight_delI = \
                calc_delComparer_delI(
                    linear_value=lin_val_weight,
                    matrix=self.weibull_weight_generator.matrix,
                    e_kl=excitee_kl,
                    node_dimension=self.node_dimension
                )

            delAlpha_delJ = \
                calc_delComparer_delJ(
                    linear_value=lin_val_alpha,
                    matrix=self.weibull_alpha_generator.matrix,
                    e_kl=excitee_kl,
                    node_dimension=self.node_dimension
                )
            delBeta_delJ = \
                calc_delComparer_delJ(
                    linear_value=lin_val_beta,
                    matrix=self.weibull_beta_generator.matrix,
                    e_kl=excitee_kl,
                    node_dimension=self.node_dimension
                )
            delWeight_delJ = \
                calc_delComparer_delJ(
                    linear_value=lin_val_weight,
                    matrix=self.weibull_weight_generator.matrix,
                    e_kl=excitee_kl,
                    node_dimension=self.node_dimension
                )

            delAlpha_delK = \
                calc_delComparer_delK(
                    linear_value=lin_val_alpha,
                    matrix=self.weibull_alpha_generator.matrix,
                    e_ij=excitor_ij,
                    node_dimension=self.node_dimension
                )
            delBeta_delK = \
                calc_delComparer_delK(
                    linear_value=lin_val_beta,
                    matrix=self.weibull_beta_generator.matrix,
                    e_ij=excitor_ij,
                    node_dimension=self.node_dimension
                )
            delWeight_delK = \
                calc_delComparer_delK(
                    linear_value=lin_val_weight,
                    matrix=self.weibull_weight_generator.matrix,
                    e_ij=excitor_ij,
                    node_dimension=self.node_dimension
                )

            delAlpha_delL = \
                calc_delComparer_delL(
                    linear_value=lin_val_alpha,
                    matrix=self.weibull_alpha_generator.matrix,
                    e_ij=excitor_ij,
                    node_dimension=self.node_dimension
                )
            delBeta_delL = \
                calc_delComparer_delL(
                    linear_value=lin_val_beta,
                    matrix=self.weibull_beta_generator.matrix,
                    e_ij=excitor_ij,
                    node_dimension=self.node_dimension
                )
            delWeight_delL = \
                calc_delComparer_delL(
                    linear_value=lin_val_weight,
                    matrix=self.weibull_weight_generator.matrix,
                    e_ij=excitor_ij,
                    node_dimension=self.node_dimension
                )

            delAlphaComparerdelMatrix = \
                calc_delComparer_delMatrix(
                    linear_value=lin_val_alpha,
                    e_ij=excitor_ij,
                    e_kl=excitee_kl,
                )

            delBetaComparerdelMatrix = \
                calc_delComparer_delMatrix(
                    linear_value=lin_val_beta,
                    e_ij=excitor_ij,
                    e_kl=excitee_kl,
                )

            delWeightComparerdelMatrix = \
                calc_delComparer_delMatrix(
                    linear_value=lin_val_weight,
                    e_ij=excitor_ij,
                    e_kl=excitee_kl,
                )

            # Apply the gradient updates
            # Node i
            node_i.causal_source.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delI +
                    delIntensity_delBeta[excite_index] *
                    delBeta_delI +
                    delIntensity_delWeight[excite_index] *
                    delWeight_delI
                )
            )

            # Node j
            node_j.causal_dest.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delJ +
                    delIntensity_delBeta[excite_index] *
                    delBeta_delJ +
                    delIntensity_delWeight[excite_index] *
                    delWeight_delJ
                )
            )

            # Node k
            node_k.causal_source.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delK +
                    delIntensity_delBeta[excite_index] *
                    delBeta_delK +
                    delIntensity_delWeight[excite_index] *
                    delWeight_delK
                )
            )

            # Node l
            node_l.causal_dest.add_gradient_update(
                inverse_probability * delP_delIntensity * (
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
                inverse_probability * delP_delIntensity *
                delIntensity_delWeight[excite_index] *
                delWeightComparerdelMatrix
            )

            # Alpha matrix
            self.weibull_alpha_generator.add_gradient_update(
                inverse_probability * delP_delIntensity *
                delIntensity_delAlpha[excite_index] *
                delAlphaComparerdelMatrix
            )

            # Beta matrix
            self.weibull_beta_generator.add_gradient_update(
                inverse_probability * delP_delIntensity *
                delIntensity_delBeta[excite_index] *
                delBetaComparerdelMatrix
            )

            # Add to debug log
            if self.debug_mode and self.debug_monitor_excitement:
                self.add_to_debug_log(
                    f'lin_val_alpha (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    lin_val_alpha,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'lin_val_beta (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    lin_val_beta,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'lin_val_weight (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    lin_val_weight,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delAlpha_delI (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delAlpha_delI,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delBeta_delI (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delBeta_delI,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delWeight_delI (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delWeight_delI,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delAlpha_delJ (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delAlpha_delJ,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delBeta_delJ (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delBeta_delJ,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delWeight_delJ (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delWeight_delJ,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delAlpha_delK (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delAlpha_delK,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delBeta_delK (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delBeta_delK,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delWeight_delK (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delWeight_delK,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delAlpha_delL (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delAlpha_delL,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delBeta_delL (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delBeta_delL,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delWeight_delL (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delWeight_delL,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delAlphaComparerdelMatrix (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delAlphaComparerdelMatrix,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delBetaComparerdelMatrix (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delBetaComparerdelMatrix,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delWeightComparerdelMatrix (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delWeightComparerdelMatrix,
                    edges=[(i,j), (k,l)])

        # Reset the calculations cache
        self.gradient_log = dict()

    def calculate_derivatives_dot(self):
        """The calculate_derivatives function for the 'dot' mode
        """

        # Calculate the inverse of the probability
        inverse_probability = 1 / max(0.00001, self.gradient_log['P'])

        # Calculate partial derivatives that are
        # independent of the excitor edge.
        delP_delIntensity = \
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

        # Get the indices of the nodes in the excitee edge
        k = self.gradient_log['k']
        l = self.gradient_log['l']

        # Get the nodes
        node_k = self.nodes[k]
        node_l = self.nodes[l]

        if self.gradient_log['spontaneous_on']:
            # Calculate the spontaneous intensity gradient updates
            # - Calculate the partial derivative
            baseline_linear_value = \
                self.gradient_log['baseline_linear_value']
            delBaselineIntensity_delZero = \
                calc_delBaselineIntensity_delZero(
                    baseline_linear_value
                )

            source_balance = \
                self.gradient_log['source_balance']
            delBaselineIntensity_delOne = \
                calc_delBaselineIntensity_delOne(
                    baseline_linear_value,
                    source_balance
                )

            dest_balance = \
                self.gradient_log['dest_balance']
            delBaselineIntensity_delTwo = \
                calc_delBaselineIntensity_delTwo(
                    baseline_linear_value,
                    dest_balance
                )

            s_k_0 = node_k.spontaneous_source_0.value
            s_l_0 = node_l.spontaneous_dest_0.value

            s_k_1 = node_k.spontaneous_source_1.value
            s_l_1 = node_l.spontaneous_dest_1.value

            s_k_2 = node_k.spontaneous_source_2.value
            s_l_2 = node_l.spontaneous_dest_2.value


            delZero_delK = \
                calc_delBaselineDotproduct_delParam(
                    s_l_0
                )
            delZero_delL = \
                calc_delBaselineDotproduct_delParam(
                    s_k_0
                )
            delOne_delK = \
                calc_delBaselineDotproduct_delParam(
                    s_l_1
                )
            delOne_delL = \
                calc_delBaselineDotproduct_delParam(
                    s_k_1
                )
            delTwo_delK = \
                calc_delBaselineDotproduct_delParam(
                    s_l_2
                )
            delTwo_delL = \
                calc_delBaselineDotproduct_delParam(
                    s_k_2
                )

            # - Apply the updates
            node_k.spontaneous_source_0.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delZero *
                    delZero_delK
                )
            )
            node_k.spontaneous_source_1.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delOne *
                    delOne_delK
                )
            )
            node_k.spontaneous_source_2.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delTwo *
                    delTwo_delK
                )
            )

            node_l.spontaneous_dest_0.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delZero *
                    delZero_delL
                )
            )
            node_l.spontaneous_dest_1.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delOne *
                    delOne_delL
                )
            )
            node_l.spontaneous_dest_2.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delTwo *
                    delTwo_delL
                )
            )

        # Get the causal node embeddings for the excitee
        e_k_alpha = node_k.causal_excitee_source_alpha.value
        e_l_alpha = node_l.causal_excitee_dest_alpha.value

        e_k_beta = node_k.causal_excitee_source_beta.value
        e_l_beta = node_l.causal_excitee_dest_beta.value

        e_k_weight = node_k.causal_excitee_source_weight.value
        e_l_weight = node_l.causal_excitee_dest_weight.value

        # Get the edge embeddings for the excitee
        excitee_kl_alpha = self.edge_embedder.embed_edge(e_k_alpha, e_l_alpha)
        excitee_kl_beta = self.edge_embedder.embed_edge(e_k_beta, e_l_beta)
        excitee_kl_weight = self.edge_embedder.embed_edge(e_k_weight, e_l_weight)

        # Add to debug log
        if self.debug_mode:
            self.add_to_debug_log(
                f'Inverse probability (edge {k}, {l})',
                inverse_probability,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delP_delIntensity (edge {k}, {l})',
                delP_delIntensity,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delIntensity_delAlpha (edge {k}, {l})',
                delIntensity_delAlpha,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delIntensity_delBeta (edge {k}, {l})',
                delIntensity_delBeta,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delIntensity_delWeight (edge {k}, {l})',
                delIntensity_delWeight,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'baseline_linear_value (edge {k}, {l})',
                baseline_linear_value,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delBaselineIntensity_delZero (edge {k}, {l})',
                delBaselineIntensity_delZero,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'source_balance (edge {k}, {l})',
                source_balance,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delBaselineIntensity_delOne (edge {k}, {l})',
                delBaselineIntensity_delOne,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'dest_balance (edge {k}, {l})',
                dest_balance,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delBaselineIntensity_delTwo (edge {k}, {l})',
                delBaselineIntensity_delTwo,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delZero_delK (edge {k}, {l})',
                delZero_delK,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delZero_delL (edge {k}, {l})',
                delZero_delL,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delOne_delK (edge {k}, {l})',
                delOne_delK,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delOne_delL (edge {k}, {l})',
                delOne_delL,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delTwo_delK (edge {k}, {l})',
                delZero_delL,
                edges=[(k,l)])
            self.add_to_debug_log(
                f'delTwo_delL (edge {k}, {l})',
                delOne_delK,
                edges=[(k,l)])

        for excite_index, (i, j) in enumerate(self.gradient_log['excitor_nodes']):
            # Get linear values from the calculations of the
            # Comparer objects
            lin_val_alpha = self.gradient_log['lin_alphas'][excite_index]
            lin_val_beta = self.gradient_log['lin_betas'][excite_index]
            lin_val_weight = self.gradient_log['lin_weights'][excite_index]

            # Get the nodes for the excitor edge
            node_i = self.nodes[i]
            node_j = self.nodes[j]

            # Get the causal node embeddings for the excitor
            r_i_alpha = node_i.causal_excitor_source_alpha.value
            r_j_alpha = node_j.causal_excitor_dest_alpha.value

            r_i_beta = node_i.causal_excitor_source_beta.value
            r_j_beta = node_j.causal_excitor_dest_beta.value

            r_i_weight = node_i.causal_excitor_source_weight.value
            r_j_weight = node_j.causal_excitor_dest_weight.value

            # Get the edge embeddings for the excitor
            excitor_ij_alpha = self.edge_embedder.embed_edge(r_i_alpha, r_j_alpha)
            excitor_ij_beta = self.edge_embedder.embed_edge(r_i_beta, r_j_beta)
            excitor_ij_weight = self.edge_embedder.embed_edge(r_i_weight, r_j_weight)

            # Calculate the partial derivates that depend
            # on the excitor edge
            delAlpha_delI = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_alpha,
                    node_embedding=r_j_alpha,
                    edge_embedding=excitee_kl_alpha
                )
            delBeta_delI = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_beta,
                    node_embedding=r_j_beta,
                    edge_embedding=excitee_kl_beta
                )
            delWeight_delI = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_weight,
                    node_embedding=r_j_weight,
                    edge_embedding=excitee_kl_weight
                )

            delAlpha_delJ = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_alpha,
                    node_embedding=r_i_alpha,
                    edge_embedding=excitee_kl_alpha
                )
            delBeta_delJ = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_beta,
                    node_embedding=r_i_beta,
                    edge_embedding=excitee_kl_beta
                )
            delWeight_delJ = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_weight,
                    node_embedding=r_i_weight,
                    edge_embedding=excitee_kl_weight
                )

            delAlpha_delK = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_alpha,
                    node_embedding=e_l_alpha,
                    edge_embedding=excitor_ij_alpha
                )
            delBeta_delK = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_beta,
                    node_embedding=e_l_beta,
                    edge_embedding=excitor_ij_beta
                )
            delWeight_delK = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_weight,
                    node_embedding=e_l_weight,
                    edge_embedding=excitor_ij_weight
                )

            delAlpha_delL = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_alpha,
                    node_embedding=e_k_alpha,
                    edge_embedding=excitor_ij_alpha
                )
            delBeta_delL = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_beta,
                    node_embedding=e_k_beta,
                    edge_embedding=excitor_ij_beta
                )
            delWeight_delL = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_weight,
                    node_embedding=e_k_weight,
                    edge_embedding=excitor_ij_weight
                )

            # Apply the gradient updates
            # Node i
            node_i.causal_excitor_source_alpha.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delI
                )
            )
            node_i.causal_excitor_source_beta.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delBeta[excite_index] *
                    delBeta_delI
                )
            )
            node_i.causal_excitor_source_weight.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delWeight[excite_index] *
                    delWeight_delI
                )
            )

            # Node j
            node_j.causal_excitor_dest_alpha.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delJ
                )
            )
            node_j.causal_excitor_dest_beta.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delBeta[excite_index] *
                    delBeta_delJ
                )
            )
            node_j.causal_excitor_dest_weight.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delWeight[excite_index] *
                    delWeight_delJ
                )
            )

            # Node k
            node_k.causal_excitee_source_alpha.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delK
                )
            )
            node_k.causal_excitee_source_beta.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delBeta[excite_index] *
                    delBeta_delK
                )
            )
            node_k.causal_excitee_source_weight.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delWeight[excite_index] *
                    delWeight_delK
                )
            )

            # Node l
            node_l.causal_excitee_dest_alpha.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delL
                )
            )
            node_l.causal_excitee_dest_beta.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delBeta[excite_index] *
                    delBeta_delL
                )
            )
            node_l.causal_excitee_dest_weight.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delWeight[excite_index] *
                    delWeight_delL
                )
            )

            # Add to debug log
            if self.debug_mode and self.debug_monitor_excitement:
                self.add_to_debug_log(
                    f'lin_val_alpha (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    lin_val_alpha,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'lin_val_beta (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    lin_val_beta,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'lin_val_weight (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    lin_val_weight,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delAlpha_delI (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delAlpha_delI,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delBeta_delI (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delBeta_delI,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delWeight_delI (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delWeight_delI,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delAlpha_delJ (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delAlpha_delJ,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delBeta_delJ (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delBeta_delJ,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delWeight_delJ (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delWeight_delJ,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delAlpha_delK (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delAlpha_delK,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delBeta_delK (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delBeta_delK,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delWeight_delK (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delWeight_delK,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delAlpha_delL (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delAlpha_delL,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delBeta_delL (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delBeta_delL,
                    edges=[(i,j), (k,l)])
                self.add_to_debug_log(
                    f'delWeight_delL (edge: {k}, {l} - excitee {excite_index}: {i}, {j})',
                    delWeight_delL,
                    edges=[(i,j), (k,l)])

        # Reset the calculations cache
        self.gradient_log = dict()

    def apply_gradient_updates(self):
        """Take the pending gradient updates and
        update the parameters accordingly
        """

        # Update node embeddings
        for node in self.nodes:
            node.apply_gradient_updates()

        # Update matrices in the Comparer objects
        if self.mode == 'matrix':
            self.weibull_alpha_generator.apply_gradient_updates()
            self.weibull_beta_generator.apply_gradient_updates()
            self.weibull_weight_generator.apply_gradient_updates()

    def reset(self, discard_gradient_updates=False,
              spontaneous_on=True):
        """Remove edges and excitation to prepare for another
        epoch of training (embeddings and matrice learnings
        are retained)

        Args:
            discard_gradient_updates (bool, optional): Whether to discard
                the gradient updates (True), or to apply them (False).
                Defaults to False.
            spontaneous_on (bool, optional): Whether the spontaneous part
                of the model is enabled. Recommended to have off for the
                early part of the model learning, to allow the causal part
                of the model to dominate.
                Defaults to True.
        """

        # Set the time back to the start of the period
        self.time = 0

        # If gradient updates are being applied, increase the
        # epoch number
        if not discard_gradient_updates:
            self.epoch += 1

        # Clear any recorded edges
        self.edge_log = []
        self.new_edges = dict()

        # Clear any excitation
        self.possible_excitees = dict()
        self.current_excitees = dict()

        # Apply or discard gradient updates and reset
        # the node balances
        # - Nodes
        for node in self.nodes:
            node.reset(
                discard_gradient_updates=discard_gradient_updates,
                spontaneous_on=spontaneous_on
            )

        # - Causal matrices
        if self.mode == 'matrix':
            self.weibull_alpha_generator.reset(
                    discard_gradient_updates=discard_gradient_updates
                )
            self.weibull_beta_generator.reset(
                    discard_gradient_updates=discard_gradient_updates
                )
            self.weibull_weight_generator.reset(
                    discard_gradient_updates=discard_gradient_updates
                )

            # - Spontaneous matrices
            if spontaneous_on:
                self.base_param_0.reset(
                        discard_gradient_updates=discard_gradient_updates
                    )
                self.base_param_1.reset(
                        discard_gradient_updates=discard_gradient_updates
                    )
                self.base_param_2.reset(
                        discard_gradient_updates=discard_gradient_updates
                    )
            else:
                self.base_param_0.reset(
                        discard_gradient_updates=True
                    )
                self.base_param_1.reset(
                        discard_gradient_updates=True
                    )
                self.base_param_2.reset(
                        discard_gradient_updates=True
                    )

        # Update pairs of edges which could
        # excite each other under the updated
        # model parameters
        self.find_excitors()
