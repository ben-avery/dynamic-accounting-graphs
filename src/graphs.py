"""Main module for the Dynamic Accounting Graph class
"""
from scipy.stats import poisson
from numpy import exp

from nodes_and_edges import Node, EdgeEmbedder, Comparer
from excitement import Excitation
from utilities import (
    calc_inverse_probability_delP_delIntensity,
    calc_delIntensity_delAlpha,
    calc_delIntensity_delBeta, calc_delIntensity_delWeight,
    calc_delBaselineIntensity_delZero,
    calc_delBaselineIntensity_delOne,
    calc_delBaselineIntensity_delTwo,
    calc_delCausalDotproduct_delParam,
    calc_delBaselineDotproduct_delParam,
    log_exp_function, find_log_exp_shift,
    find_lin_exp_shift
)


class DynamicAccountingGraph():
    """Class for a Dynamic Accounting Graph
    """
    def __init__(self, accounts, node_dimension,
                 average_balances, average_weight,
                 possible_edges,
                 causal_learning_rate=0.001,
                 causal_learning_boost=1,
                 alpha_regularisation_rate=10**(-7),
                 beta_regularisation_rate=10**(-7),
                 weight_regularisation_rate=10**(-6),
                 spontaneous_learning_rate=0.0001,
                 spontaneous_regularisation_rate=10**(-6)):
        """Initialise the class

        Args:
            accounts (list): A list of Account object (with attributes 
                for name, balance, number and mapping)
            node_dimension (int): The dimension for the node embeddings
            average_balances (list): A list of the daily average balances
                for each account
            average_weight (float): The daily average weight contributed
                by causal excitation to any potential excitee from the
                initialisation parameters of the causal parts of the model
            possible_edges (set): A set of tuple pairs, (i, j), for all the
                edges, i -> j, that occured in the period.
            causal_learning_rate (float, optional): The learning rate for the
                optimisation of causal parameters. Defaults to 0.001.
            causal_learning_boost (float, optional): Multiple to boost the causal
                learning rate by during the training of only the causal part
                of the model (i.e. when the spontaneous part of the model is
                deactivated). Defaults to 1 - i.e. no boost.
            alpha_regularisation_rate (float, optional): The weight towards the
                L2 regularisation penalty of Weibull alpha parameters. Defaults to 10**(-7).
            beta_regularisation_rate (float, optional): The weight towards the
                L2 regularisation penalty of Weibull beta parameters. Defaults to 10**(-7).
            weight_regularisation_rate  (float, optional): The weight towards the
                L2 regularisation penalty of Weibull weight parameters. Defaults to 10**(-6).
            spontaneous_learning_rate (float, optional): The learning rate for the
                optimisation of spontaneous parameters. Defaults to 0.0001.
            spontaneous_regularisation_rate (float, optional): The weight towards the
                L2 regularisation penalty of spontaneous parameters. Defaults to 10**(-6).
        """
        # Start at the beginning of the accounting period
        self.time = 0

        # Start training at the first epoch
        self.epoch = 0

        # Record parameter values
        self.causal_learning_rate = causal_learning_rate
        self.causal_learning_boost = causal_learning_boost

        self.alpha_regularisation_rate = alpha_regularisation_rate
        self.beta_regularisation_rate = beta_regularisation_rate
        self.weight_regularisation_rate = weight_regularisation_rate

        self.spontaneous_learning_rate = spontaneous_learning_rate
        self.spontaneous_regularisation_rate = spontaneous_regularisation_rate

        # Set the threshold below which excitation is seen as trivial
        self.excitement_threshold = 0.0001

        # Set the scales for the smooth, positive functions
        self.f_shift_spontaneous = find_log_exp_shift(self.excitement_threshold)
        self.g_shift_alpha = find_lin_exp_shift(13/2)
        self.g_shift_beta = find_lin_exp_shift(2)
        self.f_shift_weight = find_log_exp_shift(self.excitement_threshold)

        # Create the nodes with initialised embeddings
        self.nodes = []
        self.node_dimension = node_dimension
        for i, account in enumerate(accounts):
            node = Node(
                name=account.name,
                opening_balance=account.balance,
                dimension=node_dimension,
                average_balance=average_balances[i],
                average_weight=average_weight,
                causal_learning_rate=self.causal_learning_rate,
                causal_learning_boost=self.causal_learning_boost,
                alpha_regularisation_rate=self.alpha_regularisation_rate,
                beta_regularisation_rate=self.beta_regularisation_rate,
                weight_regularisation_rate=self.weight_regularisation_rate,
                spontaneous_learning_rate=self.spontaneous_learning_rate,
                spontaneous_regularisation_rate=self.spontaneous_regularisation_rate,
                meta_data={
                    'account_number': account.number,
                    'mapping': account.mapping
                }
            )

            self.nodes.append(node)
        self.count_nodes = len(self.nodes)

        # Store which edges are able to occur
        self.possible_edges = possible_edges

        # Create an edge embedder which turns two
        # node embeddings into an embedding for an
        # edge between the nodes
        self.edge_embedder = EdgeEmbedder()

        # Set up the dot product parameter generation
        # for comparing two edge embeddings
        self.causal_comparer_weight = Comparer(
            f_shift=self.f_shift_weight,
            positive_output=True
        )
        self.causal_comparer_alpha = Comparer(
            g_shift=self.g_shift_alpha,
            positive_output=True,
            min_at=0.5
        )
        self.causal_comparer_beta = Comparer(
            g_shift=self.g_shift_beta,
            positive_output=True,
            min_at=1.0
        )

        # Set up the dot product parameter generation
        # for comparing two node embeddings
        self.spontaneous_comparer = Comparer(
            positive_output=False
        )

        # Create attributes to record all edges and any
        # new edges for this day
        self.edge_log = []
        self.new_edges = dict()

        # Store any pairs of edges which will excite each
        # other based on the weight of the corresponding
        # Weibull distribution
        self.possible_excitees = dict()
        self.find_excitors()

        # Create an attribute to store any edges that are
        # currently excited by previous edges
        self.current_excitations = dict()

        # Store the linear parts of the spontaneous intensity
        # calculation
        self.spontaneous_linear_parts = dict()
        self.calculate_spontaneous_coefficients()

        # Create an attribute to store gradients and values
        # for the gradient algorithm
        self.gradient_log = dict()

    def find_excitors(self):
        """Find any pairs of edges which would excite each
        other based on the weight of the corresponding
        Weibull distribution
        """
        # Clear the current possible excitors
        self.possible_excitees = dict()

        for excitor_i in range(self.count_nodes):
            # Get the node embedding
            r_i = self.nodes[excitor_i].causal_excitor_source_weight.value

            for excitor_j in range(self.count_nodes):
                if excitor_i == excitor_j:
                    # Don't allow self-loops
                    continue
                if (excitor_i, excitor_j) not in self.possible_edges:
                    # No need to calculate values with excitors that
                    # didn't occur in the period
                    continue

                # Get the node embedding
                r_j = self.nodes[excitor_j].causal_excitor_dest_weight.value

                # Get the edge embedding
                excitor_edge_embedding = self.edge_embedder.embed_edge(r_i, r_j)

                # Create an entry for this excitor
                self.possible_excitees[(excitor_i, excitor_j)] = dict()

                for excitee_k in range(self.count_nodes):
                    # Get the node embedding
                    e_k = self.nodes[excitee_k].causal_excitee_source_weight.value

                    for excitee_l in range(self.count_nodes):
                        if excitee_k == excitee_l:
                            # Don't allow self-loops
                            continue

                        # Get the node embedding
                        e_l = self.nodes[excitee_l].causal_excitee_dest_weight.value

                        # Get the edge embedding
                        excitee_edge_embedding = \
                            self.edge_embedder.embed_edge(e_k, e_l)

                        # Get the Weibull weight (the total probability of the excitor
                        # edge exciting the excitee edge)
                        weibull_weight = \
                            self.causal_comparer_weight.compare_embeddings(
                                excitor_edge_embedding, excitee_edge_embedding
                            )

                        # Extract the pre-scaled value from the calculation, for use
                        # in the learning algorithm
                        lin_val_weight = self.causal_comparer_weight.last_linear_value

                        # If there is sufficient probability, add as a possible excitee
                        if weibull_weight > self.excitement_threshold:
                            # Calculate the Weibull parameters (characterising the average
                            # time to excitation, and the spread of likely times to
                            # excitation)

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
                            # in the gradient algorithm
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
                            # in the gradient algorithm
                            lin_val_beta = self.causal_comparer_beta.last_linear_value

                            # Store the potential pairs, and the requisite parameters
                            self.possible_excitees[
                                (excitor_i, excitor_j)
                                ][
                                    (excitee_k, excitee_l)
                                    ] = \
                                        ((weibull_weight, weibull_alpha, weibull_beta),
                                         (lin_val_weight, lin_val_alpha, lin_val_beta))

    def calculate_spontaneous_coefficients(self):
        """Cache the linear coefficients for the spontaneous
        intensity calculations for each edge
        """

        # Clear the cache
        self.spontaneous_linear_parts = dict()

        for i in range(self.count_nodes):
            # Get the embeddings for node i
            s_0_i = self.nodes[i].spontaneous_source_0.value
            s_1_i = self.nodes[i].spontaneous_source_1.value
            s_2_i = self.nodes[i].spontaneous_source_2.value

            for j in range(self.count_nodes):
                if i == j:
                    # Don't allow self-loops
                    continue

                # Get the embeddings for node j
                s_0_j = self.nodes[j].spontaneous_dest_0.value
                s_1_j = self.nodes[j].spontaneous_dest_1.value
                s_2_j = self.nodes[j].spontaneous_dest_2.value

                # Calculate the linear coefficients
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

                # Cache the result
                self.spontaneous_linear_parts[(i, j)] = \
                    (linear_output_0, linear_output_1, linear_output_2)

    def increment_time(self):
        """Advance to the next day
        """

        # Move on the day counter
        self.time += 1

        # Increment time for the active excitations
        for excitations in self.current_excitations.values():
            for excitation in excitations:
                excitation.increment_time()

        # Remove any inactive excitations (where the remaining probability
        # of an excitation is below the threshold)
        self.current_excitations = {
            excitee: [
                excitation
                for excitation in excitations
                if excitation.alive
            ]
            for excitee, excitations in self.current_excitations.items()
        }

        # Remove any empty excitees (i.e. any edges where
        # there are no more alive 'excitations')
        self.current_excitations = {
            excitee: excitations
            for excitee, excitations in self.current_excitations.items()
            if len(excitations) > 0
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

        # Record any edges that are excited by this edge occurring
        for excitee_nodes, excitee_parameters in self.possible_excitees[(i,j)].items():
            # Get the key parameters for this pair of edges
            weibull_params, lin_val_params = excitee_parameters
            weibull_weight, weibull_alpha, weibull_beta = weibull_params
            lin_val_weight, lin_val_alpha, lin_val_beta = lin_val_params

            # Add a new slot for this pair of edges, if required
            if excitee_nodes not in self.current_excitations:
                self.current_excitations[excitee_nodes] = []

            # Add the excitation to the list of excitations for
            # this pair of edges
            self.current_excitations[excitee_nodes].append(
                Excitation(
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

        # Store the balances for the gradient algorithm
        self.gradient_log['source_balance'] = balance_i
        self.gradient_log['dest_balance'] = balance_j

        # Get the linear output from the cache
        linear_output_0, linear_output_1, linear_output_2 = \
            self.spontaneous_linear_parts[(i, j)]

        full_linear_output = \
            linear_output_0 + \
            balance_i * linear_output_1 + \
            balance_j * linear_output_2

        # Store the linear output for the gradient algorithm
        self.gradient_log['baseline_linear_value'] = \
            full_linear_output

        # Make it positive
        return log_exp_function(
            full_linear_output, f_shift=self.f_shift_spontaneous
            ) + min_intensity

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

        # For each excitation that exists for this edge,
        # increase the intensity accordingly
        nodes = (i, j)
        if nodes in self.current_excitations:
            # Increase the intensity by the weighted
            # probability mass function of the generated,
            # discrete Weibull distribution
            intensity += sum(
                excitation.probability
                for excitation in self.current_excitations[nodes]
                if not excitation.dormant
                )

            # Record all the necessary values for the gradient algorithm

            # Record the Weibull parameters
            self.gradient_log['alphas'] = \
                [excitation.weibull_alpha
                 for excitation in self.current_excitations[nodes]
                 if not excitation.dormant]
            self.gradient_log['betas'] = \
                [excitation.weibull_beta
                 for excitation in self.current_excitations[nodes]
                 if not excitation.dormant]
            # Record the weighting of the Weibull distribution
            self.gradient_log['weights'] = \
                [excitation.weibull_weight
                 for excitation in self.current_excitations[nodes]
                 if not excitation.dormant]

            # Record the linear value from the parameter
            # calculations (before passed through the smooth,
            # continuous function mapping R->R+)
            self.gradient_log['lin_alphas'] = \
                [excitation.lin_val_alpha
                 for excitation in self.current_excitations[nodes]
                 if not excitation.dormant]
            self.gradient_log['lin_betas'] = \
                [excitation.lin_val_beta
                 for excitation in self.current_excitations[nodes]
                 if not excitation.dormant]
            self.gradient_log['lin_weights'] = \
                [excitation.lin_val_weight
                 for excitation in self.current_excitations[nodes]
                 if not excitation.dormant]

            # Record the original time that the excitor edge
            # occurred
            self.gradient_log['times'] = \
                [excitation.time
                 for excitation in self.current_excitations[nodes]
                 if not excitation.dormant]

            # Record the nodes in the excitor edge
            self.gradient_log['excitor_nodes'] = \
                [excitation.excitor_nodes
                 for excitation in self.current_excitations[nodes]
                 if not excitation.dormant]
        else:
            # Record empty lists if there are no active excitations
            # for this excitee
            self.gradient_log['alphas'] = []
            self.gradient_log['betas'] = []
            self.gradient_log['weights'] = []
            self.gradient_log['times'] = []
            self.gradient_log['excitor_nodes'] = []

        # Record the total intensity for the gradient algorithm
        self.gradient_log['sum_Intensity'] = intensity

        return intensity

    def edge_log_probability(self, i, j, count,
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

        # Record the parameters for the gradient algorithm
        self.gradient_log['k'] = i
        self.gradient_log['l'] = j
        self.gradient_log['count'] = count

        # Calculate the log probability using a Poisson
        # distribution with the intensity as the mean
        log_probability = poisson._logpmf(
            k=count,
            mu=self.edge_intensity(
                i, j, spontaneous_on=spontaneous_on
            )
        )

        # Get the probability, truncating at underflows
        if log_probability < -30:
            log_probability = -30
        probability = exp(log_probability)

        # Record the probability for the gradient algorithm
        self.gradient_log['P'] = probability

        # Calculate partial derivatives to create
        # pending parameter updates (to be applied when
        # the entire period has been added to the graph)
        self.calculate_derivatives()

        return log_probability

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
                if i == j:
                    # Don't allow self-loops
                    continue

                # Count how many times (if any) that
                # edge has occurred
                if (i, j) in self.new_edges:
                    count = self.new_edges[(i, j)]
                else:
                    count = 0

                # Take the log of the probability of
                # that edge occurring that number of
                # times, and add to the running total
                log_probability = \
                    self.edge_log_probability(
                        i, j, count,
                        spontaneous_on=spontaneous_on
                    )
                total_log_probability += log_probability

        return total_log_probability

    def calculate_derivatives(self):
        """Calculate an element of the derivatives of the log likelihood
        of the edges occuring in the frequencies that they did on each
        day in the period (with the current values of all the parameters),
        with respect to each parameter
        """

        # Calculate partial derivatives that are
        # independent of the excitor edge.
        delIntensity_delAlpha = [
            calc_delIntensity_delAlpha(
                time,
                self.gradient_log['alphas'][excitation_index],
                self.gradient_log['betas'][excitation_index],
                self.gradient_log['weights'][excitation_index]
            )
            for excitation_index, time in enumerate(self.gradient_log['times'])
        ]
        delIntensity_delBeta = [
            calc_delIntensity_delBeta(
                time,
                self.gradient_log['alphas'][excitation_index],
                self.gradient_log['betas'][excitation_index],
                self.gradient_log['weights'][excitation_index]
            )
            for excitation_index, time in enumerate(self.gradient_log['times'])
        ]
        delIntensity_delWeight = [
            calc_delIntensity_delWeight(
                time,
                self.gradient_log['alphas'][excitation_index],
                self.gradient_log['betas'][excitation_index]
            )
            for excitation_index, time in enumerate(self.gradient_log['times'])
        ]

        # Calculate the inverse probability multiplied by the partial derivative
        # of the probability by the total intensity (since these two terms only
        # ever appear multiplied together, and have a significant amount of
        # cancellation)
        inverse_probability_delP_delIntensity = \
            calc_inverse_probability_delP_delIntensity(
                self.gradient_log['count'],
                self.gradient_log['sum_Intensity'])

        # Get the indices of the nodes in the excitee edge
        k = self.gradient_log['k']
        l = self.gradient_log['l']

        # Get the excitee nodes
        node_k = self.nodes[k]
        node_l = self.nodes[l]

        # Only update the spontaneous parameters if they were enabled
        if self.gradient_log['spontaneous_on']:
            # Calculate the spontaneous intensity gradient updates
            # - Calculate the partial derivative
            baseline_linear_value = \
                self.gradient_log['baseline_linear_value']
            delBaselineIntensity_delZero = \
                calc_delBaselineIntensity_delZero(
                    baseline_linear_value,
                    f_shift=self.f_shift_spontaneous
                )

            source_balance = \
                self.gradient_log['source_balance']
            delBaselineIntensity_delOne = \
                calc_delBaselineIntensity_delOne(
                    baseline_linear_value,
                    source_balance,
                    f_shift=self.f_shift_spontaneous
                )

            dest_balance = \
                self.gradient_log['dest_balance']
            delBaselineIntensity_delTwo = \
                calc_delBaselineIntensity_delTwo(
                    baseline_linear_value,
                    dest_balance,
                    f_shift=self.f_shift_spontaneous
                )

            # - - Get the embeddings
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
                inverse_probability_delP_delIntensity * (
                    delBaselineIntensity_delZero *
                    delZero_delK
                )
            )
            node_k.spontaneous_source_1.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delBaselineIntensity_delOne *
                    delOne_delK
                )
            )
            node_k.spontaneous_source_2.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delBaselineIntensity_delTwo *
                    delTwo_delK
                )
            )

            node_l.spontaneous_dest_0.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delBaselineIntensity_delZero *
                    delZero_delL
                )
            )
            node_l.spontaneous_dest_1.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delBaselineIntensity_delOne *
                    delOne_delL
                )
            )
            node_l.spontaneous_dest_2.add_gradient_update(
                inverse_probability_delP_delIntensity * (
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

        for excitation_index, (i, j) in enumerate(self.gradient_log['excitor_nodes']):
            # Get linear values from the calculations of the
            # Comparer objects
            lin_val_alpha = self.gradient_log['lin_alphas'][excitation_index]
            lin_val_beta = self.gradient_log['lin_betas'][excitation_index]
            lin_val_weight = self.gradient_log['lin_weights'][excitation_index]

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
                    edge_embedding=excitee_kl_alpha,
                    g_shift=self.g_shift_alpha
                )
            delBeta_delI = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_beta,
                    node_embedding=r_j_beta,
                    edge_embedding=excitee_kl_beta,
                    g_shift=self.g_shift_beta
                )
            delWeight_delI = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_weight,
                    node_embedding=r_j_weight,
                    edge_embedding=excitee_kl_weight,
                    f_shift=self.f_shift_weight
                )

            delAlpha_delJ = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_alpha,
                    node_embedding=r_i_alpha,
                    edge_embedding=excitee_kl_alpha,
                    g_shift=self.g_shift_alpha
                )
            delBeta_delJ = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_beta,
                    node_embedding=r_i_beta,
                    edge_embedding=excitee_kl_beta,
                    g_shift=self.g_shift_beta
                )
            delWeight_delJ = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_weight,
                    node_embedding=r_i_weight,
                    edge_embedding=excitee_kl_weight,
                    f_shift=self.f_shift_weight
                )

            delAlpha_delK = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_alpha,
                    node_embedding=e_l_alpha,
                    edge_embedding=excitor_ij_alpha,
                    g_shift=self.g_shift_alpha
                )
            delBeta_delK = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_beta,
                    node_embedding=e_l_beta,
                    edge_embedding=excitor_ij_beta,
                    g_shift=self.g_shift_beta
                )
            delWeight_delK = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_weight,
                    node_embedding=e_l_weight,
                    edge_embedding=excitor_ij_weight,
                    f_shift=self.f_shift_weight
                )

            delAlpha_delL = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_alpha,
                    node_embedding=e_k_alpha,
                    edge_embedding=excitor_ij_alpha,
                    g_shift=self.g_shift_alpha
                )
            delBeta_delL = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_beta,
                    node_embedding=e_k_beta,
                    edge_embedding=excitor_ij_beta,
                    g_shift=self.g_shift_beta
                )
            delWeight_delL = \
                calc_delCausalDotproduct_delParam(
                    linear_value=lin_val_weight,
                    node_embedding=e_k_weight,
                    edge_embedding=excitor_ij_weight,
                    f_shift=self.f_shift_weight
                )

            # Apply the gradient updates
            # Node i
            node_i.causal_excitor_source_alpha.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delAlpha[excitation_index] *
                    delAlpha_delI
                )
            )
            node_i.causal_excitor_source_beta.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delBeta[excitation_index] *
                    delBeta_delI
                )
            )
            node_i.causal_excitor_source_weight.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delWeight[excitation_index] *
                    delWeight_delI
                )
            )

            # Node j
            node_j.causal_excitor_dest_alpha.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delAlpha[excitation_index] *
                    delAlpha_delJ
                )
            )
            node_j.causal_excitor_dest_beta.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delBeta[excitation_index] *
                    delBeta_delJ
                )
            )
            node_j.causal_excitor_dest_weight.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delWeight[excitation_index] *
                    delWeight_delJ
                )
            )

            # Node k
            node_k.causal_excitee_source_alpha.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delAlpha[excitation_index] *
                    delAlpha_delK
                )
            )
            node_k.causal_excitee_source_beta.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delBeta[excitation_index] *
                    delBeta_delK
                )
            )
            node_k.causal_excitee_source_weight.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delWeight[excitation_index] *
                    delWeight_delK
                )
            )

            # Node l
            node_l.causal_excitee_dest_alpha.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delAlpha[excitation_index] *
                    delAlpha_delL
                )
            )
            node_l.causal_excitee_dest_beta.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delBeta[excitation_index] *
                    delBeta_delL
                )
            )
            node_l.causal_excitee_dest_weight.add_gradient_update(
                inverse_probability_delP_delIntensity * (
                    delIntensity_delWeight[excitation_index] *
                    delWeight_delL
                )
            )

        # Reset the calculations cache
        self.gradient_log = dict()

    def apply_gradient_updates(self):
        """Take the pending gradient updates and
        update the parameters accordingly
        """

        # Update node embeddings
        for node in self.nodes:
            node.apply_gradient_updates()

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

        # Clear any excitations
        self.current_excitations = dict()

        # Reset which pairs of edges could excite
        # each other
        self.possible_excitees = dict()

        # Apply or discard gradient updates and reset
        # the node balances
        for node in self.nodes:
            node.reset(
                discard_gradient_updates=discard_gradient_updates,
                spontaneous_on=spontaneous_on
            )

        # Update pairs of edges which could
        # excite each other under the updated
        # model parameters
        self.find_excitors()

        # Update the cache for the linear parts of the
        # spontaneous intensity calculations
        self.calculate_spontaneous_coefficients()
