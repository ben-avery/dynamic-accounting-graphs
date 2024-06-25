"""Main module for the Dynamic Accounting Graph class
"""
from scipy.stats import poisson
from numpy import log, exp

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
    log_exp_function
)


class DynamicAccountingGraph():
    """Class for a Dynamic Accounting Graph
    """
    def __init__(self, accounts, node_dimension,
                 edge_embedder_mode='concatenate',
                 weibull_weight_generator_mode='matrix',
                 weibull_alpha_generator_mode='matrix',
                 weibull_beta_generator_mode='matrix',
                 learning_rate=0.0001):
        """Initialise the class

        Args:
            accounts (list): A list of Account object (with attributes 
                for name, balance, number and mapping)
            node_dimension (int): The dimension for the node embeddings
            edge_embedder_mode (str, optional): The method of combining 
                node embeddings to get edge embeddings. Defaults to
                'concatenate'.
            weibull_weight_generator_mode (str, optional): The method of
                combining two edge embeddings to get a real-valued
                parameter for the weight of a discrete Weibull distribution.
                Defaults to 'matrix'.
            weibull_alpha_generator_mode (str, optional): The method of
                combining two edge embeddings to get a real-valued
                parameter for the alpha parameter of a discrete Weibull
                distribution (which controls the likely time between
                two edges exciting each other). Defaults to 'matrix'.
            weibull_beta_generator_mode (str, optional): The method of
                combining two edge embeddings to get a real-valued
                parameter for the beta parameter of a discrete Weibull
                distribution (which controls the spread of likely times
                between two edges exciting each other). Defaults
                to 'matrix'.
            learning_rate (float, optional): The learning rate for the
                gradient ascent algorithm. Defaults to 0.0001.
        """
        self.time = 0

        self.learning_rate = learning_rate

        # Create the nodes with random embeddings
        self.nodes = []
        self.node_dimension = node_dimension
        for account in accounts:
            node = Node(
                name=account.name,
                opening_balance=account.balance,
                dimension=node_dimension,
                learning_rate=self.learning_rate,
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
            mode=edge_embedder_mode
        )

        # Create generators for the Weibull distribution
        # parameters
        self.weibull_weight_generator = EdgeComparer(
            dimension=self.edge_embedder.output_dimension,
            learning_rate=self.learning_rate,
            mode=weibull_weight_generator_mode
        )
        self.weibull_alpha_generator = EdgeComparer(
            dimension=self.edge_embedder.output_dimension,
            learning_rate=self.learning_rate,
            mode=weibull_alpha_generator_mode
        )
        self.weibull_beta_generator = EdgeComparer(
            dimension=self.edge_embedder.output_dimension,
            learning_rate=self.learning_rate,
            mode=weibull_beta_generator_mode,
            min_at=1
        )

        # Create generators for linear parameters of
        # baseline intensity
        self.base_param_0 = EdgeComparer(
            dimension=self.node_dimension,
            learning_rate=self.learning_rate,
            mode='matrix', positive_output=False
        )
        self.base_param_1 = EdgeComparer(
            dimension=self.node_dimension,
            learning_rate=self.learning_rate,
            mode='matrix', positive_output=False
        )
        self.base_param_2 = EdgeComparer(
            dimension=self.node_dimension,
            learning_rate=self.learning_rate,
            mode='matrix', positive_output=False
        )

        # Create attributes to record all edges and any
        # new edges for this day
        self.edge_log = []
        self.new_edges = dict()

        # Store any pairs of edges which will excite each
        # other based on the weight of the corresponding
        # Weibull distribution
        self.possible_excitees = dict()
        self.excitment_threshold = 0.01
        self.find_excitors()

        # Create an attribute to store any edges that are
        # currently excited by previous edges
        self.current_excitees = dict()

        # Create an attribute to store gradients and values
        # for the gradient ascent algorithm
        self.gradient_log = dict()

    def find_excitors(self):
        """Find any pairs of edges which will excite each
        other based on the weight of the corresponding
        Weibull distribution
        """
        # Clear the current excitors
        self.possible_excitees = dict()

        for excitor_i in range(self.count_nodes):
            # Get the node embedding
            excitor_x_i = self.nodes[excitor_i].causal_embeddings.source_value

            for excitor_j in range(self.count_nodes):
                # Get the node embedding
                excitor_x_j = self.nodes[excitor_j].causal_embeddings.dest_value

                # Get the edge embedding
                excitor_edge_embedding = self.edge_embedder.embed_edge(excitor_x_i, excitor_x_j)

                # Create an entry for this excitor
                self.possible_excitees[(excitor_i, excitor_j)] = dict()

                for excitee_i in range(self.count_nodes):
                    # Get the node embedding
                    excitee_x_i = self.nodes[excitee_i].causal_embeddings.source_value

                    for excitee_j in range(self.count_nodes):
                        # Get the node embedding
                        excitee_x_j = self.nodes[excitee_j].causal_embeddings.dest_value

                        # Get the edge embedding
                        excitee_edge_embedding = \
                            self.edge_embedder.embed_edge(excitee_x_i, excitee_x_j)

                        # Get the Weibull weight (the total probability of the excitor
                        # edge exciting the excitee edge)
                        weibull_weight = \
                            self.weibull_weight_generator.compare_embeddings(
                                excitor_edge_embedding, excitee_edge_embedding
                            )
                        # Extract the pre-scaled value from the calculation, for use
                        # in the gradient ascent algorithm
                        lin_val_weight = self.weibull_weight_generator.last_linear_value

                        # If there is sufficient probability, add as a possible excitee
                        if weibull_weight > self.excitment_threshold:
                            # Calculate the Weibull parameters (characterising the average
                            # time to excitation, and the spread of likely times to
                            # excitation)
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

                            # Store the potential pairs, and the requisite parameters
                            self.possible_excitees[
                                (excitor_i, excitor_j)
                                ][
                                    (excitee_i, excitee_j)
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
                    alive_threshold=self.excitment_threshold
                )
            )

    def edge_baseline(self, i, j):
        """The baseline intensity for edges i->j based
        on the nodes' relationship and the respective
        cumulative balances.

        Args:
            i (int): The index of the source node
            j (int): The index of the destination node

        Returns:
            float: The baseline intensity
        """
        # Get the balances at the start of the day
        balance_i = self.nodes[i].balance
        balance_j = self.nodes[j].balance

        # Store the balances for the gradient algorithm
        self.gradient_log['source_balance'] = balance_i
        self.gradient_log['dest_balance'] = balance_j

        # Get the embeddings
        y_i = self.nodes[i].spontaneous_embeddings.source_value
        y_j = self.nodes[j].spontaneous_embeddings.dest_value

        # Calculate the linear output
        linear_output_0 = \
            self.base_param_0.compare_embeddings(
                y_i, y_j
            )
        linear_output_1 = \
            self.base_param_1.compare_embeddings(
                y_i, y_j
            )
        linear_output_2 = \
            self.base_param_2.compare_embeddings(
                y_i, y_j
            )

        full_linear_output = \
            linear_output_0 + \
            balance_i * linear_output_1 + \
            balance_j * linear_output_2

        # Store the linear output for the gradient algorithm
        self.gradient_log['baseline_linear_value'] = \
            full_linear_output

        # Make it positive
        return log_exp_function(full_linear_output)

    def edge_intensity(self, i, j):
        """Get the total intensity for a particular edge

        Args:
            i (int): Index of the source node
            j (int): Index of the destination node

        Returns:
            float: The total intensity of the edge
        """

        # Start with the baseline intensity of the edge
        intensity = self.edge_baseline(i, j)

        # For each 'excite' that exists for this edge,
        # increase the intensity accordingly
        nodes = (i,j)
        if nodes in self.current_excitees:
            # Increase the intensity by the weighted
            # probability mass function of the generated,
            # discrete Weibull distribution
            intensity += sum(
                excite.probability
                for excite in self.current_excitees[nodes]
                )

            # Record all the necessary values for the gradient
            # ascent algorithm

            # Record the Weibull parameters
            self.gradient_log['alphas'] = \
                [excite.weibull_alpha
                 for excite in self.current_excitees[nodes]]
            self.gradient_log['betas'] = \
                [excite.weibull_beta
                 for excite in self.current_excitees[nodes]]
            # Record the weighting of the Weibull distribution
            self.gradient_log['weights'] = \
                [excite.weibull_weight
                 for excite in self.current_excitees[nodes]]

            # Record the linear value from the parameter
            # calculations (before passed through the smooth,
            # continuous function mapping R->R+)
            self.gradient_log['lin_alphas'] = \
                [excite.lin_val_alpha
                 for excite in self.current_excitees[nodes]]
            self.gradient_log['lin_betas'] = \
                [excite.lin_val_beta
                 for excite in self.current_excitees[nodes]]
            self.gradient_log['lin_weights'] = \
                [excite.lin_val_weight
                 for excite in self.current_excitees[nodes]]

            # Record the original time that the excitor edge
            # occurred
            self.gradient_log['times'] = \
                [excite.time
                 for excite in self.current_excitees[nodes]]

            # Record the nodes in the excitor edge
            self.gradient_log['excitor_nodes'] = \
                [excite.excitor_nodes
                 for excite in self.current_excitees[nodes]]
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

        return intensity

    def edge_probability(self, i, j, count):
        """The probability that a particular edge occured a
        given number of times on this day.

        Args:
            i (int): Index of the source node
            j (int): Index of the destination node
            count (int): The number of times that this edge
                occurred on this day

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
            mu=self.edge_intensity(i, j)
        )

        # Record the probability for gradient ascent
        self.gradient_log['P'] = probability

        # Run the gradient ascent algorithm to create
        # pending parameter updates (to be applied when
        # the entire period has been added to the graph)
        self.gradient_ascent()

        return probability

    def day_log_probability(self):
        """The log probability of all the edges that have
        occurred on this day

        Returns:
            float: The log probability
        """

        # Iterate through all possible edges
        log_probability = 0
        for i in range(self.count_nodes):
            for j in range(self.count_nodes):
                # Count how many times (if any) that
                # edge has occurred
                if (i, j) in self.new_edges:
                    count = self.new_edges[(i, j)]
                else:
                    count = 0

                # Take the log of the probability of
                # that edge occurring that number of
                # times, and add to the running total
                log_probability += log(
                    self.edge_probability(
                        i, j, count
                    )
                    )

        return log_probability

    def gradient_ascent(self):
        """Run the gradient ascent algorithm using
        cached calculations of the probability of
        today's edges occuring in the frequency which
        they did.
        Gradient ascent will maximise this probability,
        thereby giving a maximum likelihood estimation
        for the parameters.
        The cached calculations will be for the probability
        of a certain edge occuring a certain number of times.
        Therefore, the excitee edge is fixed, but there
        are potentially a number of different excitor edges.
        Therefore, node embeddings for the nodes in any of
        these involved edges can be learned. The matrices
        in the Comparer objects can also be learned.
        """

        # Calculate the inverse of the probability
        inverse_probability = 1/self.gradient_log['P']

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
        x_k = node_k.causal_embeddings.dest_value
        node_l = self.nodes[l]
        x_l = node_l.causal_embeddings.dest_value

        # Get the edge embedding
        e_kl = self.edge_embedder.embed_edge(x_k, x_l)

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

        y_k = node_k.spontaneous_embeddings.dest_value
        y_l = node_l.spontaneous_embeddings.dest_value
        delZero_delK = \
            calc_delBaselineComparer_delK(
                self.base_param_0.matrix,
                y_l
            )
        delZero_delL = \
            calc_delBaselineComparer_delL(
                self.base_param_0.matrix,
                y_k
            )
        delOne_delK = \
            calc_delBaselineComparer_delK(
                self.base_param_1.matrix,
                y_l
            )
        delOne_delL = \
            calc_delBaselineComparer_delL(
                self.base_param_1.matrix,
                y_k
            )
        delTwo_delK = \
            calc_delBaselineComparer_delK(
                self.base_param_2.matrix,
                y_l
            )
        delTwo_delL = \
            calc_delBaselineComparer_delL(
                self.base_param_2.matrix,
                y_k
            )
        delBaselineComparer_delMatrix = \
            calc_delBaselineComparer_delMatrix(
                y_k, y_l
            )

        # - Apply the updates
        node_k.add_spontaneous_gradient_update(
            inverse_probability * delP_delIntensity * (
                delBaselineIntensity_delZero *
                delZero_delK +
                delBaselineIntensity_delOne *
                delOne_delK +
                delBaselineIntensity_delTwo *
                delTwo_delK
            ),
            node_type='source'
        )

        node_l.add_spontaneous_gradient_update(
            inverse_probability * delP_delIntensity * (
                delBaselineIntensity_delZero *
                delZero_delL +
                delBaselineIntensity_delOne *
                delOne_delL +
                delBaselineIntensity_delTwo *
                delTwo_delL
            ),
            node_type='dest'
        )

        try:
            self.base_param_0.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delZero *
                    delBaselineComparer_delMatrix
                )
            )
        except FloatingPointError as e:
            print(
                'Base Param 0, Floating point error',
                inverse_probability,
                delP_delIntensity,
                delBaselineIntensity_delZero,
                delBaselineComparer_delMatrix
                )
            raise e

        try:
            self.base_param_1.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delOne *
                    delBaselineComparer_delMatrix
                )
            )
        except FloatingPointError as e:
            print(
                'Base Param 1, Floating point error',
                inverse_probability,
                delP_delIntensity,
                delBaselineIntensity_delZero,
                delBaselineComparer_delMatrix
                )
            raise e

        try:
            self.base_param_2.add_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delBaselineIntensity_delTwo *
                    delBaselineComparer_delMatrix
                )
            )
        except FloatingPointError as e:
            print(
                'Base Param 2, Floating point error',
                inverse_probability,
                delP_delIntensity,
                delBaselineIntensity_delTwo,
                delBaselineComparer_delMatrix
                )
            raise e

        for excite_index, (i, j) in enumerate(self.gradient_log['excitor_nodes']):
            # Get linear values from the calculations of the
            # Comparer objects
            lin_val_alpha = self.gradient_log['lin_alphas'][excite_index]
            lin_val_beta = self.gradient_log['lin_betas'][excite_index]
            lin_val_weight = self.gradient_log['lin_weights'][excite_index]

            # Get the node embeddings for the nodes in
            # the excitor edge
            node_i = self.nodes[i]
            x_i = node_i.causal_embeddings.dest_value
            node_j = self.nodes[j]
            x_j = node_j.causal_embeddings.dest_value

            # Get the edge embedding
            e_ij = self.edge_embedder.embed_edge(x_i, x_j)

            # Calculate the partial derivates that depend
            # on the excitor edge
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

            # Apply the gradient updates
            # Node i
            node_i.add_causal_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delI +
                    delIntensity_delBeta[excite_index] *
                    delBeta_delI +
                    delIntensity_delWeight[excite_index] *
                    delWeight_delI
                ),
                node_type='source'
            )

            # Node j
            node_j.add_causal_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delJ +
                    delIntensity_delBeta[excite_index] *
                    delBeta_delJ +
                    delIntensity_delWeight[excite_index] *
                    delWeight_delJ
                ),
                node_type='dest'
            )

            # Node k
            node_k.add_causal_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delK +
                    delIntensity_delBeta[excite_index] *
                    delBeta_delK +
                    delIntensity_delWeight[excite_index] *
                    delWeight_delK
                ),
                node_type='source'
            )

            # Node l
            node_l.add_causal_gradient_update(
                inverse_probability * delP_delIntensity * (
                    delIntensity_delAlpha[excite_index] *
                    delAlpha_delL +
                    delIntensity_delBeta[excite_index] *
                    delBeta_delL +
                    delIntensity_delWeight[excite_index] *
                    delWeight_delL
                ),
                node_type='dest'
            )

            # Weight matrix
            self.weibull_weight_generator.add_gradient_update(
                inverse_probability * delP_delIntensity *
                delIntensity_delAlpha[excite_index] *
                delWeightComparerdelMatrix
            )

            # Alpha matrix
            self.weibull_alpha_generator.add_gradient_update(
                inverse_probability * delP_delIntensity *
                delIntensity_delBeta[excite_index] *
                delAlphaComparerdelMatrix
            )

            # Beta matrix
            self.weibull_beta_generator.add_gradient_update(
                inverse_probability * delP_delIntensity *
                delIntensity_delWeight[excite_index] *
                delBetaComparerdelMatrix
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

        # Update matrices in the Comparer objects
        self.weibull_alpha_generator.apply_gradient_updates()
        self.weibull_beta_generator.apply_gradient_updates()
        self.weibull_weight_generator.apply_gradient_updates()

    def reset(self):
        """Remove edges and excitation to prepare for another
        epoch of training (embeddings and matrice learnings
        are retained)
        """

        # Set the time back to the start of the period
        self.time = 0

        # Clear any recorded edges
        self.edge_log = []
        self.new_edges = dict()

        # Clear any excitation
        self.possible_excitees = dict()
        self.current_excitees = dict()

        # Apply gradient updates
        self.apply_gradient_updates()

        # Update pairs of edges which could
        # excite each other under the updated
        # model parameters
        self.find_excitors()
