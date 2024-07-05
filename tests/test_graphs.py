import unittest
import numpy as np

import graphs


class Test_GradientAscentCalculations(unittest.TestCase):
    """Test the derivative of the log likelihood for a
    particular edge with respect to every parameter
    """
    def setUp(self):
        """Create a Graph class
        """
        # Fix seed
        np.random.seed(628496)

        # Create accounts
        class Account:
            """A class to hold all the attributes of an account
            """
            def __init__(self, name, number, balance, mapping):
                """Initialise the class

                Args:
                    name (str): The name of the account
                    number (str): The unique identifier for the account
                    balance (float): The monetary balance on the account
                        at the start of the accounting period
                    mapping (str): The accounting concept to which the
                        account belongs (e.g. Revenue or Debtors)
                """
                self.name = name
                self.number = number
                self.balance = balance
                self.mapping = mapping

        sales_account = Account('sales', '7000', -0.1, 'revenue')
        debtors_account = Account('debtors', '4000', 0.4, 'debtors')
        bank_account = Account('bank', '5500', 0.6, 'bank')

        self.graph = graphs.DynamicAccountingGraph(
            accounts=[sales_account, debtors_account, bank_account],
            node_dimension=4,
            learning_rate=1
        )

    def reset_graph(self, edges):
        self.graph.reset(discard_gradient_updates=True)

        for time, day_edges in edges.items():
            for i, j, value in day_edges:
                # Add an edge
                self.graph.add_edge(i, j, value)

            # Move time on
            self.graph.increment_time()

    def derivative_helper(self, i, j, count, edges,
                          sales_x_i=None, sales_x_j=None, sales_y_i=None, sales_y_j=None,
                          debtors_x_i=None, debtors_x_j=None, debtors_y_i=None, debtors_y_j=None,
                          bank_x_i=None, bank_x_j=None, bank_y_i=None, bank_y_j=None,
                          A_weight=None, A_alpha=None, A_beta=None,
                          A_zero=None, A_one=None, A_two=None,
                          epsilon=10**(-8)):
        node_dimension = 4

        # Override any provided parameters
        if sales_x_i is not None:
            self.graph.nodes[0].causal_embeddings.source_value = sales_x_i
        else:
            sales_x_i = self.graph.nodes[0].causal_embeddings.source_value
        if sales_x_j is not None:
            self.graph.nodes[0].causal_embeddings.dest_value = sales_x_j
        else:
            sales_x_j = self.graph.nodes[0].causal_embeddings.dest_value
        if debtors_x_i is not None:
            self.graph.nodes[1].causal_embeddings.source_value = debtors_x_i
        else:
            debtors_x_i = self.graph.nodes[1].causal_embeddings.source_value
        if debtors_x_j is not None:
            self.graph.nodes[1].causal_embeddings.dest_value = debtors_x_j
        else:
            debtors_x_j = self.graph.nodes[1].causal_embeddings.dest_value
        if bank_x_i is not None:
            self.graph.nodes[2].causal_embeddings.source_value = bank_x_i
        else:
            bank_x_i = self.graph.nodes[2].causal_embeddings.source_value
        if bank_x_j is not None:
            self.graph.nodes[2].causal_embeddings.dest_value = bank_x_j
        else:
            bank_x_j = self.graph.nodes[2].causal_embeddings.dest_value

        if sales_y_i is not None:
            self.graph.nodes[0].spontaneous_embeddings.source_value = sales_y_i
        else:
            sales_y_i = self.graph.nodes[0].spontaneous_embeddings.source_value
        if sales_y_j is not None:
            self.graph.nodes[0].spontaneous_embeddings.dest_value = sales_y_j
        else:
            sales_y_j = self.graph.nodes[0].spontaneous_embeddings.dest_value
        if debtors_y_i is not None:
            self.graph.nodes[1].spontaneous_embeddings.source_value = debtors_y_i
        else:
            debtors_y_i = self.graph.nodes[1].spontaneous_embeddings.source_value
        if debtors_y_j is not None:
            self.graph.nodes[1].spontaneous_embeddings.dest_value = debtors_y_j
        else:
            debtors_y_j = self.graph.nodes[1].spontaneous_embeddings.dest_value
        if bank_y_i is not None:
            self.graph.nodes[2].spontaneous_embeddings.source_value = bank_y_i
        else:
            bank_y_i = self.graph.nodes[2].spontaneous_embeddings.source_value
        if bank_y_j is not None:
            self.graph.nodes[2].spontaneous_embeddings.dest_value = bank_y_j
        else:
            bank_y_j = self.graph.nodes[2].spontaneous_embeddings.dest_value

        if A_weight is not None:
            self.graph.weibull_weight_generator.matrix = A_weight
        else:
            A_weight = self.graph.weibull_weight_generator.matrix
        if A_alpha is not None:
            self.graph.weibull_alpha_generator.matrix = A_alpha
        else:
            A_alpha = self.graph.weibull_alpha_generator.matrix
        if A_beta is not None:
            self.graph.weibull_beta_generator.matrix = A_beta
        else:
            A_beta = self.graph.weibull_beta_generator.matrix

        if A_zero is not None:
            self.graph.base_param_0.matrix = A_zero
        else:
            A_zero = self.graph.base_param_0.matrix
        if A_one is not None:
            self.graph.base_param_1.matrix = A_one
        else:
            A_one = self.graph.base_param_1.matrix
        if A_two is not None:
            self.graph.base_param_2.matrix = A_two
        else:
            A_two = self.graph.base_param_2.matrix

        # Update the graph excitement based on these parameters
        self.reset_graph(edges)

        # Calculate the function value
        base_value = \
            np.log(self.graph.edge_probability(
                i, j, count
            ))

        # Extract the calculated derivatives
        calc_deriv_sales_x_i = \
            self.graph.nodes[0].causal_embeddings.source_pending_updates.copy()
        calc_deriv_sales_x_j = \
            self.graph.nodes[0].causal_embeddings.dest_pending_updates.copy()
        calc_deriv_debtors_x_i = \
            self.graph.nodes[1].causal_embeddings.source_pending_updates.copy()
        calc_deriv_debtors_x_j = \
            self.graph.nodes[1].causal_embeddings.dest_pending_updates.copy()
        calc_deriv_bank_x_i = \
            self.graph.nodes[2].causal_embeddings.source_pending_updates.copy()
        calc_deriv_bank_x_j = \
            self.graph.nodes[2].causal_embeddings.dest_pending_updates.copy()

        calc_deriv_sales_y_i = \
            self.graph.nodes[0].spontaneous_embeddings.source_pending_updates.copy()
        calc_deriv_sales_y_j = \
            self.graph.nodes[0].spontaneous_embeddings.dest_pending_updates.copy()
        calc_deriv_debtors_y_i = \
            self.graph.nodes[1].spontaneous_embeddings.source_pending_updates.copy()
        calc_deriv_debtors_y_j = \
            self.graph.nodes[1].spontaneous_embeddings.dest_pending_updates.copy()
        calc_deriv_bank_y_i = \
            self.graph.nodes[2].spontaneous_embeddings.source_pending_updates.copy()
        calc_deriv_bank_y_j = \
            self.graph.nodes[2].spontaneous_embeddings.dest_pending_updates.copy()

        calc_deriv_A_weight = \
            self.graph.weibull_weight_generator.pending_updates.copy()
        calc_deriv_A_alpha = \
            self.graph.weibull_alpha_generator.pending_updates.copy()
        calc_deriv_A_beta = \
            self.graph.weibull_beta_generator.pending_updates.copy()

        calc_deriv_A_zero = \
            self.graph.base_param_0.pending_updates.copy()
        calc_deriv_A_one = \
            self.graph.base_param_1.pending_updates.copy()
        calc_deriv_A_two = \
            self.graph.base_param_2.pending_updates.copy()

        # Estimate the derivatives
        # Sales source node (causal)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_embeddings.source_value = sales_x_i + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_embeddings.source_value = sales_x_i

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales node I (causal), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_x_i[a],
                    estimated_deriv,
                    places=5
                )

        # Sales destination node (causal)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_embeddings.dest_value = sales_x_j + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_embeddings.dest_value = sales_x_j

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales node J (causal), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_x_j[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors source node (causal)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_embeddings.source_value = debtors_x_i + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_embeddings.source_value = debtors_x_i

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'debtors node I (causal), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_x_i[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors destination node (causal)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_embeddings.dest_value = debtors_x_j + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_embeddings.dest_value = debtors_x_j

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'debtors node J (causal), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_x_j[a],
                    estimated_deriv,
                    places=5
                )

        # Bank source node (causal)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_embeddings.source_value = bank_x_i + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_embeddings.source_value = bank_x_i

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'bank node I (causal), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_x_i[a],
                    estimated_deriv,
                    places=5
                )

        # Bank destination node (causal)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_embeddings.dest_value = bank_x_j + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_embeddings.dest_value = bank_x_j

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'bank node J (causal), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_x_j[a],
                    estimated_deriv,
                    places=5
                )

        # Sales source node (spontaneous)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].spontaneous_embeddings.source_value = sales_y_i + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].spontaneous_embeddings.source_value = sales_y_i

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales node I (spontaneous), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_y_i[a],
                    estimated_deriv,
                    places=5
                )

        # Sales destination node (spontaneous)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].spontaneous_embeddings.dest_value = sales_y_j + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].spontaneous_embeddings.dest_value = sales_y_j

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales node J (spontaneous), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_y_j[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors source node (spontaneous)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].spontaneous_embeddings.source_value = debtors_y_i + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].spontaneous_embeddings.source_value = debtors_y_i

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'debtors node I (spontaneous), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_y_i[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors destination node (spontaneous)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].spontaneous_embeddings.dest_value = debtors_y_j + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].spontaneous_embeddings.dest_value = debtors_y_j

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'debtors node J (spontaneous), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_y_j[a],
                    estimated_deriv,
                    places=5
                )

        # Bank source node (spontaneous)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].spontaneous_embeddings.source_value = bank_y_i + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].spontaneous_embeddings.source_value = bank_y_i

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'bank node I (spontaneous), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_y_i[a],
                    estimated_deriv,
                    places=5
                )

        # Bank destination node (spontaneous)
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].spontaneous_embeddings.dest_value = bank_y_j + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].spontaneous_embeddings.dest_value = bank_y_j

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'bank node J (spontaneous), dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_y_j[a],
                    estimated_deriv,
                    places=5
                )

        # Weibull alpha matrix
        for a in range(node_dimension*2):
            for b in range(node_dimension*2):
                # Create a step in the appropriate dimension
                increment = np.zeros(
                    (node_dimension*2, node_dimension*2))
                increment[a][b] = epsilon

                # Adjust the parameter accordingly
                self.graph.weibull_alpha_generator.matrix = A_alpha + increment

                # Update the graph excitement based on these parameters
                self.reset_graph(edges)

                # Recalculate the function value
                next_value = \
                    np.log(self.graph.edge_probability(
                        i, j, count
                    ))

                # Reset the parameter change
                self.graph.weibull_alpha_generator.matrix = A_alpha

                # Estimate the partial derivative wrt this parameter
                estimated_deriv = (next_value-base_value)/epsilon

                with self.subTest(msg=f'Weibull Alpha matrix, component {a},{b}'):
                    self.assertAlmostEqual(
                        calc_deriv_A_alpha[a][b],
                        estimated_deriv,
                        places=5
                    )

        # Weibull beta matrix
        for a in range(node_dimension*2):
            for b in range(node_dimension*2):
                # Create a step in the appropriate dimension
                increment = np.zeros(
                    (node_dimension*2, node_dimension*2))
                increment[a][b] = epsilon

                # Adjust the parameter accordingly
                self.graph.weibull_beta_generator.matrix = A_beta + increment

                # Update the graph excitement based on these parameters
                self.reset_graph(edges)

                # Recalculate the function value
                next_value = \
                    np.log(self.graph.edge_probability(
                        i, j, count
                    ))

                # Reset the parameter change
                self.graph.weibull_beta_generator.matrix = A_beta

                # Estimate the partial derivative wrt this parameter
                estimated_deriv = (next_value-base_value)/epsilon

                with self.subTest(msg=f'Weibull beta matrix, component {a},{b}'):
                    self.assertAlmostEqual(
                        calc_deriv_A_beta[a][b],
                        estimated_deriv,
                        places=5
                    )

        # Weibull weight matrix
        for a in range(node_dimension*2):
            for b in range(node_dimension*2):
                # Create a step in the appropriate dimension
                increment = np.zeros(
                    (node_dimension*2, node_dimension*2))
                increment[a][b] = epsilon

                # Adjust the parameter accordingly
                self.graph.weibull_weight_generator.matrix = A_weight + increment

                # Update the graph excitement based on these parameters
                self.reset_graph(edges)

                # Recalculate the function value
                next_value = \
                    np.log(self.graph.edge_probability(
                        i, j, count
                    ))

                # Reset the parameter change
                self.graph.weibull_weight_generator.matrix = A_weight

                # Estimate the partial derivative wrt this parameter
                estimated_deriv = (next_value-base_value)/epsilon

                with self.subTest(msg=f'Weibull weight matrix, component {a},{b}'):
                    self.assertAlmostEqual(
                        calc_deriv_A_weight[a][b],
                        estimated_deriv,
                        places=5
                    )

        # A_0 matrix
        for a in range(node_dimension):
            for b in range(node_dimension):
                # Create a step in the appropriate dimension
                increment = np.zeros(
                    (node_dimension, node_dimension))
                increment[a][b] = epsilon

                # Adjust the parameter accordingly
                self.graph.base_param_0.matrix = A_zero + increment

                # Update the graph excitement based on these parameters
                self.reset_graph(edges)

                # Recalculate the function value
                next_value = \
                    np.log(self.graph.edge_probability(
                        i, j, count
                    ))

                # Reset the parameter change
                self.graph.base_param_0.matrix = A_zero

                # Estimate the partial derivative wrt this parameter
                estimated_deriv = (next_value-base_value)/epsilon

                with self.subTest(msg=f'A_0 matrix, component {a},{b}'):
                    self.assertAlmostEqual(
                        calc_deriv_A_zero[a][b],
                        estimated_deriv,
                        places=5
                    )

        # A_1 matrix
        for a in range(node_dimension):
            for b in range(node_dimension):
                # Create a step in the appropriate dimension
                increment = np.zeros(
                    (node_dimension, node_dimension))
                increment[a][b] = epsilon

                # Adjust the parameter accordingly
                self.graph.base_param_1.matrix = A_one + increment

                # Update the graph excitement based on these parameters
                self.reset_graph(edges)

                # Recalculate the function value
                next_value = \
                    np.log(self.graph.edge_probability(
                        i, j, count
                    ))

                # Reset the parameter change
                self.graph.base_param_1.matrix = A_one

                # Estimate the partial derivative wrt this parameter
                estimated_deriv = (next_value-base_value)/epsilon

                with self.subTest(msg=f'A_1 matrix, component {a},{b}'):
                    self.assertAlmostEqual(
                        calc_deriv_A_one[a][b],
                        estimated_deriv,
                        places=5
                    )

        # A_2 matrix
        for a in range(node_dimension):
            for b in range(node_dimension):
                # Create a step in the appropriate dimension
                increment = np.zeros(
                    (node_dimension, node_dimension))
                increment[a][b] = epsilon

                # Adjust the parameter accordingly
                self.graph.base_param_2.matrix = A_two + increment

                # Update the graph excitement based on these parameters
                self.reset_graph(edges)

                # Recalculate the function value
                next_value = \
                    np.log(self.graph.edge_probability(
                        i, j, count
                    ))

                # Reset the parameter change
                self.graph.base_param_2.matrix = A_two

                # Estimate the partial derivative wrt this parameter
                estimated_deriv = (next_value-base_value)/epsilon

                with self.subTest(msg=f'A_2 matrix, component {a},{b}'):
                    self.assertAlmostEqual(
                        calc_deriv_A_two[a][b],
                        estimated_deriv,
                        places=5
                    )

    def test_deriv(self):
        # Define edges
        edges = {}

        # Choose the derivate
        self.derivative_helper(
            0, 1, 2,
            edges=edges
        )

    def test_with_edge(self):
        # Define edges
        edges = {
            0: [(0, 1, 10)]
        }

        # Calculate the derivate
        self.derivative_helper(
            0, 1, 2,
            edges=edges
        )

    def test_with_edge_debtor_settle(self):
        # Define edges
        edges = {
            0: [(1, 2, 0)]
        }

        # Calculate the derivate
        self.derivative_helper(
            0, 1, 2,
            edges=edges
        )

    def test_with_edge_cash_sale(self):
        # Define edges
        edges = {
            0: [(0, 1, 10)]
        }

        # Calculate the derivate
        self.derivative_helper(
            0, 2, 3,
            edges=edges
        )
