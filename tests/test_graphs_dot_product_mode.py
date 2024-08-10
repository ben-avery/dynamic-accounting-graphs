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
            average_balances=[1,1,1],
            average_weight=1,
            node_dimension=4
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
                          epsilon=10**(-8)):
        node_dimension = 4

        # Extract parameters
        sales_r_i_alpha = self.graph.nodes[0].causal_excitor_source_alpha.value
        sales_r_i_beta = self.graph.nodes[0].causal_excitor_source_beta.value
        sales_r_i_weight = self.graph.nodes[0].causal_excitor_source_weight.value

        sales_e_k_alpha = self.graph.nodes[0].causal_excitee_source_alpha.value
        sales_e_k_beta = self.graph.nodes[0].causal_excitee_source_beta.value
        sales_e_k_weight = self.graph.nodes[0].causal_excitee_source_weight.value

        debtors_r_i_alpha = self.graph.nodes[1].causal_excitor_source_alpha.value
        debtors_r_i_beta = self.graph.nodes[1].causal_excitor_source_beta.value
        debtors_r_i_weight = self.graph.nodes[1].causal_excitor_source_weight.value

        debtors_e_k_alpha = self.graph.nodes[1].causal_excitee_source_alpha.value
        debtors_e_k_beta = self.graph.nodes[1].causal_excitee_source_beta.value
        debtors_e_k_weight = self.graph.nodes[1].causal_excitee_source_weight.value

        bank_r_i_alpha = self.graph.nodes[2].causal_excitor_source_alpha.value
        bank_r_i_beta = self.graph.nodes[2].causal_excitor_source_beta.value
        bank_r_i_weight = self.graph.nodes[2].causal_excitor_source_weight.value

        bank_e_k_alpha = self.graph.nodes[2].causal_excitee_source_alpha.value
        bank_e_k_beta = self.graph.nodes[2].causal_excitee_source_beta.value
        bank_e_k_weight = self.graph.nodes[2].causal_excitee_source_weight.value

        sales_r_j_alpha = self.graph.nodes[0].causal_excitor_dest_alpha.value
        sales_r_j_beta = self.graph.nodes[0].causal_excitor_dest_beta.value
        sales_r_j_weight = self.graph.nodes[0].causal_excitor_dest_weight.value

        sales_e_l_alpha = self.graph.nodes[0].causal_excitee_dest_alpha.value
        sales_e_l_beta = self.graph.nodes[0].causal_excitee_dest_beta.value
        sales_e_l_weight = self.graph.nodes[0].causal_excitee_dest_weight.value

        debtors_r_j_alpha = self.graph.nodes[1].causal_excitor_dest_alpha.value
        debtors_r_j_beta = self.graph.nodes[1].causal_excitor_dest_beta.value
        debtors_r_j_weight = self.graph.nodes[1].causal_excitor_dest_weight.value

        debtors_e_l_alpha = self.graph.nodes[1].causal_excitee_dest_alpha.value
        debtors_e_l_beta = self.graph.nodes[1].causal_excitee_dest_beta.value
        debtors_e_l_weight = self.graph.nodes[1].causal_excitee_dest_weight.value

        bank_r_j_alpha = self.graph.nodes[2].causal_excitor_dest_alpha.value
        bank_r_j_beta = self.graph.nodes[2].causal_excitor_dest_beta.value
        bank_r_j_weight = self.graph.nodes[2].causal_excitor_dest_weight.value

        bank_e_l_alpha = self.graph.nodes[2].causal_excitee_dest_alpha.value
        bank_e_l_beta = self.graph.nodes[2].causal_excitee_dest_beta.value
        bank_e_l_weight = self.graph.nodes[2].causal_excitee_dest_weight.value

        sales_s_i_zero = self.graph.nodes[0].spontaneous_source_0.value
        sales_s_i_one = self.graph.nodes[0].spontaneous_source_1.value
        sales_s_i_two = self.graph.nodes[0].spontaneous_source_2.value

        sales_s_j_zero = self.graph.nodes[0].spontaneous_dest_0.value
        sales_s_j_one = self.graph.nodes[0].spontaneous_dest_1.value
        sales_s_j_two = self.graph.nodes[0].spontaneous_dest_2.value

        debtors_s_i_zero = self.graph.nodes[1].spontaneous_source_0.value
        debtors_s_i_one = self.graph.nodes[1].spontaneous_source_1.value
        debtors_s_i_two = self.graph.nodes[1].spontaneous_source_2.value

        debtors_s_j_zero = self.graph.nodes[1].spontaneous_dest_0.value
        debtors_s_j_one = self.graph.nodes[1].spontaneous_dest_1.value
        debtors_s_j_two = self.graph.nodes[1].spontaneous_dest_2.value

        bank_s_i_zero = self.graph.nodes[2].spontaneous_source_0.value
        bank_s_i_one = self.graph.nodes[2].spontaneous_source_1.value
        bank_s_i_two = self.graph.nodes[2].spontaneous_source_2.value

        bank_s_j_zero = self.graph.nodes[2].spontaneous_dest_0.value
        bank_s_j_one = self.graph.nodes[2].spontaneous_dest_1.value
        bank_s_j_two = self.graph.nodes[2].spontaneous_dest_2.value

        self.reset_graph(edges)

        # Calculate the function value
        base_value = \
            np.log(self.graph.edge_probability(
                i, j, count
            ))

        # Extract the calculated derivatives
        # Causal excitor (alpha)
        calc_deriv_sales_r_i_alpha = \
            self.graph.nodes[0].causal_excitor_source_alpha.pending_updates.copy()
        calc_deriv_sales_r_j_alpha = \
            self.graph.nodes[0].causal_excitor_dest_alpha.pending_updates.copy()
        calc_deriv_debtors_r_i_alpha = \
            self.graph.nodes[1].causal_excitor_source_alpha.pending_updates.copy()
        calc_deriv_debtors_r_j_alpha = \
            self.graph.nodes[1].causal_excitor_dest_alpha.pending_updates.copy()
        calc_deriv_bank_r_i_alpha = \
            self.graph.nodes[2].causal_excitor_source_alpha.pending_updates.copy()
        calc_deriv_bank_r_j_alpha = \
            self.graph.nodes[2].causal_excitor_dest_alpha.pending_updates.copy()

        # Causal excitor (beta)
        calc_deriv_sales_r_i_beta = \
            self.graph.nodes[0].causal_excitor_source_beta.pending_updates.copy()
        calc_deriv_sales_r_j_beta = \
            self.graph.nodes[0].causal_excitor_dest_beta.pending_updates.copy()
        calc_deriv_debtors_r_i_beta = \
            self.graph.nodes[1].causal_excitor_source_beta.pending_updates.copy()
        calc_deriv_debtors_r_j_beta = \
            self.graph.nodes[1].causal_excitor_dest_beta.pending_updates.copy()
        calc_deriv_bank_r_i_beta = \
            self.graph.nodes[2].causal_excitor_source_beta.pending_updates.copy()
        calc_deriv_bank_r_j_beta = \
            self.graph.nodes[2].causal_excitor_dest_beta.pending_updates.copy()

        # Causal excitor (weight)
        calc_deriv_sales_r_i_weight = \
            self.graph.nodes[0].causal_excitor_source_weight.pending_updates.copy()
        calc_deriv_sales_r_j_weight = \
            self.graph.nodes[0].causal_excitor_dest_weight.pending_updates.copy()
        calc_deriv_debtors_r_i_weight = \
            self.graph.nodes[1].causal_excitor_source_weight.pending_updates.copy()
        calc_deriv_debtors_r_j_weight = \
            self.graph.nodes[1].causal_excitor_dest_weight.pending_updates.copy()
        calc_deriv_bank_r_i_weight = \
            self.graph.nodes[2].causal_excitor_source_weight.pending_updates.copy()
        calc_deriv_bank_r_j_weight = \
            self.graph.nodes[2].causal_excitor_dest_weight.pending_updates.copy()

        # Causal excitee (alpha)
        calc_deriv_sales_e_k_alpha = \
            self.graph.nodes[0].causal_excitee_source_alpha.pending_updates.copy()
        calc_deriv_sales_e_l_alpha = \
            self.graph.nodes[0].causal_excitee_dest_alpha.pending_updates.copy()
        calc_deriv_debtors_e_k_alpha = \
            self.graph.nodes[1].causal_excitee_source_alpha.pending_updates.copy()
        calc_deriv_debtors_e_l_alpha = \
            self.graph.nodes[1].causal_excitee_dest_alpha.pending_updates.copy()
        calc_deriv_bank_e_k_alpha = \
            self.graph.nodes[2].causal_excitee_source_alpha.pending_updates.copy()
        calc_deriv_bank_e_l_alpha = \
            self.graph.nodes[2].causal_excitee_dest_alpha.pending_updates.copy()

        # Causal excitee (beta)
        calc_deriv_sales_e_k_beta = \
            self.graph.nodes[0].causal_excitee_source_beta.pending_updates.copy()
        calc_deriv_sales_e_l_beta = \
            self.graph.nodes[0].causal_excitee_dest_beta.pending_updates.copy()
        calc_deriv_debtors_e_k_beta = \
            self.graph.nodes[1].causal_excitee_source_beta.pending_updates.copy()
        calc_deriv_debtors_e_l_beta = \
            self.graph.nodes[1].causal_excitee_dest_beta.pending_updates.copy()
        calc_deriv_bank_e_k_beta = \
            self.graph.nodes[2].causal_excitee_source_beta.pending_updates.copy()
        calc_deriv_bank_e_l_beta = \
            self.graph.nodes[2].causal_excitee_dest_beta.pending_updates.copy()

        # Causal excitee (weight)
        calc_deriv_sales_e_k_weight = \
            self.graph.nodes[0].causal_excitee_source_weight.pending_updates.copy()
        calc_deriv_sales_e_l_weight = \
            self.graph.nodes[0].causal_excitee_dest_weight.pending_updates.copy()
        calc_deriv_debtors_e_k_weight = \
            self.graph.nodes[1].causal_excitee_source_weight.pending_updates.copy()
        calc_deriv_debtors_e_l_weight = \
            self.graph.nodes[1].causal_excitee_dest_weight.pending_updates.copy()
        calc_deriv_bank_e_k_weight = \
            self.graph.nodes[2].causal_excitee_source_weight.pending_updates.copy()
        calc_deriv_bank_e_l_weight = \
            self.graph.nodes[2].causal_excitee_dest_weight.pending_updates.copy()

        # Spontaneous (zero)
        calc_deriv_sales_s_i_zero = \
            self.graph.nodes[0].spontaneous_source_0.pending_updates.copy()
        calc_deriv_sales_s_j_zero = \
            self.graph.nodes[0].spontaneous_dest_0.pending_updates.copy()
        calc_deriv_debtors_s_i_zero = \
            self.graph.nodes[1].spontaneous_source_0.pending_updates.copy()
        calc_deriv_debtors_s_j_zero = \
            self.graph.nodes[1].spontaneous_dest_0.pending_updates.copy()
        calc_deriv_bank_s_i_zero = \
            self.graph.nodes[2].spontaneous_source_0.pending_updates.copy()
        calc_deriv_bank_s_j_zero = \
            self.graph.nodes[2].spontaneous_dest_0.pending_updates.copy()

        # Spontaneous (one)
        calc_deriv_sales_s_i_one = \
            self.graph.nodes[0].spontaneous_source_1.pending_updates.copy()
        calc_deriv_sales_s_j_one = \
            self.graph.nodes[0].spontaneous_dest_1.pending_updates.copy()
        calc_deriv_debtors_s_i_one = \
            self.graph.nodes[1].spontaneous_source_1.pending_updates.copy()
        calc_deriv_debtors_s_j_one = \
            self.graph.nodes[1].spontaneous_dest_1.pending_updates.copy()
        calc_deriv_bank_s_i_one = \
            self.graph.nodes[2].spontaneous_source_1.pending_updates.copy()
        calc_deriv_bank_s_j_one = \
            self.graph.nodes[2].spontaneous_dest_1.pending_updates.copy()

        # Spontaneous (two)
        calc_deriv_sales_s_i_two = \
            self.graph.nodes[0].spontaneous_source_2.pending_updates.copy()
        calc_deriv_sales_s_j_two = \
            self.graph.nodes[0].spontaneous_dest_2.pending_updates.copy()
        calc_deriv_debtors_s_i_two = \
            self.graph.nodes[1].spontaneous_source_2.pending_updates.copy()
        calc_deriv_debtors_s_j_two = \
            self.graph.nodes[1].spontaneous_dest_2.pending_updates.copy()
        calc_deriv_bank_s_i_two = \
            self.graph.nodes[2].spontaneous_source_2.pending_updates.copy()
        calc_deriv_bank_s_j_two = \
            self.graph.nodes[2].spontaneous_dest_2.pending_updates.copy()

        # Estimate the derivatives
        # Sales excitor source node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitor_source_alpha.value = \
                sales_r_i_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitor_source_alpha.value = \
                sales_r_i_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitor node I (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_r_i_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Sales excitor source node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitor_source_beta.value = \
                sales_r_i_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitor_source_beta.value = \
                sales_r_i_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitor node I (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_r_i_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Sales excitor source node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitor_source_weight.value = \
                sales_r_i_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitor_source_weight.value = \
                sales_r_i_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitor node I (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_r_i_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Sales excitor dest node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitor_dest_alpha.value = \
                sales_r_j_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitor_dest_alpha.value = \
                sales_r_j_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitor node J (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_r_j_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Sales excitor dest node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitor_dest_beta.value = \
                sales_r_j_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitor_dest_beta.value = \
                sales_r_j_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitor node J (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_r_j_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Sales excitor dest node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitor_dest_weight.value = \
                sales_r_j_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitor_dest_weight.value = \
                sales_r_j_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitor node J (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_r_j_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Sales excitee source node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitee_source_alpha.value = \
                sales_e_k_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitee_source_alpha.value = \
                sales_e_k_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitee node K (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_e_k_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Sales excitee source node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitee_source_beta.value = \
                sales_e_k_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitee_source_beta.value = \
                sales_e_k_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitee node K (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_e_k_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Sales excitee source node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitee_source_weight.value = \
                sales_e_k_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitee_source_weight.value = \
                sales_e_k_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitee node K (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_e_k_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Sales excitee dest node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitee_dest_alpha.value = \
                sales_e_l_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitee_dest_alpha.value = \
                sales_e_l_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitee node L (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_e_l_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Sales excitee dest node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitee_dest_beta.value = \
                sales_e_l_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitee_dest_beta.value = \
                sales_e_l_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitee node L (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_e_l_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Sales excitee dest node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].causal_excitee_dest_weight.value = \
                sales_e_l_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].causal_excitee_dest_weight.value = \
                sales_e_l_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales excitee node L (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_e_l_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Sales source node (spontaneous) zero
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].spontaneous_source_0.value = \
                sales_s_i_zero + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].spontaneous_source_0.value = \
                sales_s_i_zero

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales node I (spontaneous) zero, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_s_i_zero[a],
                    estimated_deriv,
                    places=5
                )

        # Sales source node (spontaneous) one
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].spontaneous_source_1.value = \
                sales_s_i_one + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].spontaneous_source_1.value = \
                sales_s_i_one

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales node I (spontaneous) one, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_s_i_one[a],
                    estimated_deriv,
                    places=5
                )

        # Sales source node (spontaneous) two
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].spontaneous_source_2.value = \
                sales_s_i_two + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].spontaneous_source_2.value = \
                sales_s_i_two

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales node I (spontaneous) two, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_s_i_two[a],
                    estimated_deriv,
                    places=5
                )

        # Sales dest node (spontaneous) zero
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].spontaneous_dest_0.value = \
                sales_s_j_zero + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].spontaneous_dest_0.value = \
                sales_s_j_zero

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales node J (spontaneous) zero, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_s_j_zero[a],
                    estimated_deriv,
                    places=5
                )

        # Sales dest node (spontaneous) one
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].spontaneous_dest_1.value = \
                sales_s_j_one + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].spontaneous_dest_1.value = \
                sales_s_j_one

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales node J (spontaneous) one, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_s_j_one[a],
                    estimated_deriv,
                    places=5
                )

        # Sales dest node (spontaneous) two
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[0].spontaneous_dest_2.value = \
                sales_s_j_two + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[0].spontaneous_dest_2.value = \
                sales_s_j_two

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Sales node J (spontaneous) two, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_sales_s_j_two[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors ---------------------------------------------------------------------------------------

        # Estimate the derivatives
        # Debtors excitor source node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitor_source_alpha.value = \
                debtors_r_i_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitor_source_alpha.value = \
                debtors_r_i_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitor node I (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_r_i_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors excitor source node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitor_source_beta.value = \
                debtors_r_i_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitor_source_beta.value = \
                debtors_r_i_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitor node I (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_r_i_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors excitor source node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitor_source_weight.value = \
                debtors_r_i_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitor_source_weight.value = \
                debtors_r_i_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitor node I (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_r_i_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors excitor dest node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitor_dest_alpha.value = \
                debtors_r_j_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitor_dest_alpha.value = \
                debtors_r_j_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitor node J (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_r_j_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors excitor dest node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitor_dest_beta.value = \
                debtors_r_j_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitor_dest_beta.value = \
                debtors_r_j_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitor node J (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_r_j_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors excitor dest node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitor_dest_weight.value = \
                debtors_r_j_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitor_dest_weight.value = \
                debtors_r_j_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitor node J (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_r_j_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors excitee source node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitee_source_alpha.value = \
                debtors_e_k_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitee_source_alpha.value = \
                debtors_e_k_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitee node K (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_e_k_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors excitee source node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitee_source_beta.value = \
                debtors_e_k_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitee_source_beta.value = \
                debtors_e_k_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitee node K (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_e_k_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors excitee source node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitee_source_weight.value = \
                debtors_e_k_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitee_source_weight.value = \
                debtors_e_k_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitee node K (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_e_k_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors excitee dest node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitee_dest_alpha.value = \
                debtors_e_l_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitee_dest_alpha.value = \
                debtors_e_l_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitee node L (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_e_l_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors excitee dest node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitee_dest_beta.value = \
                debtors_e_l_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitee_dest_beta.value = \
                debtors_e_l_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitee node L (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_e_l_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors excitee dest node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].causal_excitee_dest_weight.value = \
                debtors_e_l_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].causal_excitee_dest_weight.value = \
                debtors_e_l_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors excitee node L (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_e_l_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors source node (spontaneous) zero
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].spontaneous_source_0.value = \
                debtors_s_i_zero + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].spontaneous_source_0.value = \
                debtors_s_i_zero

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors node I (spontaneous) zero, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_s_i_zero[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors source node (spontaneous) one
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].spontaneous_source_1.value = \
                debtors_s_i_one + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].spontaneous_source_1.value = \
                debtors_s_i_one

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors node I (spontaneous) one, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_s_i_one[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors source node (spontaneous) two
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].spontaneous_source_2.value = \
                debtors_s_i_two + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].spontaneous_source_2.value = \
                debtors_s_i_two

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors node I (spontaneous) two, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_s_i_two[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors dest node (spontaneous) zero
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].spontaneous_dest_0.value = \
                debtors_s_j_zero + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].spontaneous_dest_0.value = \
                debtors_s_j_zero

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors node J (spontaneous) zero, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_s_j_zero[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors dest node (spontaneous) one
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].spontaneous_dest_1.value = \
                debtors_s_j_one + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].spontaneous_dest_1.value = \
                debtors_s_j_one

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors node J (spontaneous) one, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_s_j_one[a],
                    estimated_deriv,
                    places=5
                )

        # Debtors dest node (spontaneous) two
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[1].spontaneous_dest_2.value = \
                debtors_s_j_two + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[1].spontaneous_dest_2.value = \
                debtors_s_j_two

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Debtors node J (spontaneous) two, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_debtors_s_j_two[a],
                    estimated_deriv,
                    places=5
                )

        # Bank ---------------------------------------------------------------------------------------

        # Estimate the derivatives
        # Bank excitor source node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitor_source_alpha.value = \
                bank_r_i_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitor_source_alpha.value = \
                bank_r_i_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitor node I (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_r_i_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Bank excitor source node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitor_source_beta.value = \
                bank_r_i_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitor_source_beta.value = \
                bank_r_i_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitor node I (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_r_i_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Bank excitor source node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitor_source_weight.value = \
                bank_r_i_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitor_source_weight.value = \
                bank_r_i_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitor node I (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_r_i_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Bank excitor dest node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitor_dest_alpha.value = \
                bank_r_j_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitor_dest_alpha.value = \
                bank_r_j_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitor node J (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_r_j_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Bank excitor dest node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitor_dest_beta.value = \
                bank_r_j_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitor_dest_beta.value = \
                bank_r_j_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitor node J (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_r_j_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Bank excitor dest node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitor_dest_weight.value = \
                bank_r_j_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitor_dest_weight.value = \
                bank_r_j_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitor node J (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_r_j_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Bank excitee source node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitee_source_alpha.value = \
                bank_e_k_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitee_source_alpha.value = \
                bank_e_k_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitee node K (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_e_k_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Bank excitee source node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitee_source_beta.value = \
                bank_e_k_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitee_source_beta.value = \
                bank_e_k_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitee node K (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_e_k_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Bank excitee source node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitee_source_weight.value = \
                bank_e_k_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitee_source_weight.value = \
                bank_e_k_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitee node K (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_e_k_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Bank excitee dest node (causal) alpha
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitee_dest_alpha.value = \
                bank_e_l_alpha + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitee_dest_alpha.value = \
                bank_e_l_alpha

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitee node L (causal) alpha, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_e_l_alpha[a],
                    estimated_deriv,
                    places=5
                )

        # Bank excitee dest node (causal) beta
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitee_dest_beta.value = \
                bank_e_l_beta + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitee_dest_beta.value = \
                bank_e_l_beta

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitee node L (causal) beta, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_e_l_beta[a],
                    estimated_deriv,
                    places=5
                )

        # Bank excitee dest node (causal) weight
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].causal_excitee_dest_weight.value = \
                bank_e_l_weight + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].causal_excitee_dest_weight.value = \
                bank_e_l_weight

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank excitee node L (causal) weight, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_e_l_weight[a],
                    estimated_deriv,
                    places=5
                )

        # Bank source node (spontaneous) zero
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].spontaneous_source_0.value = \
                bank_s_i_zero + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].spontaneous_source_0.value = \
                bank_s_i_zero

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank node I (spontaneous) zero, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_s_i_zero[a],
                    estimated_deriv,
                    places=5
                )

        # Bank source node (spontaneous) one
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].spontaneous_source_1.value = \
                bank_s_i_one + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].spontaneous_source_1.value = \
                bank_s_i_one

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank node I (spontaneous) one, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_s_i_one[a],
                    estimated_deriv,
                    places=5
                )

        # Bank source node (spontaneous) two
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].spontaneous_source_2.value = \
                bank_s_i_two + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].spontaneous_source_2.value = \
                bank_s_i_two

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank node I (spontaneous) two, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_s_i_two[a],
                    estimated_deriv,
                    places=5
                )

        # Bank dest node (spontaneous) zero
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].spontaneous_dest_0.value = \
                bank_s_j_zero + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].spontaneous_dest_0.value = \
                bank_s_j_zero

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank node J (spontaneous) zero, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_s_j_zero[a],
                    estimated_deriv,
                    places=5
                )

        # Bank dest node (spontaneous) one
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].spontaneous_dest_1.value = \
                bank_s_j_one + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].spontaneous_dest_1.value = \
                bank_s_j_one

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank node J (spontaneous) one, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_s_j_one[a],
                    estimated_deriv,
                    places=5
                )

        # Bank dest node (spontaneous) two
        for a in range(node_dimension):
            # Create a step in the appropriate dimension
            increment = np.eye(1, node_dimension, a)[0]*epsilon

            # Adjust the parameter accordingly
            self.graph.nodes[2].spontaneous_dest_2.value = \
                bank_s_j_two + increment

            # Update the graph excitement based on these parameters
            self.reset_graph(edges)

            # Recalculate the function value
            next_value = \
                np.log(self.graph.edge_probability(
                    i, j, count
                ))

            # Reset the parameter change
            self.graph.nodes[2].spontaneous_dest_2.value = \
                bank_s_j_two

            # Estimate the partial derivative wrt this parameter
            estimated_deriv = (next_value-base_value)/epsilon

            with self.subTest(msg=f'Bank node J (spontaneous) two, dimension {a}'):
                self.assertAlmostEqual(
                    calc_deriv_bank_s_j_two[a],
                    estimated_deriv,
                    places=5
                )

    def test_deriv(self):
        # Define edges
        edges = {}

        # Choose the derivative
        self.derivative_helper(
            0, 1, 2,
            edges=edges
        )

    def test_with_edge(self):
        # Define edges
        edges = {
            0: [(0, 1, 10)]
        }

        # Calculate the derivative
        self.derivative_helper(
            0, 1, 2,
            edges=edges
        )

    def test_with_edge_debtor_settle(self):
        # Define edges
        edges = {
            0: [(1, 2, 0.5)]
        }

        # Calculate the derivative
        self.derivative_helper(
            0, 1, 2,
            edges=edges
        )

    def test_with_edge_cash_sale(self):
        # Define edges
        edges = {
            0: [(0, 1, 10)]
        }

        # Calculate the derivative
        self.derivative_helper(
            0, 2, 3,
            edges=edges
        )

    def test_with_multiple_edges(self):
        # Define edges
        edges = {
            0: [(0, 1, 10)],
            1: [],
            2: [(0, 2, -1), (1, 2, 0.5)],
            3: []
        }

        # Calculate the derivative
        self.derivative_helper(
            0, 1, 2,
            edges=edges
        )
