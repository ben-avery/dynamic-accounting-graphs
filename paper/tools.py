"""Helper functions for running experiments and generating figures for the report
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import poisson

from utilities import discrete_weibull_pmf
from train import train


def extract_gradients(graph, edges_by_day, last_day,
                      node_index, node_dimension, embedding_type,
                      overwrite_value=None):
    """Extract the partial derivatives and log-likelihood for a
    particular value of a particular embedding

    Args:
        graph (DynamicAccountingGraph): A graph object
        edges_by_day (dict): A record of all the edges in the dynamic graph,
            with the day number as the key, and a list of edges each given
            as a tuple, (source_node, dest_node, weight)
        last_day (int): The final day of the period (in case there are days
            at the end of the period with no edges so they don't appear in
            edges_by_day).
        node_index (int): The index of the target node
        node_dimension (int): Which dimension of the embedding is to be changed?
        embedding_type (str): A string describing which embedding to choose.
            For example, the embedding relating to the zero coefficient of a
            spontaneous source node would be 's_k_0'.
        overwrite_value (float, optional): The value to overwrite. If left as None,
            the value will be unchanged.
            Defaults to None.

    Returns:
        gradient: The partial derivative of the log likelihood at this value
        log_likelihood: The log likelihood at this value
    """

    # Get the relevant embedding
    if embedding_type == 's_k_0':
        node = graph.nodes[node_index].spontaneous_source_0
    elif embedding_type == 's_l_0':
        node = graph.nodes[node_index].spontaneous_dest_0
    elif embedding_type == 's_k_1':
        node = graph.nodes[node_index].spontaneous_source_1
    elif embedding_type == 's_l_1':
        node = graph.nodes[node_index].spontaneous_dest_1
    elif embedding_type == 's_k_2':
        node = graph.nodes[node_index].spontaneous_source_2
    elif embedding_type == 's_l_2':
        node = graph.nodes[node_index].spontaneous_dest_2
    elif embedding_type == 'r_i_alpha':
        node = graph.nodes[node_index].causal_excitor_source_alpha
    elif embedding_type == 'r_j_alpha':
        node = graph.nodes[node_index].causal_excitor_dest_alpha
    elif embedding_type == 'r_i_beta':
        node = graph.nodes[node_index].causal_excitor_source_beta
    elif embedding_type == 'r_j_beta':
        node = graph.nodes[node_index].causal_excitor_dest_beta
    elif embedding_type == 'r_i_w':
        node = graph.nodes[node_index].causal_excitor_source_weight
    elif embedding_type == 'r_j_w':
        node = graph.nodes[node_index].causal_excitor_dest_weight
    elif embedding_type == 'e_k_alpha':
        node = graph.nodes[node_index].causal_excitee_source_alpha
    elif embedding_type == 'e_l_alpha':
        node = graph.nodes[node_index].causal_excitee_dest_alpha
    elif embedding_type == 'e_k_beta':
        node = graph.nodes[node_index].causal_excitee_source_beta
    elif embedding_type == 'e_l_beta':
        node = graph.nodes[node_index].causal_excitee_dest_beta
    elif embedding_type == 'e_k_w':
        node = graph.nodes[node_index].causal_excitee_source_weight
    elif embedding_type == 'e_l_w':
        node = graph.nodes[node_index].causal_excitee_dest_weight

    # Overwrite value, if required
    if overwrite_value is not None:
        # Save previous value
        previous_value = node.value[node_dimension]

        # Overwrite with new value
        node.value[node_dimension] = overwrite_value

        # Reset the graph
        graph.reset(spontaneous_on=True, discard_gradient_updates=True)

    # Calculate the function value at a particular point.
    # This triggers the calculation of the gradient too.
    log_probability = 0
    for day, edges in edges_by_day.items():
        # Move the graph onto the next day, until a new
        # day that contains some edges
        while graph.time < day:
            # Calculate the probability of the current day
            log_probability += graph.day_log_probability(
                spontaneous_on=True
            )

            # Move the graph onto the next day
            graph.increment_time()

        # Add all of this day's edges to the graph
        for i, j, edge_weight in edges:
            graph.add_edge(i, j, edge_weight)

    # Any days at the end of the period with no edges
    # also need their probabilities calculated
    while graph.time <= last_day:
        # Calculate the probability of the current day
        log_probability += graph.day_log_probability()

        # Move the graph onto the next day
        graph.increment_time()

    # Extract gradients
    gradient = node.pending_updates[node_dimension]

    # Revert the value, if required
    if overwrite_value is not None:
        # Overwrite with previous value
        node.value[node_dimension] = previous_value

    # Update the parameters using gradient ascent, and
    # then remove the edges and excitation ready for
    # the next epoch
    graph.reset(spontaneous_on=True, discard_gradient_updates=True)

    # Return the gradient and the total log probability
    return gradient, log_probability


def plot_gradients_and_probabilities(
        graph, edges_by_day, last_day,
        node_index, node_dimension, embedding_type, possible_values):
    """Plot the partial derivatives and log-likelihoods as a particular
    dimension of a particular embedding is gradually changed.

    Args:
        graph (DynamicAccountingGraph): A graph object
        edges_by_day (dict): A record of all the edges in the dynamic graph,
            with the day number as the key, and a list of edges each given
            as a tuple, (source_node, dest_node, weight)
        last_day (int): The final day of the period (in case there are days
            at the end of the period with no edges so they don't appear in
            edges_by_day).
        node_index (int): The index of the target node
        node_dimension (int): Which dimension of the embedding is to be changed?
        embedding_type (str): A string describing which embedding to choose.
            For example, the embedding relating to the zero coefficient of a
            spontaneous source node would be 's_k_0'.
        possible_values (array): The list/numpy array with the values for the
            embedding to take.
    """
    # Calculate the gradients and log likelihoods at each of the requested points
    gradients_and_probabilities = [
        extract_gradients(
            graph, edges_by_day, last_day,
            node_index, node_dimension, embedding_type,
            overwrite_value=overwrite_value
        )
        for overwrite_value in tqdm(possible_values)
    ]

    # Format the node notation in LaTeX
    x_label = \
        'Node %s, $\overrightarrow{%s}_%s^{(%s)}$' % (str(node_index), embedding_type[0], embedding_type[2], f'\{embedding_type[4:]}' if len(embedding_type) != 5 else embedding_type[-1])

    # Add coordinate axes to the gradient, so zeros can easily be seen
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.plot(
        possible_values,
        [grad_and_prob[0] for grad_and_prob in gradients_and_probabilities]
    )

    plt.xlabel(x_label)
    plt.ylabel('Log-likelihood derivative w.r.t. parameter')
    plt.title('Gradient at different parameter values')

    plt.show()

    # Plot the log likelihoods too
    plt.axvline(0, color='black')

    plt.plot(
        possible_values,
        [grad_and_prob[1] for grad_and_prob in gradients_and_probabilities]
    )
    plt.xlabel(x_label)
    plt.ylabel('Log-likelihood')
    plt.title('Log-likelihood at different parameter values')

    plt.show()


def simulate_simple_data():
    """Simulate some transactions in which sales settle exactly
    five days after occuring for the first sales account, and
    exactly eight days after occuring for the second sales
    account
    """

    # Set up variables to track the age of debtors from the
    # two sources
    debtor_days_1 = 5
    debtor_days_2 = 8
    sales1_in_debtors_by_day = [0 for _ in range(debtor_days_1)]
    sales2_in_debtors_by_day = [0 for _ in range(debtor_days_2)]

    # Set up a variable to store the generated edges
    edges_by_day = {}

    # Simulate a year of transactions
    total_days = 365
    for day in range(total_days):
        # Get overall activity level
        activity_level_1 = np.sin(day*6.28318/total_days)+1.00001
        activity_level_2 = np.sin((day+50)*6.28318/total_days)+1.00001

        # Pick how many new sales will occur this day
        new_sales_1 = np.random.poisson(0.25*activity_level_1)
        new_sales_2 = np.random.poisson(0.1*activity_level_2)

        # Create the edges for one type of sales
        new_edges = [
            (0, 2, 100)
            for _ in range(new_sales_1)
        ]
        # Create the edges for the other type of sales
        new_edges.extend([
            (1, 2, 100)
            for _ in range(new_sales_2)
        ])
        # Settle any debtors from sales1 that have existed for five days
        new_edges.extend([
            (2, 3, 100)
            for _ in range(sales1_in_debtors_by_day[-1])
        ])
        # Settle any debtors from sales2 that have existed for eight days
        new_edges.extend([
            (2, 3, 100)
            for _ in range(sales2_in_debtors_by_day[-1])
        ])

        # Record the day's edges
        edges_by_day[day] = new_edges[:]

        # Update the ages of unpaid debtors
        for day in range(debtor_days_1-1):
            sales1_in_debtors_by_day[debtor_days_1-1-day] = \
                sales1_in_debtors_by_day[debtor_days_1-2-day]
        sales1_in_debtors_by_day[0] = new_sales_1

        for day in range(debtor_days_2-1):
            sales2_in_debtors_by_day[debtor_days_2-1-day] = \
                sales2_in_debtors_by_day[debtor_days_2-2-day]
        sales2_in_debtors_by_day[0] = new_sales_2

    # Also return the accounts in the correct format
    raw_accounts = [
        ('sales1', '7000', 0, 'revenue'),
        ('sales2', '7500', 0, 'revenue'),
        ('debtors', '4000', 0, 'debtors'),
        ('bank', '5500', 0, 'bank')
    ]

    return edges_by_day, total_days - 1, raw_accounts


def simulate_many_nodes(count_sales_nodes=2, debtor_days_by_acc=(5, 8),
                        average_rates_by_acc=(0.25, 0.1), total_days=365):
    """Simulate some transactions in which sales settle an exact
    number of days after they occur.
    """

    # Set up variables to track the age of debtors from the
    # different sales accounts
    sales_in_debtors_by_day = [
        [0 for _ in range(debtor_days)]
        for debtor_days in debtor_days_by_acc
    ]

    # Set up a variable to store the generated edges
    edges_by_day = {}

    # Simulate transactions
    for day in range(total_days):
        # Get overall activity level
        activity_levels = [
            np.sin((day+50*i)*6.28318/total_days)+1.00001
            for i in range(count_sales_nodes)
        ]

        # Pick how many new sales will occur this day
        new_sales = np.random.poisson(
            np.array(average_rates_by_acc)*np.array(activity_levels)
        )

        # Create the edges for the sales
        new_edges = []
        for i in range(count_sales_nodes):
            new_edges.extend([
                (i, count_sales_nodes, 100)
                for _ in range(new_sales[i])
            ])

        # Create any debtor settle edges
        for i in range(count_sales_nodes):
            new_edges.extend([
                (count_sales_nodes, count_sales_nodes + 1, 100)
                for _ in range(sales_in_debtors_by_day[i][-1])
            ])

        # Record the day's edges
        edges_by_day[day] = new_edges[:]

        # Update the ages of unpaid debtors
        for i in range(count_sales_nodes):
            for day in range(debtor_days_by_acc[i]-1):
                sales_in_debtors_by_day[i][debtor_days_by_acc[i]-1-day] = \
                    sales_in_debtors_by_day[i][debtor_days_by_acc[i]-2-day]
            sales_in_debtors_by_day[i][0] = new_sales[i]

    # Create the accounts with names, unique identifiers, starting
    # cumulative balances and an associated accounting concept
    raw_accounts = [
        (f'sales{i}', f'{7000+i}', 0, 'revenue')
        for i in range(count_sales_nodes)
    ] + \
    [
        ('debtors', '4000', 0, 'debtors'),
        ('bank', '5500', 0, 'bank')
    ]

    return edges_by_day, total_days - 1, raw_accounts


def extract_weibull_parameters(i, j, k, l, graph):
    """Get the trained Weibull parameters from a graph
    for a particular edge pair

    Args:
        i (int): Index of the excitor source node
        j (int): Index of the excitor source node
        k (int): Index of the excitor source node
        l (int): Index of the excitor source node
        graph (DynamicAccountingGraph): The trained graph class
    """

    # Get the node embeddings
    x_i = graph.nodes[i].causal_excitor_source_weight.value
    x_j = graph.nodes[j].causal_excitor_dest_weight.value
    x_k = graph.nodes[k].causal_excitee_source_weight.value
    x_l = graph.nodes[l].causal_excitee_dest_weight.value

    # Get the edge embeddings
    e_ij = graph.edge_embedder.embed_edge(x_i, x_j)
    e_kl = graph.edge_embedder.embed_edge(x_k, x_l)

    # Get the Weibull parameters
    weight = \
        graph.causal_comparer_weight.compare_embeddings(
            e_ij, e_kl
        )

    # Get the node embeddings
    x_i = graph.nodes[i].causal_excitor_source_alpha.value
    x_j = graph.nodes[j].causal_excitor_dest_alpha.value
    x_k = graph.nodes[k].causal_excitee_source_alpha.value
    x_l = graph.nodes[l].causal_excitee_dest_alpha.value

    # Get the edge embeddings
    e_ij = graph.edge_embedder.embed_edge(x_i, x_j)
    e_kl = graph.edge_embedder.embed_edge(x_k, x_l)

    alpha = \
        graph.causal_comparer_alpha.compare_embeddings(
            e_ij, e_kl
        )

    # Get the node embeddings
    x_i = graph.nodes[i].causal_excitor_source_beta.value
    x_j = graph.nodes[j].causal_excitor_dest_beta.value
    x_k = graph.nodes[k].causal_excitee_source_beta.value
    x_l = graph.nodes[l].causal_excitee_dest_beta.value

    # Get the edge embeddings
    e_ij = graph.edge_embedder.embed_edge(x_i, x_j)
    e_kl = graph.edge_embedder.embed_edge(x_k, x_l)

    beta = \
        graph.causal_comparer_beta.compare_embeddings(
            e_ij, e_kl
        )

    return weight, alpha, beta


def modal_transit_time(alpha, beta, rounded=False):
    """Get the (modelled) modal time between the excitor edge
    and the excitee edge, given the Weibull parameters

    Args:
        alpha (float): Discrete Weibull alpha parameter
        beta (float): Discrete Weibull beta parameter
    """

    # Return the mode
    if rounded:
        return round((alpha-0.5) * ((beta-1)/beta)**(1/beta))
    else:
        return (alpha-0.5) * ((beta-1)/beta)**(1/beta)


def summarise_trained_excitation(i, j, k, l, graph):
    """Return the key interpretable characteristics of the
    trained Weibull distributions. The weight (expected 
    number of excitees for each excitor) and the modal time
    interval (the most likely temporal gap between excitor
    and excitee)

    Args:
        i (int): Index of the excitor source node
        j (int): Index of the excitor destination node
        k (int): Index of the excitee source node
        l (int): Index of the excitee destination node
        graph (DynamicAccountingGraph): The trained graph class

    Returns:
        weight (float): The expected number of excitees
            for each excitor
        edge_modal_transit_time (float): The most likely
            temporal gap between excitor and excitee
    """

    # Get the Weibull parameters
    weight, alpha, beta = \
        extract_weibull_parameters(
            i=i, j=j,
            k=k, l=l,
            graph=graph
        )

    if beta > 1:
        # Use the approximation for the mode
        # where it's defined
        edge_modal_transit_time = \
            modal_transit_time(
                alpha, beta
            )

    if beta <= 1 or np.isnan(edge_modal_transit_time):
        # If the approximation is undefined, find the
        # mode experimentally

        # Set up variables for tracking the current
        # most likely time
        mode = None
        max_prob = 0
        remaining_prob = 1
        time = 0

        # Iterate until the probability of remaining
        # times is sufficiently small, or at time=500
        # if sooner.
        increment = 1
        while remaining_prob > 0.01 and time < 500:
            # Calculate the probability at the current time
            prob = discrete_weibull_pmf(time, alpha, beta)

            # If this is higher than the previous highest
            # observation, record it as the new highest
            # observation
            if prob > max_prob:
                mode = time
                max_prob = prob

            # Update the probability of remaining times
            if time - int(time) < increment/2:
                remaining_prob -= prob

            # Increment the time
            time += increment

        # The highest observed time is the mode
        edge_modal_transit_time = mode

        if remaining_prob > 0.01:
            print('Mode had remaining probability',
                  remaining_prob, alpha, beta)

    # Since excitation starts from the day after the
    # excitor, the transit time starts from 1.
    return weight, edge_modal_transit_time + 1


def summarise_trained_baseline(i, j, graph):
    """Return the three parameters of spontaneous intensity for
    a given edge

    Args:
        i (int): Index of the source node
        j (int): Index of the destination node
        graph (DynamicAccountingGraph): The trained graph class

    Returns:
        param_zero: The intercept parameter
        param_one: The source node balance parameter
        param_two: The destination node balance parameter
    """

    # Get the embeddings
    y_i = graph.nodes[i].spontaneous_source_0.value
    y_j = graph.nodes[j].spontaneous_dest_0.value

    # Calculate the linear output
    param_zero = \
        graph.spontaneous_comparer.compare_embeddings(
            y_i, y_j
        )

    # Get the embeddings
    y_i = graph.nodes[i].spontaneous_source_1.value
    y_j = graph.nodes[j].spontaneous_dest_1.value

    # Calculate the linear output
    param_one = \
        graph.spontaneous_comparer.compare_embeddings(
            y_i, y_j
        )

    # Get the embeddings
    y_i = graph.nodes[i].spontaneous_source_2.value
    y_j = graph.nodes[j].spontaneous_dest_2.value

    # Calculate the linear output
    param_two = \
        graph.spontaneous_comparer.compare_embeddings(
            y_i, y_j
        )

    return param_zero, param_one, param_two


def train_recording_params(graph, edges_by_day, last_day=None,
                           training_epochs=1500, extraction_interval=1,
                           start_spontaneous_from=500):
    """Run the training algorithm on the Synthetic Data in short bursts,
    saving key information about the parameters
    after each burst

    Args:
        graph (DynamicAccountingGraph): The graph object
        edges_by_day (dict): The list of all edges in the
            period (keys are the day numbers, and values
            are list of tuples, where each tuple is
            (source_node, dest_node, weight))

    Returns:
        all_log_likelihoods (list): The log-likelihood at each epoch
        weights_chain1 (list): The weight parameter for the
            edge pair Sales1->Debtors and Debtors->Bank at
            every 10th epoch
        times_chain1 (list): The modal time interval for the
            edge pair Sales1->Debtors and Debtors->Bank at
            every 10th epoch
        weights_chain2 (list): The weight parameter for the
            edge pair Sales2->Debtors and Debtors->Bank at
            every 10th epoch
        times_chain2 (list): The modal time interval for the
            edge pair Sales2->Debtors and Debtors->Bank at
            every 10th epoch
    """

    # Extract the last date in the period (in case there
    # were any days at the end of the period with no edges)
    if last_day is None:
        last_day = max(int(day) for day in edges_by_day.keys())

    # Run the training algorithm
    all_log_likelihoods = []
    times = {
        ('NewSale1', 'DebtorSettle'): [],
        ('NewSale2', 'DebtorSettle'): [],
        ('NewSale1', 'NewSale1'): [],
        ('NewSale2', 'NewSale1'): [],
        ('NewSale1', 'NewSale2'): [],
        ('NewSale2', 'NewSale2'): [],
        ('DebtorSettle', 'NewSale1'): [],
        ('DebtorSettle', 'NewSale2'): [],
        ('DebtorSettle', 'DebtorSettle'): []
    }
    weights = {
        ('NewSale1', 'DebtorSettle'): [],
        ('NewSale2', 'DebtorSettle'): [],
        ('NewSale1', 'NewSale1'): [],
        ('NewSale2', 'NewSale1'): [],
        ('NewSale1', 'NewSale2'): [],
        ('NewSale2', 'NewSale2'): [],
        ('DebtorSettle', 'NewSale1'): [],
        ('DebtorSettle', 'NewSale2'): [],
        ('DebtorSettle', 'DebtorSettle'): []
    }
    baselines = {
        0: {
            'NewSale1': [],
            'NewSale2': [],
            'DebtorSettle': []},
        1: {
            'NewSale1': [],
            'NewSale2': [],
            'DebtorSettle': []},
        2: {
            'NewSale1': [],
            'NewSale2': [],
            'DebtorSettle': []}
    }

    for repeat in tqdm(range(int((training_epochs)/extraction_interval))):
        # Pass the full set of transactions through the graph
        # another set of times
        if repeat < start_spontaneous_from:
            spontaneous_learning_startpoint = extraction_interval
        else:
            spontaneous_learning_startpoint = 0

        log_likelihoods = train(
            graph, edges_by_day, last_day,
            iterations=extraction_interval,
            plot_log_likelihood=False, use_tqdm=False,
            spontaneous_learning_startpoint=spontaneous_learning_startpoint
        )

        # Record the log_likelihoods
        all_log_likelihoods.extend(log_likelihoods)

        # Record the weights and modal transit times
        weight, edge_modal_transit_time = \
            summarise_trained_excitation(
                i=0, j=2, k=2, l=3, graph=graph)
        weights[('NewSale1', 'DebtorSettle')].append(weight)
        times[('NewSale1', 'DebtorSettle')].append(edge_modal_transit_time)

        weight, edge_modal_transit_time = \
            summarise_trained_excitation(
                i=1, j=2, k=2, l=3, graph=graph)
        weights[('NewSale2', 'DebtorSettle')].append(weight)
        times[('NewSale2', 'DebtorSettle')].append(edge_modal_transit_time)

        weight, edge_modal_transit_time = \
            summarise_trained_excitation(
                i=0, j=2, k=0, l=2, graph=graph)
        weights[('NewSale1', 'NewSale1')].append(weight)
        times[('NewSale1', 'NewSale1')].append(edge_modal_transit_time)

        weight, edge_modal_transit_time = \
            summarise_trained_excitation(
                i=1, j=2, k=0, l=2, graph=graph)
        weights[('NewSale2', 'NewSale1')].append(weight)
        times[('NewSale2', 'NewSale1')].append(edge_modal_transit_time)

        weight, edge_modal_transit_time = \
            summarise_trained_excitation(
                i=0, j=2, k=1, l=2, graph=graph)
        weights[('NewSale1', 'NewSale2')].append(weight)
        times[('NewSale1', 'NewSale2')].append(edge_modal_transit_time)

        weight, edge_modal_transit_time = \
            summarise_trained_excitation(
                i=1, j=2, k=1, l=2, graph=graph)
        weights[('NewSale2', 'NewSale2')].append(weight)
        times[('NewSale2', 'NewSale2')].append(edge_modal_transit_time)

        weight, edge_modal_transit_time = \
            summarise_trained_excitation(
                i=2, j=3, k=0, l=2, graph=graph)
        weights[('DebtorSettle', 'NewSale1')].append(weight)
        times[('DebtorSettle', 'NewSale1')].append(edge_modal_transit_time)

        weight, edge_modal_transit_time = \
            summarise_trained_excitation(
                i=2, j=3, k=1, l=2, graph=graph)
        weights[('DebtorSettle', 'NewSale2')].append(weight)
        times[('DebtorSettle', 'NewSale2')].append(edge_modal_transit_time)

        weight, edge_modal_transit_time = \
            summarise_trained_excitation(
                i=1, j=2, k=1, l=2, graph=graph)
        weights[('DebtorSettle', 'DebtorSettle')].append(weight)
        times[('DebtorSettle', 'DebtorSettle')].append(edge_modal_transit_time)

        # Record baseline intensity parameters
        edge = 'NewSale1'
        i, j = 0, 2
        param_zero, param_one, param_two = \
            summarise_trained_baseline(i, j, graph)
        baselines[0][edge].append(param_zero)
        baselines[1][edge].append(param_one)
        baselines[2][edge].append(param_two)

        edge = 'NewSale2'
        i, j = 1, 2
        param_zero, param_one, param_two = \
            summarise_trained_baseline(i, j, graph)
        baselines[0][edge].append(param_zero)
        baselines[1][edge].append(param_one)
        baselines[2][edge].append(param_two)

        edge = 'DebtorSettle'
        i, j = 2, 3
        param_zero, param_one, param_two = \
            summarise_trained_baseline(i, j, graph)
        baselines[0][edge].append(param_zero)
        baselines[1][edge].append(param_one)
        baselines[2][edge].append(param_two)

    return (
        all_log_likelihoods, times, weights, baselines
    )


def plot_training_params(
        log_likelihoods, times, weights, baselines,
        training_epochs=1500, extraction_interval=1):
    """Plot the results of train_recording_params

    Args:
        log_likelihoods (list): The log-likelihood at each epoch
        times (dict): Dictionary of the interval times
            with edge pairs as keys
        weights (dict): Dictionary of the weights with
            edge pairs as keys
    """

    # Time coordinate for log_likelihoods
    time = np.arange(training_epochs)

    # Time coordinate for the semantic characteristics
    weibull_time = \
        extraction_interval * np.arange(
            1, int((training_epochs) / extraction_interval) + 1
            )

    # Log-likelihood --------------------------------------
    # Get subplots
    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(8, 2)
    fig.set_tight_layout(True)

    # Plot log_likelihoods
    axes.plot(time, log_likelihoods)
    axes.set_xlabel('')
    axes.set_ylabel('Log likelihood')
    axes.set_title('Training Convergence')
    axes.set_xlim((0, training_epochs))
    axes.set_ylim((-1500, 0))
    axes.set_yticks(list(range(-1500, 250, 250)))

    # Display the plot
    plt.show()

    # Causal intensity --------------------------------------
    # Get subplots
    fig, axes = plt.subplots(3, 1)
    fig.set_size_inches(8, 6)
    fig.set_tight_layout(True)

    # Plot weights
    axes[0].plot(
        weibull_time, weights[('NewSale1', 'DebtorSettle')],
        label='NewSale1->DebtorSettle')
    axes[0].plot(
        weibull_time, weights[('NewSale2', 'DebtorSettle')],
        label='NewSale2->DebtorSettle')
    axes[0].plot(
        weibull_time, weights[('NewSale1', 'NewSale1')],
        label='NewSale1->NewSale1')
    axes[0].plot(
        weibull_time, weights[('NewSale2', 'NewSale1')],
        label='NewSale2->NewSale1')
    axes[0].plot(
        weibull_time, weights[('NewSale1', 'NewSale2')],
        label='NewSale1->NewSale2')
    axes[0].plot(
        weibull_time, weights[('NewSale2', 'NewSale2')],
        label='NewSale2->NewSale2')
    axes[0].plot(
        weibull_time, weights[('DebtorSettle', 'NewSale1')],
        label='DebtorSettle->NewSale1')
    axes[0].plot(
        weibull_time, weights[('DebtorSettle', 'NewSale2')],
        label='DebtorSettle->NewSale2')
    axes[0].plot(
        weibull_time, weights[('DebtorSettle', 'DebtorSettle')],
        label='DebtorSettle->DebtorSettle')

    axes[0].axhline(
        y=1, label='NewSale->DebtorSettle True',
        c='black', linestyle='--')

    axes[0].set_xlabel('')
    axes[0].set_ylabel('Weight')
    axes[0].set_title('Causal Behaviour During Training')
    axes[0].set_xlim((0, training_epochs))

    # Plot interval times
    axes[1].plot(
        weibull_time, times[('NewSale1', 'DebtorSettle')],
        label='NewSale1->DebtorSettle')
    axes[1].plot(
        weibull_time, times[('NewSale2', 'DebtorSettle')],
        label='NewSale2->DebtorSettle')
    axes[1].plot(
        weibull_time, times[('NewSale1', 'NewSale1')],
        label='NewSale1->NewSale1')
    axes[1].plot(
        weibull_time, times[('NewSale2', 'NewSale1')],
        label='NewSale2->NewSale1')
    axes[1].plot(
        weibull_time, times[('NewSale1', 'NewSale2')],
        label='NewSale1->NewSale2')
    axes[1].plot(
        weibull_time, times[('NewSale2', 'NewSale2')],
        label='NewSale2->NewSale2')
    axes[1].plot(
        weibull_time, times[('DebtorSettle', 'NewSale1')],
        label='DebtorSettle->NewSale1')
    axes[1].plot(
        weibull_time, times[('DebtorSettle', 'NewSale2')],
        label='DebtorSettle->NewSale2')
    axes[1].plot(
        weibull_time, times[('DebtorSettle', 'DebtorSettle')],
        label='DebtorSettle->DebtorSettle')

    axes[1].axhline(y=5, label='NewSale1->DebtorSettle True',
                    c='blue', linestyle='--')
    axes[1].axhline(y=8, label='NewSale2->DebtorSettle True',
                    c='orange', linestyle='--')

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Interval time')
    axes[1].set_ylim((-0.05, 12.0))
    axes[1].set_yticks(list(range(0,14,2)))
    axes[1].set_title('')
    axes[1].set_xlim((0, training_epochs))

    # Empty plot for legend
    zero = [0]
    axes[2].plot(
        zero, zero, label='NewSale1 -> DebtorSettle')
    axes[2].plot(
        zero, zero, label='NewSale2 -> DebtorSettle')
    axes[2].plot(
        zero, zero, label='NewSale1 -> NewSale1')
    axes[2].plot(
        zero, zero, label='NewSale2 -> NewSale1')
    axes[2].plot(
        zero, zero, label='NewSale1 -> NewSale2')
    axes[2].plot(
        zero, zero, label='NewSale2 -> NewSale2')
    axes[2].plot(
        zero, zero, label='DebtorSettle -> NewSale1')
    axes[2].plot(
        zero, zero, label='DebtorSettle -> NewSale2')
    axes[2].plot(
        zero, zero, label='DebtorSettle -> DebtorSettle')
    axes[2].plot(
        zero, zero, label='NewSale1/2 -> DebtorSettle true weight',
        c='black', linestyle='--')
    axes[2].plot(
        zero, zero, label='NewSale1 -> DebtorSettle true interval time',
        c='blue', linestyle='--')
    axes[2].plot(
        zero, zero, label='NewSale2 -> DebtorSettle true interval time',
        c='orange', linestyle='--')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')
    axes[2].set_title('')
    axes[2].legend(loc='center', ncol=2)
    axes[2].set_xlim((0, training_epochs))
    axes[2].axis('off')

    # Display the plot
    plt.show()

    # Spontaneous intensity --------------------------------------
    # Get subplots
    fig, axes = plt.subplots(3, 1)
    fig.set_size_inches(8, 6)
    fig.set_tight_layout(True)

    # Plot param_0
    axes[0].plot(weibull_time, baselines[0]['NewSale1'],
                 label='NewSale1')
    axes[0].plot(weibull_time, baselines[0]['NewSale2'],
                 label='NewSale2')
    axes[0].plot(weibull_time, baselines[0]['DebtorSettle'],
                 label='DebtorSettle')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('$a_0$')
    axes[0].set_title('Spontaneous Behaviour During Training')
    axes[0].legend(loc='lower left')
    axes[0].set_xlim((0, training_epochs))

    # Plot param_1
    axes[1].plot(weibull_time, baselines[1]['NewSale1'],
                 label='NewSale1')
    axes[1].plot(weibull_time, baselines[1]['NewSale2'],
                 label='NewSale2')
    axes[1].plot(weibull_time, baselines[1]['DebtorSettle'],
                 label='DebtorSettle')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('$a_1$')
    axes[1].set_title('')
    axes[1].legend(loc='lower left')
    axes[1].set_xlim((0, training_epochs))

    # Plot param_2
    axes[2].plot(weibull_time, baselines[2]['NewSale1'],
                 label='NewSale1')
    axes[2].plot(weibull_time, baselines[2]['NewSale2'],
                 label='NewSale2')
    axes[2].plot(weibull_time, baselines[2]['DebtorSettle'],
                 label='DebtorSettle')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('$a_2$')
    axes[2].set_title('')
    axes[2].legend(loc='lower left')
    axes[2].set_xlim((0, training_epochs))

    # Display the plot
    plt.show()


def return_probabilities_of_added_edges(graph, edges_by_day):
    """Find the probability that the number of a particular edge type on a particular
    day is higher than expected (suggesting edges have been added), for each each
    day and each edge that occurred at least once on that day.

    Args:
        graph (DynamicAccountingGraph): A graph object
        edges_by_day (dict): A record of all the edges in the dynamic graph,
            with the day number as the key, and a list of edges each given
            as a tuple, (source_node, dest_node, weight)

    Returns:
        list: A list of tuples of the form (time, i, j, probability), where
            time is the day that the edge occurred, (i, j) is the edge, and
            probability is the likelihood that some of these edges are not
            genuine.
    """

    # Reset graph
    graph.reset(spontaneous_on=True, discard_gradient_updates=True)

    all_edges = []
    added_edges = False
    # Work through each edge on each day, adding it to the graph
    for day in tqdm(sorted(list(edges_by_day.keys()))):
        edges = edges_by_day[day]
        # Move the graph onto the next day, until a new
        # day that contains some edges
        while graph.time < day:
            # Calculate the probabilities of any new edges
            if added_edges:
                for i in range(graph.count_nodes):
                    for j in range(graph.count_nodes):
                        if i == j:
                            # Don't allow self-loops
                            continue

                        # Count how many times (if any) that
                        # edge has occurred
                        if (i, j) in graph.new_edges:
                            count = graph.new_edges[(i, j)]

                            # Calculate the probability that fewer edges
                            # should have occurred (i.e. if edges were added)
                            probability = poisson.cdf(
                                k=count - 1,
                                mu=graph.edge_intensity(
                                    i, j, spontaneous_on=True)
                            )

                            # Record the result
                            all_edges.append(
                                (graph.time, i, j, probability)
                            )
                # Reset flag
                added_edges = False

            # Move the graph onto the next day
            graph.increment_time()

        # Add all of this day's edges to the graph
        for i, j, edge_weight in edges:
            graph.add_edge(i, j, edge_weight)

            if not added_edges:
                # Record that new edges have been added, so
                # the probabilities will have to be calculated
                added_edges = True

    # Reset the graph
    graph.reset(spontaneous_on=True, discard_gradient_updates=True)

    return all_edges


def return_probabilities_of_missing_edges(graph, edges_by_day, last_day):
    """Find the probability that the number of a particular edge type on a particular
    day is lower than expected (suggesting edges have been removed/omitted), for each each
    day and each edge that occurred at least once on that day.

    Args:
        graph (DynamicAccountingGraph): A graph object
        edges_by_day (dict): A record of all the edges in the dynamic graph,
            with the day number as the key, and a list of edges each given
            as a tuple, (source_node, dest_node, weight)
        last_day (int): The final day of the period (in case there are days
            at the end of the period with no edges so they don't appear in
            edges_by_day).

    Returns:
        list: A list of tuples of the form (time, i, j, probability), where
            time is the day that the edge occurred, (i, j) is the edge, and
            probability is the likelihood that some of these edges are not
            genuine.
    """

    # Reset graph
    graph.reset(spontaneous_on=True, discard_gradient_updates=True)

    all_edges = []
    # Work through each edge on each day, adding it to the graph
    for day in tqdm(sorted(list(edges_by_day.keys()))):
        edges = edges_by_day[day]
        # Move the graph onto the next day, until a new
        # day that contains some edges
        while graph.time < day:
            # Calculate the probabilities of possible edges
            for i in range(graph.count_nodes):
                for j in range(graph.count_nodes):
                    if i == j:
                        # Don't allow self-loops
                        continue

                    # Count how many times (if any) that
                    # edge has occurred
                    if (i, j) in graph.new_edges:
                        count = graph.new_edges[(i, j)]
                    else:
                        count = 0

                    # Calculate the probability that more edges
                    # should have occurred (i.e. if edges were removed)
                    edge_intensity = \
                        graph.edge_intensity(
                            i, j, spontaneous_on=True
                    )
                    baseline_intensity = \
                        graph.edge_baseline(
                            i, j
                        )
                    causal_intensity = \
                        edge_intensity - baseline_intensity
                    probability = 1 - poisson.cdf(
                        k=count,
                        mu=edge_intensity
                    )

                    # Record the results
                    all_edges.append(
                        (graph.time, i, j, probability,
                         count, edge_intensity,
                         baseline_intensity, causal_intensity)
                    )

            # Move the graph onto the next day
            graph.increment_time()

        # Add all of this day's edges to the graph
        for i, j, edge_weight in edges:
            graph.add_edge(i, j, edge_weight)

    # Any days at the end of the period with no edges
    # also need their probabilities calculated
    while graph.time <= last_day:
        # Calculate the probabilities of possible edges
        for i in range(graph.count_nodes):
            for j in range(graph.count_nodes):
                if i == j:
                    # Don't allow self-loops
                    continue

                # Count how many times (if any) that
                # edge has occurred
                if (i, j) in graph.new_edges:
                    count = graph.new_edges[(i, j)]
                else:
                    count = 0

                # Calculate the probability that more edges
                # should have occurred (i.e. if edges were removed)
                edge_intensity = \
                    graph.edge_intensity(
                        i, j, spontaneous_on=True
                )
                baseline_intensity = \
                    graph.edge_baseline(
                        i, j
                    )
                causal_intensity = \
                    edge_intensity - baseline_intensity
                probability = 1 - poisson.cdf(
                    k=count,
                    mu=edge_intensity
                )

                # Record the results
                all_edges.append(
                    (graph.time, i, j, probability,
                        count, edge_intensity,
                        baseline_intensity, causal_intensity)
                )

        # Move the graph onto the next day
        graph.increment_time()

    # Reset the graph
    graph.reset(spontaneous_on=True, discard_gradient_updates=True)

    return all_edges


def return_probabilities_of_edges(graph, edges_by_day, last_day):
    """Analyse the behaviour of a trained graph, by extracting the
    probabilities of each edge, with the relevant components of the
    intensity.

    Args:
        graph (DynamicAccountingGraph): A graph object
        edges_by_day (dict): A record of all the edges in the dynamic graph,
            with the day number as the key, and a list of edges each given
            as a tuple, (source_node, dest_node, weight)
        last_day (int): The final day of the period (in case there are days
            at the end of the period with no edges so they don't appear in
            edges_by_day).

    Returns:
        list: A list of tuples of the form:
                (time, i, j, probability, total_intensity, spontaneous_intensity, causal_intensity)
            Where:
             - time is the day that the edge occurred
             - (i, j) is the edge
             - probability is the probability of this edge occurring the
               number of times observed in the dataset
             - total_intensity is the intensity of this edge on this day
             - spontaneous_intensity is the spontaneous component of the intensity
             - causal_intensity is the causal component of the intensity
    """
    # Reset graph
    graph.reset(spontaneous_on=True, discard_gradient_updates=True)

    all_edges = []
    # Work through each edge on each day, adding it to the graph
    for day in tqdm(sorted(list(edges_by_day.keys()))):
        edges = edges_by_day[day]
        # Move the graph onto the next day, until a new
        # day that contains some edges
        while graph.time < day:
            # Calculate the probabilities of possible edges
            for i in range(graph.count_nodes):
                for j in range(graph.count_nodes):
                    if i == j:
                        # Don't allow self-loops
                        continue

                    # Count how many times (if any) that
                    # edge has occurred
                    if (i, j) in graph.new_edges:
                        count = graph.new_edges[(i, j)]
                    else:
                        count = 0

                    # Calculate the probability of this
                    # edge, with the relevant components of
                    # intensity
                    total_intensity = \
                        graph.edge_intensity(
                            i, j, spontaneous_on=True)
                    spontaneous_intensity = \
                        graph.edge_baseline(i, j)
                    causal_intensity = \
                        total_intensity - spontaneous_intensity
                    probability = poisson.pmf(
                        k=count,
                        mu=graph.edge_intensity(
                            i, j, spontaneous_on=True)
                    )

                    # Record the results
                    all_edges.append(
                        (graph.time, i, j, count, probability, total_intensity, spontaneous_intensity, causal_intensity)
                    )

            # Move the graph onto the next day
            graph.increment_time()

        # Add all of this day's edges to the graph
        for i, j, edge_weight in edges:
            graph.add_edge(i, j, edge_weight)

    # Any days at the end of the period with no edges
    # also need their probabilities calculated
    while graph.time <= last_day:
        # Calculate the probabilities of possible edges
        for i in range(graph.count_nodes):
            for j in range(graph.count_nodes):
                if i == j:
                    # Don't allow self-loops
                    continue

                # Count how many times (if any) that
                # edge has occurred
                if (i, j) in graph.new_edges:
                    count = graph.new_edges[(i, j)]
                else:
                    count = 0

                # Calculate the probability of this
                # edge, with the relevant components of
                # intensity
                total_intensity = \
                    graph.edge_intensity(
                        i, j, spontaneous_on=False)
                spontaneous_intensity = \
                    graph.edge_baseline(i, j)
                causal_intensity = \
                    total_intensity - spontaneous_intensity
                probability = poisson.pmf(
                    k=count,
                    mu=total_intensity
                )

                # Record the results
                all_edges.append(
                    (graph.time, i, j, count, probability, total_intensity, spontaneous_intensity, causal_intensity)
                )

        # Move the graph onto the next day
        graph.increment_time()

    # Reset the graph
    graph.reset(spontaneous_on=True, discard_gradient_updates=True)

    return all_edges
