"""Module for training a Dynamic Accounting Graph
"""

from tqdm import tqdm
import matplotlib.pyplot as plt

from initialise_params import find_average_balances, find_average_initial_weight
from graphs import DynamicAccountingGraph


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


def generate_graph(raw_accounts, edges_by_day, last_day, node_dimension,
                   graph_kwords={}):
    """Create a graph with the appropriate accounts, initialisation
    and hyperparameters

    Args:
        accounts (list): List of tuples, (name, number, balance, mapping)
        edges_by_day (dict): A record of all the edges in the dynamic graph,
            with the day number as the key, and a list of edges each given
            as a tuple, (source_node, dest_node, weight)
        last_day (int): The final day of the period (in case there are days
            at the end of the period with no edges so they don't appear in
            edges_by_day).
        node_dimension (int): The dimension for the node embeddings
        graph_kwords (dict, optional): Any keyword arguments to set the graph
            hyperparameters. Defaults to {}.
    """
    # Create the accounts
    accounts = [
        Account(name, number, balance, mapping)
        for name, number, balance, mapping in raw_accounts
    ]

    # Find the daily average closing balance for each node
    average_balances = find_average_balances(
        opening_balances=[account_details[2] for account_details in raw_accounts],
        edges_by_day=edges_by_day,
        last_day=last_day
    )

    # Find the average excitation for any edge based on the initialised
    # values of the causal parameters
    average_weight = find_average_initial_weight(
        edges_by_day=edges_by_day,
        last_day=last_day,
        initial_alpha=8,
        intial_beta=4
    )

    # Collate all the types of edges which occur in the year
    possible_edges = set()
    for day in sorted(list(edges_by_day.keys())):
        edges = edges_by_day[day]
        for i, j, value in edges:
            possible_edges.add((i, j))

    # Create and return the DynamicAccountingGraph object
    return DynamicAccountingGraph(
        **graph_kwords,
        accounts=accounts,
        node_dimension=node_dimension,
        average_balances=average_balances,
        average_weight=average_weight,
        possible_edges=possible_edges
    )


def train(graph, edges_by_day, last_day, iterations=300,
          plot_log_likelihood=True, use_tqdm=True,
          spontaneous_learning_startpoint=100):
    """Train a model maximising log-likelihood for a particular
    dataset, using the Adam algorithm

    Args:
        graph (DynamicAccountingGraph): The DynamicAccountingGraph object
        edges_by_day (dict): A record of all the edges in the dynamic graph,
            with the day number as the key, and a list of edges each given
            as a tuple, (source_node, dest_node, weight)
        last_day (int): The final day of the period (in case there are days
            at the end of the period with no edges so they don't appear in
            edges_by_day).
        iterations (int, optional): How many epochs to train for.
            Defaults to 300.
        plot_log_likelihood (bool, optional): Whether to plot the log-likelihood
            over the training run.
            Defaults to True.
        use_tqdm (bool, optional): Whether to print the progress bar during
            training using the tqdm library.
            Defaults to True.
        spontaneous_learning_startpoint (int, optional): Which epoch should the
            spontaneous part of the model be activated? The recommendation is to
            have this part of the model disabled for the start of the learning
            process, to encourage the causal part to dominate.
            Defaults to 100.

    Returns:
        log_likelihoods (list): A record of the log-likelihood at each epoch.
    """

    # Add the whole period's edges to the graph, calculating the
    # probability of that number of edges each day under the current
    # parameters. Then update the parameters to maximise that
    # probability, using a gradient-based algorithm.
    log_likelihoods = [0.0 for _ in range(iterations)]
    if use_tqdm:
        iterator = tqdm(range(iterations))
    else:
        iterator = range(iterations)

    spontaneous_on = False
    for iteration in iterator:
        log_probability = 0

        # Activate the spontaneous part of the model at the requested epoch number
        if not spontaneous_on and iteration >= spontaneous_learning_startpoint:
            spontaneous_on = True

        for day in sorted(list(edges_by_day.keys())):
            edges = edges_by_day[day]
            # Move the graph onto the next day, until a new
            # day that contains some edges
            while graph.time < day:
                # Calculate the probability of the current day
                log_probability += graph.day_log_probability(
                    spontaneous_on=spontaneous_on
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

        # Record the total log likelihood for this epoch
        log_likelihoods[iteration] = log_probability

        # Update the parameters using the gradient algorithm, and
        # then remove the edges and excitation ready for
        # the next epoch
        graph.reset(spontaneous_on=spontaneous_on)

    if plot_log_likelihood:
        # Plot log likelihood
        plt.plot(log_likelihoods)
        plt.xlabel('Epoch')
        plt.ylabel('Log likelihood')
        plt.title('Training results')
        plt.show()

    return log_likelihoods
