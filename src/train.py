"""Module for training a Dynamic Accounting Graph
"""

from tqdm import tqdm
import matplotlib.pyplot as plt


def train(graph, edges_by_day, last_day, iterations=1500,
          plot_log_likelihood=True, use_tqdm=True,
          spontaneous_learning_startpoint=500):
    """Train a model maximising log-likelihood for a particular
    set of edges, using simple gradient ascent

    Args:
        graph (DynamicAccountingGraph): The DynamicAccountingGraph object
        edges_by_day (dict): A record of all the edges in the dynamic graph,
            with the day number as the key, and a list of edges each given
            as a tuple, (source_node, dest_node, weight)
        last_day (int): The final day of the period (in case there are days
            at the end of the period with no edges so they don't appear in
            edges_by_day).
        iterations (int, optional): How many epochs to train for.
            Defaults to 1500.
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
            Defaults to 500.

    Returns:
        log_likelihoods (list): A record of the log-likelihood at each epoch.
    """

    # Add the whole period's edges to the graph, calculating the
    # probability of that number of edges each day under the current
    # parameters. Then update the parameters to maximise that
    # probability, using simple gradient ascent.
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

        for day, edges in edges_by_day.items():
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

        # Update the parameters using gradient ascent, and
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
