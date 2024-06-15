"""Module for training a Dynamic Accounting Graph
"""

from tqdm import tqdm
import matplotlib.pyplot as plt


def train(graph, edges_by_day, last_day, iterations=10000,
          plot_log_likelihood=True, use_tqdm=True):
    """Train a model maximising log-likelihood for a particular
    set of edges, using simple gradient ascent
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
    for iteration in iterator:
        log_probability = 0

        for day, edges in edges_by_day.items():
            # Move the graph onto the next day, until a new
            # day that contains some edges
            while graph.time < day:
                # Calculate the probability of the current day
                log_probability += graph.day_log_probability()

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
        graph.reset()

    if plot_log_likelihood:
        # Plot log likelihood
        plt.plot(log_likelihoods)
        plt.xlabel('Epoch')
        plt.ylabel('Log likelihood')
        plt.title('Training results')
        plt.show()

    return log_likelihoods
