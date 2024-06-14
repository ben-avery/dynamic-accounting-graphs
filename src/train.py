from tqdm import tqdm
import matplotlib.pyplot as plt


def train(graph, edges_by_day, last_day):
    iterations = 1000

    losses = [0.0 for _ in range(iterations)]
    for iteration in tqdm(range(iterations)):
        log_probability = 0

        for day, edges in edges_by_day.items():
            while graph.time < day:
                log_probability += graph.day_log_probability()
                graph.increment_time()

            for i, j, edge_weight in edges:
                graph.add_edge(i, j, day, edge_weight)

        while graph.time <= last_day:
            log_probability += graph.day_log_probability()
            graph.increment_time()

        losses[iteration] = log_probability
        graph.reset()

    # Plot losses
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    from graphs import DynamicAccountingGraph

    class Account:
        def __init__(self, name, number, balance, mapping):
            self.name = name
            self.number = number
            self.balance = balance
            self.mapping = mapping

    sales_account = Account('sales', '7000', 0, 'revenue')
    debtors_account = Account('debtors', '4000', 0, 'debtors')
    bank_account = Account('bank', '5500', 0, 'bank')

    graph = DynamicAccountingGraph(
        accounts=[sales_account, debtors_account, bank_account],
        node_dimension=16,
        learning_rate=0.0001
    )

    edges_by_day = {
        0: [
            (0, 1, 100),
            (0, 1, 100)
        ],
        1: [
            (0, 1, 100),
        ],
        2: [
            (1, 2, 100),
            (0, 1, 100),
            (1, 2, 100)
        ],
        3: [
            (0, 1, 100),
            (0, 1, 100),
            (0, 1, 100),
            (1, 2, 100)
        ],
        4: [
            (0, 1, 100),
            (1, 2, 100)
        ]
    }

    last_day = 4

    train(graph, edges_by_day, last_day)
