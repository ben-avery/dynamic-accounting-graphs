from utilities import part_weibull


class Excitement():
    def __init__(self, weibull_weight, weibull_alpha, weibull_beta,
                 lin_val_weight, lin_val_alpha, lin_val_beta,
                 source_nodes, dest_nodes,
                 alive_threshold=0.01):
        self.alive = True
        self.time = 0

        self.source_nodes = source_nodes
        self.dest_nodes = dest_nodes

        self.weibull_weight = weibull_weight
        self.weibull_alpha = weibull_alpha
        self.weibull_beta = weibull_beta

        self.lin_val_weight = lin_val_weight
        self.lin_val_alpha = lin_val_alpha
        self.lin_val_beta = lin_val_beta

        self.prob_parts = (
            self.prob_part(self.time),
            -self.prob_part(self.time+1)
        )
        self.probability = sum(self.prob_parts)

        self.alive_threshold = alive_threshold
        self.remaining_weight = \
            self.weibull_weight - self.probability

    def prob_part(self, time):
        if not self.alive:
            raise ValueError('Excitement is not alive')

        return part_weibull(time, self.weibull_alpha, self.weibull_beta)

    def increment_probability(self):
        if not self.alive:
            raise ValueError('Excitement is not alive')

        self.prob_parts = (
            -self.prob_parts[1],
            -self.prob_part(self.time+1)
        )
        self.probability = sum(self.prob_parts)
        self.remaining_weight = \
            self.weibull_weight - self.probability

    def increment_time(self):
        if not self.alive:
            raise ValueError('Excitement is not alive')

        if self.remaining_weight < self.alive_threshold:
            self.alive = False
            self.probability = None
        else:
            self.time += 1
            self.increment_probability()
