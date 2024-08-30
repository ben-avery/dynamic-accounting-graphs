"""Module for the Excitation class.
Terminology:
 - Excitor (noun) is an edge that has the potential to excite a later edge
 - Excitee (noun) is the future edge that has been excited
 - Excitation (noun) is the state where a particular excitor edge
    has occured, and has the potential to excite a hypothetical future
    excitee edge
 - Excite (verb) is the action in which an excitor succeeds in causing an
    excitee edge to occur
"""
from utilities import part_weibull


class Excitation():
    """A class to capture the relationship between an excitor edge
    that has already occured, and a hypothetical, future excitee
    edge.
    """
    def __init__(self, weibull_weight, weibull_alpha, weibull_beta,
                 lin_val_weight, lin_val_alpha, lin_val_beta,
                 excitor_nodes, excitee_nodes,
                 alive_threshold=0.0001):
        """Initialise the class

        Args:
            weibull_weight (float): The total probability of the
                excitation being realised.
            weibull_alpha (float): The alpha parameter of the
                discrete Weibull distribution, reflecting the
                average time between excitor and excitee edges.
            weibull_beta (float): The beta parameter of the
                discrete Weibull distribution, reflecting the
                spread of reasonable times between excitor and
                excitee edges.
            lin_val_weight (float): The raw value from the linear
                function used to generate the Weibull weight,
                before the transformation to a certainly positive
                number with a smooth, continuous function.
            lin_val_alpha (float): The raw value from the linear
                function used to generate the Weibull alpha parameter,
                before the transformation to a certainly positive
                number with a smooth, continuous function.
            lin_val_beta (float): The raw value from the linear
                function used to generate the Weibull beta parameter,
                before the transformation to a certainly positive
                number with a smooth, continuous function.
            excitor_nodes (tuple): Node indices, (i,j), for the
                excitor edge
            excitee_nodes (tuple): Node indices, (k,l), for the
                excitee edge
            alive_threshold (float, optional): The threshold below
                which an Excitation is considered to be dead.
                Defaults to 0.0001.
        """

        # The excitation starts off alive
        self.alive = True

        # The excitation has its own time, relative to when the
        # excitor edge occured
        self.time = 0

        # Start dormant, so that there's no excitation on the
        # day it occurs
        self.dormant = True

        # Record the relevant nodes
        self.excitor_nodes = excitor_nodes
        self.excitee_nodes = excitee_nodes

        # Record the parameters
        self.weibull_weight = weibull_weight
        self.weibull_alpha = weibull_alpha
        self.weibull_beta = weibull_beta

        # Record the raw version of the parameters,
        # prior to being made certainly positive
        # with a smooth, continuous function
        self.lin_val_weight = lin_val_weight
        self.lin_val_alpha = lin_val_alpha
        self.lin_val_beta = lin_val_beta

        # The form of the discrete Weibull distribution
        # allows half of the calculation at time T to be
        # re-used at time T+1. Therefore, the two parts
        # are stored separately.
        self.prob_parts = (0, 0)
        self.probability = sum(self.prob_parts)

        # Record the remaining probability of this
        # excitee edge being excited. Once it goes
        # below the threshold, the excite is dead.
        self.alive_threshold = alive_threshold
        self.remaining_weight = self.weibull_weight

    def prob_part(self, time):
        """Return weighted exp(-(x/alpha)**beta), which can
        be used in the discrete Weibull PMF

        Args:
            time (int): The value for x in the expression

        Raises:
            ValueError: Will not run if the excite is already dead

        Returns:
            float: The evaluated expression
        """
        if not self.alive:
            raise ValueError('Excitation is not alive')

        return self.weibull_weight * part_weibull(time, self.weibull_alpha, self.weibull_beta)

    def increment_probability(self):
        """Use the form of the Weibull PMF to borrow half
        of the calculation from time T to evaluate the
        probability at time T+1

        Raises:
            ValueError: Will not run if the excite is already dead
        """
        if not self.alive:
            raise ValueError('Excitation is not alive')

        if self.dormant:
            # If this is the first time step, initialise the
            # probability parts
            self.dormant = False

            self.prob_parts = (
                self.prob_part(self.time),
                -self.prob_part(self.time+1)
            )
        else:
            # Increment time
            self.time += 1

            # Re-use the second probability part, and calculate
            # the new part
            self.prob_parts = (
                -self.prob_parts[1],
                -self.prob_part(self.time+1)
            )

        # The probability is the sum of the two parts
        self.probability = sum(self.prob_parts)

        # Update the probability of the edge occuring
        # at some future point
        self.remaining_weight = \
            self.remaining_weight - self.probability

    def increment_time(self):
        """Increment time to the next day, updating
        the probability accordingly

        Raises:
            ValueError: Will not run if the excite is already dead
        """
        if not self.alive:
            raise ValueError('Excitation is not alive')

        self.increment_probability()

        if self.remaining_weight < self.alive_threshold:
            # If the probability of the edge occuring on the
            # next day or later is below the alive threshold,
            # then the excite has died
            self.alive = False
