from ..general import Operator, Solution
from typing import Tuple, Optional
import random

class Mutation(Operator):
    """ Genetic algorithm search mutation operators. """

    def __init__(self, mutation_rate: float = 0.01) -> None:
        """ Initalise operator with additional parameters.

        Parameters
        ----------
        mutation_rate: float
            The probability that a mutation is applied for a specific position in a solution.
        """
        self.mutation_rate: float = mutation_rate
        self.name: str = "Mut"

    def __call__(self, children: Tuple[Solution, ...]) -> Tuple[Solution, ...]:
        raise NotImplementedError("Call method should be overwritten in subclass")

class NoMutation(Mutation):
    """"""

    def __init__(self, mutation_rate: float = 0.01) -> None:
        super().__init__(mutation_rate)
        self.name += "No"

    def __call__(self, children: Tuple[Solution, ...]) -> Tuple[Solution, ...]:
        return children

class UniformMutation(Mutation):
    """ Genetic algorithm random mutation operator.

    For testing purposes.
    """

    def __init__(self, mutation_rate: float = 0.01) -> None:
        super().__init__(mutation_rate)
        self.name += "Uni"

    def __call__(self, children: Tuple[Solution, ...]) -> Tuple[Solution, ...]:
        """ Apply a random value mutation for each position with a certain probability.

        Parameters
        ----------
        children: Tuple[Solution, ...]
            The group of genered children solutions.
        """
        for child in children:
            for p in range(len(child.representation)): # each position in the solution
                if random.random() < self.mutation_rate: # mutation rate probability
                    child.representation[p] = random.choice(range(child.arity-1))
        return children


class FlipMutation(Mutation):
    """ Genetic algorithm flip mutation operator. """

    def __init__(self, mutation_rate: float = 0.01) -> None:
        super().__init__(mutation_rate)
        self.name += "Flip"

    def __call__(self, children: Tuple[Solution, ...]) -> Tuple[Solution, ...]:
        """ Apply a flip mutation for each position with a certain probability.

        Parameters
        ----------
        children: Tuple[Solution, ...]
            The group of genered children solutions.
        """
        for child in children:
            for p in range(len(child.representation)): # each position in the solution
                if random.random() < self.mutation_rate: # mutation rate probability
                    child.representation[p] = (child.representation[p] + 1) % child.arity
        return children

class SwapMutation(Mutation):
    """ Genetic algorithm swap mutation operator. """

    def __init__(self, mutation_rate: float = 0.01) -> None:
        super().__init__(mutation_rate)
        self.name += "Swap"

    def __call__(self, children: Tuple[Solution, ...]) -> Tuple[Solution, ...]:
        """ Apply a swap mutation for each solution with a certain probability.

        Parameters
        ----------
        children: Tuple[Solution, ...]
            The group of genered children solutions.
        """
        # Scale mutation rate since it will be per solution instead of position (and each mutation will change 2 positions)
        dimension = len(children[0].representation)
        solution_mutation_rate = self.mutation_rate * dimension / 2
        for child in children:
            if random.random() < solution_mutation_rate: # scaled mutation rate probability
                first = random.randint(0, dimension-1)
                second = first
                while (first == second):
                    second = random.randint(0, dimension-1)
                child.representation[first], child.representation[second] = child.representation[second], child.representation[first] # Swap value
        return children
