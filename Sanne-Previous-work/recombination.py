from ..general import Operator, Solution
from typing import Tuple, Optional, List, Generator
import random

class Recombination(Operator):
    """ Genetic algorithm search recombination operators. """

    def __init__(self, K: int = 0, crossover_rate: float = 0.1) -> None:
        """ Initalise operator with additional parameters.

        Parameters
        ----------
        crossover_rate: float
            The probability that crossover operation is applied on children.
        """
        self.crossover_rate: float = crossover_rate
        self.K: int = K
        self.name = "Rec"

    def __call__(self, parents: Tuple[Solution, ...]) -> Tuple[Solution, ...]:
        raise NotImplementedError("Call method should be overwritten in subclass")

    def Generate_Pairs(self, parents: Tuple[Solution, ...]) -> Generator:
        """Generates random pairs from parent population"""
        if len(parents) % 2 != 0 or len(parents) < 2: # check if parent population size is an even number
            raise ValueError("Population size should be even or it will give problems with the recombination operator")

        pair_list = list(parents)
        random.shuffle(pair_list)
        for i in range(0, len(pair_list), 2):
            p1: Solution = pair_list[i].copy()
            p2: Solution = pair_list[i+1].copy()
            yield (p1, p2)

class NoRecombination(Recombination):
    def __init__(self, K: int = 0, crossover_rate: float = 0.1) -> None:
        super().__init__(K=K, crossover_rate=crossover_rate)
        self.name += "No"

    def __call__(self, parents: Tuple[Solution, ...]) -> Tuple[Solution, ...]:
        return parents

class UniformRecombination(Recombination):
    """ Genetic algorithm random recombination operator.

    For testing purposes.
    """

    def __init__(self, K: int = 0, crossover_rate: float = 0.1) -> None:
        super().__init__(K=K, crossover_rate=crossover_rate)
        self.name += "Uni"

    def __call__(self, parents: Tuple[Solution, ...]) -> Tuple[Solution, ...]:
        """ Create random recombined children from parent subset.

        Pairs of parents will be chosen randomly.
        For each position a swap will occur randomly.
        Two children are generated for every pair of parents.

        Parameters
        ----------
        parents: Tuple[Solution, ...]
            The subset of solutions that has been selected to reproduce.
        """        
        children = []      
        for child1, child2 in self.Generate_Pairs(parents):
            if random.random() < self.crossover_rate: # Apply crossover action with some probability
                # cross over randomly on each position
                for p in range(len(child1.representation)):
                    if random.random() < 0.5: # 50% chance
                        child1.representation[p], child2.representation[p] = child2.representation[p], child1.representation[p] # Swap
            children.append(child1)
            children.append(child2)
        return tuple(children)

class OnePointRecombination(Recombination):

    def __init__(self, K: int = 1, crossover_rate: float = 0.1) -> None:
        super().__init__(K=1, crossover_rate=crossover_rate)
        self.name += "1P"
    
    def __call__(self, parents: Tuple[Solution, ...]) -> Tuple[Solution, ...]:
        """ Create recombined children that consist of one half of one parent + other half of other parent.

        Pairs of parents will be chosen randomly.
        Crossover rate will decide if recombiantion occurs.
        If crossover occurs two children will be created, each consisting of one half of one parent + one half of other parent.
        Two children are generated for every pair of parents.

        Parameters
        ----------
        parents: Tuple[Solution, ...]
            The subset of solutions that has been selected to reproduce.
        """
        # Is the same as k point with k=1 so changed the implementation
        function = KPointRecombination(1, self.crossover_rate)
        return function(parents)
        
class KPointRecombination(Recombination):

    def __init__(self, K: int, crossover_rate: float = 0.1) -> None:
        super().__init__(K=K, crossover_rate=crossover_rate)
        self.name += str(K) + "P"

    def __call__(self, parents: Tuple[Solution, ...]) -> Tuple[Solution, ...]:
        """ Create recombined children that consist of one half of one parent + other half of other parent.

        Pairs of parents will be chosen randomly.
        Crossover rate will decide if recombiantion occurs.
        If crossover occurs, alternating segments defined by crossover points are swapped.
        Two children are generated for every pair of parents.

        Parameters
        ----------
        parents: Tuple[Solution, ...]
            The subset of solutions that has been selected to reproduce.
        """
        children = []
        
        if self.K >= len(parents[0].representation)- 1:
            raise ValueError("For K Point Crossover the value of K can't be larger than the lenght of the solutions - 1")

        for child1, child2 in self.Generate_Pairs(parents):
            if random.random() < self.crossover_rate: # Apply crossover action with some probability
                crossover_points = random.sample(range(1, len(child1.representation) - 1), self.K)
                crossover_points.sort() # QUESTION: is dit nodig? 
                for crossover_point in crossover_points:
                    child1.representation, child2.representation = \
                    child1.representation[:crossover_point] + child2.representation[crossover_point:],\
                    child2.representation[:crossover_point] + child1.representation[crossover_point:] # Swap solutions before and after crossover point
            children.append(child1)
            children.append(child2)
        return tuple(children)
