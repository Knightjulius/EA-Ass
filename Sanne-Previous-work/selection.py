from ..general import Operator, Solution
from typing import Tuple, Optional
import random
import math
import numpy as np

class Selection(Operator):
    """ Genetic algorithm search selection operators. """

    def __init__(self, select=10) -> None:
        """ Initalise operator with additional parameters."""
        self.t_select = select
        self.name = "Sel"

    def __call__(self, population: Tuple[Solution, ...], amount: int) -> Tuple[Solution, ...]:
        raise NotImplementedError("Call method should be overwritten in subclass")

class NoSelection(Selection):
    """ """
    def __init__(self, select=10) -> None:
        super().__init__(select)
        self.name += "No"

    def __call__(self, population: Tuple[Solution, ...], amount: int) -> Tuple[Solution, ...]:
        return population[:amount]

class UniformSelection(Selection):
    """ """
    def __init__(self, select=10) -> None:
        super().__init__(select)
        self.name += "Uni"

    def __call__(self, population: Tuple[Solution, ...], amount: int) -> Tuple[Solution, ...]:
        for solution in population:
            if solution.score == -1:
                raise ValueError("Score must be calculated")
        new_population = random.choices(population, k=amount)
        return tuple(new_population)

class RouletteSelection(Selection):
    """ """
    def __init__(self, select=10) -> None:
        super().__init__(select)
        self.name += "Roul"

    def __call__(self, population: Tuple[Solution, ...], amount: int) -> Tuple[Solution, ...]:
        for solution in population:
            if solution.score == -1:
                raise ValueError("Score must be calculated")
        scores = [solution.score for solution in population]
        return tuple(random.choices(population, k=amount, weights=scores))


class TournamentSelection(Selection):
    """ """
    def __init__(self, select=10) -> None:
        super().__init__(select)
        self.name += "Tour"

    def __call__(self, population: Tuple[Solution, ...], amount: int) -> Tuple[Solution, ...]:
        for solution in population:
            if solution.score == -1:
                raise ValueError("Score must be calculated")
        selected_population = []
        population_list = list(population)
        for i in range(amount):
            selected = random.choices(population_list, k=self.t_select)
            selected.sort(key=lambda sol: sol.score, reverse=True)
            winner = selected[0]
            selected_population.append(winner)
        return tuple(selected_population)

class RankSelection(Selection):
    """ """
    def __init__(self, select=10) -> None:
        super().__init__(select)
        self.name += "Rank"

    def __call__(self, population: Tuple[Solution, ...], amount: int) -> Tuple[Solution, ...]:
        for solution in population:
            if solution.score == -1:
                raise ValueError("Score must be calculated")
        population_list = list(population)
        population_list.sort(key=lambda sol: sol.score, reverse=True)
        scores = []
        for i in range(1, len(population_list)+1):
            scores.append(1/i)
        return tuple(random.choices(population_list, k=amount, weights=scores))
