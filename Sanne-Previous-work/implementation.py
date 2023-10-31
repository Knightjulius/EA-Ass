"""NACO assignment 22/23.

This file contains the skeleton code required to solve the first part of the
assignment for the NACO course 2022.

You can test your algorithm by using the function `test_algorithm`. For passing A1,
your GA should be able to pass the checks in this function for dimension=100.
"""

import ioh # type: ignore
import math
import random
from typing import List, Tuple, Optional, Dict, Any
from modules.operators.mutation import *
from modules.operators.recombination import *
from modules.operators.selection import *
from modules.general import Solution

class GeneticAlgorithm:
    """An implementation of the Genetic Algorithm."""

    def __init__(self,
            budget: int,
            population_size: int = 50,
            recombination: Recombination = OnePointRecombination(),
            mutation: Mutation = FlipMutation(),
            selection: Selection = RankSelection(),
            children_proportion: float = 0.8,
            children_factor: float = 1.5
        ) -> None:
        """Construct a new GA object.

        Different options for operators can be passed with keyword arguments:
        e.g. GeneticAlgorithm(x, mutation=UniformMutation(0.5), ...)
        Make sure to initialise the operators before passing them.

        Parameters
        ----------
        budget: int
            The maximum number objective function evaluations
            the GA is allowed to do when solving a problem.

        Keyword arguments (optional)
        ----------
        population_size: int
            How many solutions the population should consist of.
            Default is 100.
        recombination: function
            Recombination operator.
            Default is OnePointRecombination with a recombination rate of 0.1.
        mutation: function
            Mutation operator.
            Default is FlipMutation with a mutation rate of 0.01.
        selection: function
            Selection operator.
            Default is RouletteSelection.
        children_proportion: float
            The proportion of children to include in the new generation.
            Default is 0.8
        children_factor: float
            The factor that the amount of children produced should have compared to the population size.
            Default is 1.5
        Notes
        -----
        *   You can add more parameters to this constructor, which are specific to
            the GA or one of the (to be implemented) operators, such as a mutation rate.
        """

        if children_proportion > 1 or children_proportion < 0:
            raise ValueError("Wrong proportion")
        if population_size == 0:
            raise ValueError("Population cannot be empty")
        
        self.population: Tuple[Solution, ...]
        self.budget: int = budget
        self.population_size: int = population_size
        self.recombination: Recombination = recombination
        self.mutation: Mutation = mutation
        self.selection: Selection = selection
        self.children_proportion: float = children_proportion
        self.children_factor: float = children_factor

    def __call__(self, problem: ioh.problem.Integer, stopping: Optional[int] = None, k: int = 2) -> ioh.IntegerSolution:
        """Run the GA on a given problem instance.

        Parameters
        ----------
        problem: ioh.problem.Integer
            An integer problem, from the ioh package. This version of the GA
            should only work on binary/discrete search spaces.

        Notes
        -----
        *   This is the main body of you GA. You should implement all the logic
            for this search algorithm in this method. This does not mean that all
            the code needs to be in this method as one big block of code, you can
            use different methods you implement yourself.

        """
        def score(representation):
            """ Find score and decrease budget. """
            self.budget -= 1
            if self.budget < 0:
                raise RuntimeError("Too many evaluations were made")
            return problem(representation)

        # Initalise population of random solutions
        population = []
        for i in range(self.population_size):
            solution_representation = random.choices(range(k), k=problem.meta_data.n_variables) # Random values in dimension k
            solution_fitness = score(solution_representation) # Evaluate inital solutions
            population.append(Solution(solution_representation, solution_fitness, arity=k))
        self.population = tuple(population) # Population inmutable, indivduals mutable

        # Calculate how many indivduals to create
        num_children = math.floor(self.population_size * self.children_proportion * self.children_factor)
        # Calculate how many individuals to include in the new population
        num_current = math.ceil(self.population_size * (1 - self.children_proportion))
        num_new = 0 if self.children_factor == 0 else math.floor(num_children / self.children_factor)
        # Evaluate populations while the budget is met
        # Do not change this logic but change the initalisation to swithch operators
        while self.budget >= num_children:
            parents = self.selection(self.population, num_children) # Select which solutions in the mating pool
            children = self.recombination(parents) # Recombination of parent solutions to create 'children' solutions
            children = self.mutation(children) # Mutations on children to increase diversity
            for child in children: # Evaluate children
                child.score = score(child.representation)
            new_population = list(self.selection(self.population, num_current)) # Select which solutions will stay in the population
            new_population = new_population + list(self.selection(children, num_new)) # Add new solutions to the population
            self.population = tuple(new_population)

            # Checks
            if len(children) != len(parents):
                raise RuntimeError("Wrong amount of children were produced")
            if len(self.population) != self.population_size:
                raise RuntimeError("Population size changed after evaluation, something is wrong")
            # Stop for problems with defined optimum scores and/or defined optimum values
            if (stopping is not None and problem.state.current_best.y >= stopping) \
                    or problem.state.optimum_found: 
                break

        # Return best solution
        return problem.state.current_best

if __name__ == '__main__':
    pass
