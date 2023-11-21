import sys

import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger

budget = 5000

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

def studentname1_studentname2_GA(problem):
    # initial_pop = ... make sure you randomly create the first population
    while not problem.final_target_hit and problem.evaluations < budget * budget:
        # please implement the mutation, crossover, selection here
        # .....
        # this is how you evaluate one solution `x`
        # f = problem(x)
    # no return value needed 

def studentname1_studentname2_ES(problem):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population
    while not problem.final_target_hit and problem.evaluations < budget * budget:
        # please implement the mutation, crossover, selection here
        # .....
        # this is how you evaluate one solution `x`
        # f = problem(x)
    # no return value needed 


def create_problem(fid: int):
    # Declaration of problems to be tested.
    # We obtain an interface of the OneMax problem here.
    dimension = 30
    problem = get_problem(fid, dimension=dimension, instance=1, problem_type="PBO")

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    for run in range(20): 
        studentname1_studentname2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    F19, _logger = create_problem(19)
    for run in range(20): 
        studentname1_studentname2_GA(F19)
        F19.reset()
    _logger.close()

    F18, _logger = create_problem(18)
    for run in range(20): 
        studentname1_studentname2_ES(F18)
        F18.reset()
    _logger.close()

    F19, _logger = create_problem(19)
    for run in range(20): 
        studentname1_studentname2_ES(F19)
        F19.reset()
    _logger.close()


