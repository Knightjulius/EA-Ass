import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
import ioh

budget = 5000
dimension = 50

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`
np.random.seed(2023)
def labs_objective_function(x: np.ndarray) -> float:
    autocorr = np.correlate(x, x, mode='full')
    score = np.sum(np.abs(autocorr))  # Objective function score, adjust as needed
    
    return score


ioh.problem.wrap_real_problem(
    labs_objective_function,                                     # Handle to the function
    name="LABS",                               # Name to be used when instantiating
    optimization_type=ioh.OptimizationType.MAX, # Specify that we want to max
    lb=-5,                                               # The lower bound
    ub=5,                                                # The upper bound
)

def studentnumber1_studentnumber2_ES(problem):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population
    pop_size = 1 # TODO: look at if this matters
    lowerbound = -5
    upperbound = 5

    parent = []
    parent_sigma = []
    parent_f = []
    for i in range(pop_size):
        parent.append(np.random.uniform(low = lowerbound,high = upperbound, size = problem.meta_data.n_variables))
        parent_sigma.append(0.05 * (upperbound - lowerbound))
        print(parent[i])
    for i in range(pop_size):
        parent_f.append(problem(parent[i]))

    print(parent, parent_sigma, parent_f)
    print(problem.meta_data.n_variables)
    
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    #while problem.state.evaluations < budget:
        #recombine

        #mutate

        #select

        #evaluate
        
    # no return v 


def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="ESdata",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evoltionary strategy",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    for run in range(20): 
        studentnumber1_studentnumber2_ES(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
        break
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # F19, _logger = create_problem(19)
    # for run in range(20): 
    #     studentnumber1_studentnumber2_ES(F19)
    #     F19.reset()
    # _logger.close()


