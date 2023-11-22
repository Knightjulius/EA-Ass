import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

dimension = 50
np.random.seed(2023)

#Julius: copied from wg assignment
def crossover(p1, p2, crossover_probability):
   if(np.random.uniform(0,1) < crossover_probability):
        for i in range(len(p1)) :
            if np.random.uniform(0,1) < 0.5:
                t = p1[i]
                p1[i] = p2[i]
                p2[i] = t

def mutation(p, mutation_rate):
    for i in range(len(p)) :
        if np.random.uniform(0,1) < mutation_rate:
            p[i] = 1 - p[i]

def mating_seletion(parent, parent_f, tournament_k):
    # Using the tournament selection
    select_parent = []
    for i in range(len(parent)) :
        pre_select = np.random.choice(len(parent_f),tournament_k)
        max_f = 0
        for p in pre_select:
            if parent_f[p] > max_f:
                index = p
                max_f = parent_f[p]
        select_parent.append(parent[index].copy())
    return select_parent

def studentnumber1_studentnumber2_GA(problem):
    # initial_pop = ... make sure you randomly create the first population
    # Julius: copied from wg assignment but dunno what range is logical
    budget = 5000
    pop_size = np.random.randint(1,10)
    tournament_k = 10
    mutation_rate = 0.02
    crossover_probability = 0.5
    parent = []
    parent_f = []
    # set the initial optimum to 0
    f_opt = 0
    for i in range(pop_size):
        # Initialization
        parent.append(np.random.randint(2, size = problem.meta_data.n_variables))
        parent_f.append(problem(parent[i]))
        
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        offspring = mating_seletion(parent,parent_f, tournament_k)

        for i in range(0,pop_size - (pop_size%2),2) :
            crossover(offspring[i], offspring[i+1], crossover_probability)
        for i in range(pop_size):
            mutation(offspring[i], mutation_rate)
        parent = offspring.copy()

        for i in range(pop_size) : 
            parent_f[i] = problem(parent[i])
            budget = budget - 1
            if parent_f[i] > f_opt:
                    f_opt = parent_f[i]
                    x_opt = parent[i].copy()
            if f_opt >= problem.state.evaluations:
                break
    return f_opt, x_opt

def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

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
        f_opt, x_opt = studentnumber1_studentnumber2_GA(F18)
        print(f'The optimal values for the F18 prob are: {f_opt,x_opt}')
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    F19, _logger = create_problem(19)
    for run in range(20): 
        f_opt, x_opt = studentnumber1_studentnumber2_GA(F19)
        print(f'The optimal values for the F19 prob are: {f_opt,x_opt}')
        F19.reset()
    _logger.close()