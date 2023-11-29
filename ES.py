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

def sig(x):
    array = []
    for each in x:
        value = round(1/(1 + np.exp(-each)))
        array.append(value)
    return np.array(array)

def recombination(parent,parent_sigma):
    # choose 2 random parents, create the offspring and the corresponding sigma
    [p1,p2] = np.random.choice(len(parent),2,replace = False)
    offspring = (parent[p1] + parent[p2])/2
    sigma = (parent_sigma[p1] + parent_sigma[p2])/2 

    return offspring,sigma

def mutation(parent, parent_sigma,tau, upperbound, lowerbound):
    for i in range(len(parent)):
        # this changes the sigma randomly
        parent_sigma[i] = parent_sigma[i] * np.exp(np.random.normal(0,tau))
        for j in range(len(parent[i])):
            parent[i][j] = parent[i][j] + np.random.normal(0,parent_sigma[i])
            parent[i][j] = parent[i][j] if parent[i][j] < upperbound else upperbound
            parent[i][j] = parent[i][j] if parent[i][j] > lowerbound else lowerbound   

def mating_selection(offspring, offspring_f, offspring_sigma,lambda_,mu_ ):
    rank = np.argsort(offspring_f)
    #print(offspring_f)
    #print(rank)
    parent = []
    parent_sigma = []
    parent_f = []
    i = 0
    while ((i < lambda_) & (len(parent) < mu_)):
        if (rank[i] < mu_):
            parent.append(offspring[i])
            parent_f.append(offspring_f[i])
            parent_sigma.append(offspring_sigma[i])
        i = i + 1
    return parent, parent_sigma, parent_f

def studentnumber1_studentnumber2_ES(problem):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population
    
    budget = 5000
    
    # Parameters setting
    mu_ = 5
    lambda_ = 10
    tau =  1.0 / np.sqrt(problem.meta_data.n_variables) #equals the learning rate: this is a recommendation and probably doesnt need to be changed

    pop_size = 5 # TODO: look at if this matters === mu_ in wg assignment, needs to be higher than 1 for recombinationm to work
    lowerbound = -5
    upperbound = 5

    f_opt = 0
    x_opt = None

    parent = []
    parent_binary = []
    parent_sigma = []
    parent_f = []
    for i in range(pop_size):
        parent.append(np.random.uniform(low = lowerbound,high = upperbound, size = problem.meta_data.n_variables))
        parent_sigma.append(0.05 * (upperbound - lowerbound))
        #print(parent[i])
        parent_binary.append(sig(parent[i]))
        parent_f.append(problem(parent_binary[i]))
    #print(parent, parent_binary, parent_sigma, parent_f)
    
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:

        offspring = []
        offspring_sigma = []
        offspring_f = []
        offspring_binary = []

        # Recombination
        for i in range(lambda_):
            o, s = recombination(parent,parent_sigma)
            offspring.append(o)
            offspring_sigma.append(s)

        # this changes the offspring not the parent
        mutation(offspring,offspring_sigma,tau, upperbound, lowerbound)
        # makes the offspring binary so it can be evaluated
        for entry in offspring:
            offspring_binary.append(sig(entry))
        #print(f'len binary:{len(offspring_binary)}')

        # Evaluation
        for i in range(lambda_) : 
            offspring_f.append(problem(offspring_binary[i]))
            #print(offspring_f)
            # TODO min or max problem?
            if offspring_f[i] > f_opt:
                    f_opt = offspring_f[i]
                    # does it matte if this is non binary or binary?
                    x_opt = offspring[i].copy()
            #print(f_opt)
        # selects and sets new parents
        parent, parent_sigma, parent_f = mating_selection(offspring, offspring_f, offspring_sigma,lambda_,mu_)
    return f_opt, x_opt

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
    f18_list = []
    for run in range(20): 
        f_opt, x_opt = studentnumber1_studentnumber2_ES(F18)
        f18_list.append(f_opt)
        F18.reset() # it is necessary to reset the problem after each independent run
    print(f'The average fitness for the F18 prob is: {sum(f18_list)/len(f18_list)}')
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # F19, _logger = create_problem(19)
    # for run in range(20): 
    #     studentnumber1_studentnumber2_ES(F19)
    #     F19.reset()
    # _logger.close()


