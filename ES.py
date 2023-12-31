import numpy as np
from ioh import get_problem, logger, ProblemClass
import ioh

dimension = 50

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

def sig(x):
    # takes an array of real values and used the sigmoid function to map the real values to a binary representation
    array = []
    for each in x:
        value = round(1/(1 + np.exp(-each)))
        array.append(value)
    return np.array(array)

def recombination(recombination_input, parents,parents_sigma, lambda_):
    if recombination_input == 1:
        #intermediary combination
        # choose 2 random parents
        offsprings = []
        offsprings_sigma = []
        for i in range(lambda_):
            [p1,p2] = np.random.choice(len(parents),2,replace = False)
            new_sigma = []
            # the new offspring has the average values of the previous parents
            o = (parents[p1] + parents[p2])/2
            offsprings_sigma.append(np.divide(np.add(parents_sigma[p1], parents_sigma[p2]),2))
            offsprings.append(o)


    if recombination_input == 2:
        # discrete recombination
        offsprings = []
        offsprings_sigma =[]
        num_parents = len(parents)
        parent_length = len(parents[0])  # Assuming all parents have the same length
        for i in range(lambda_):
            # Choose 2 random parents
            [p1, p2] = np.random.choice(len(parents),2,replace = False)
            # weight values to keep track of what percentage of the values belonings to what parent
            weightp1 = 0
            weightp2 = 0
            offspring_bits = []
            for j in range(parent_length):
                # Select bit from either parent randomly
                weighted_value = np.random.rand()
                if weighted_value < 0.5:
                    bit = parents[p1][j]
                    weightp1 += 1
                else:
                    bit = parents[p2][j]
                    weightp2 += 1
                offspring_bits.append(bit)
            # Calculate the new sigma values based on the weights calculated earlier
            # the weight is appplied to each value in the sigma
            new_sigma = []
            for k in range(len(parents_sigma[0])):
                weighted_sigma = (parents_sigma[p1][k] * (weightp1/(weightp1 + weightp2))) + (parents_sigma[p2][k] * (weightp2/(weightp1 + weightp2)))
                new_sigma.append(weighted_sigma)
            offsprings_sigma.append(new_sigma)
            offsprings.append(offspring_bits)


    if recombination_input == 3:
        # global intermediary recombination
        # calculates the mean for each bit and makes the offsrping and offspring sigma that
        offsprings = []
        offsprings_sigma = []
        for i in range(lambda_):
            o = np.mean(parents,axis=0)
            s = np.mean(parents_sigma,axis=0)
            offsprings.append(o)
            offsprings_sigma.append(s)


    if recombination_input == 4:
        # global discrete recombination
        offsprings = []
        offsprings_sigma =[]
        num_parents = len(parents)
        parent_length = len(parents[0])  # Assuming all parents have the same length
        # does the same as discrete recombination but uses all parents
        for i in range(lambda_):
            # Perform discrete recombination
            weightp = [[] for _ in range(num_parents)]
            offspring_bits = []
            for j in range(parent_length):
                #choose a random position from the list of parents
                p1 = np.random.randint(0,num_parents)
                bit = parents[p1][j]
                weightp[p1].append(1)
            
                offspring_bits.append(bit)
            # calculates each value in the sigma based on the weight beloning to each bit
            new_sigma = [0 for _ in range(len(parents_sigma[0]))]
            for k in range(len(parents_sigma[0])):
                for position in range(num_parents):
                    ws = (sum(weightp[position])/parent_length)*parents_sigma[position][k]
                    new_sigma[k] += ws
            offsprings_sigma.append(new_sigma)
            offsprings.append(offspring_bits)
    return offsprings, offsprings_sigma


def mutation(mutation_input, parents, parents_sigma,tau, upperbound, lowerbound):
    if mutation_input == 1:
        # one sigma mutatioj
        # set tau according to the recommendation of Schweffel
        for i in range(len(parents)):
            # this changes the sigma randomly and applies it to it's unique individual
            parents_sigma[i][0] = parents_sigma[i][0] * np.exp(tau*np.random.normal(0,1))
            for j in range(len(parents[i])):
                parents[i][j] = parents[i][j] + np.random.normal(0,1)*parents_sigma[i][0]
                parents[i][j] = parents[i][j] if parents[i][j] < upperbound else upperbound
                parents[i][j] = parents[i][j] if parents[i][j] > lowerbound else lowerbound
    if mutation_input == 2:
        # individual sigma
        # set tauprime according to the recommendation of Schweffel
        tau =  1.0 / np.sqrt(2*dimension)
        tauprime = 1.0/np.sqrt(2*np.sqrt(dimension))
        for i in range(len(parents)):
            for j in range(len(parents[i])):
                parents_sigma[i][j] = parents_sigma[i][j] * np.exp(tau*np.random.normal(0,1) + tauprime*np.random.normal(0,1))
                #adds each sigma beloning to each bit to the value
                parents[i][j] = parents[i][j] + (parents_sigma[i][j] * np.random.normal(0,1))
                parents[i][j] = parents[i][j] if parents[i][j] < upperbound else upperbound
                parents[i][j] = parents[i][j] if parents[i][j] > lowerbound else lowerbound
    return parents, parents_sigma


def mating_selection(selection_input,offsprings, offsprings_f, offsprings_sigma, parents, parents_sigma, parents_f,lambda_,mu_):
    # Comma mating selection
    # Ranks the offspring according to the best found fitness
    if selection_input == 1:
        # First the fitness is indexed accordingly with the best fitness value being 0
        indices = np.argsort(offsprings_f)[::-1]
        ranks = np.empty_like(indices)
        ranks[indices] = np.arange(len(offsprings_f))
        parents = []
        parents_sigma = []
        parents_f = []
        # selects only the best offspring according to the fitness
        # in range for the number of parents desired
        for j in range(lambda_):
            if (ranks[j] < mu_):
                parents.append(offsprings[j])
                parents_f.append(offsprings_f[j])
                parents_sigma.append(offsprings_sigma[j])

    if selection_input == 2:
        # plus mating selection
        # Ranks the offspring and parents according to the best found fitness
        combination_f = offsprings_f + parents_f
        combination = offsprings + parents
        combination_sigma  = offsprings_sigma + parents_sigma

        indices = np.argsort(combination_f)[::-1]
        ranks = np.empty_like(indices)
        ranks[indices] = np.arange(len(combination_f))

        #ranks = np.argsort(combination_f)
        parents = []
        parents_sigma = []
        parents_f = []
        # selects only the best fitness found for both parents and offspring according to the fitness
        # in range for the number of parents desired
        for j in range(len(combination)):
            if (ranks[j] < mu_):
                parents.append(combination[j])
                parents_f.append(combination_f[j])
                parents_sigma.append(combination_sigma[j])
    return parents, parents_sigma, parents_f

def s3663841_s2960710_ES(problem,mutation_input, selection_input, recombination_input, initial_sigma, num_parents, num_offspring):
    budget = 5000
    
    # Parameters setting
    #mu_ is the number of parents
    mu_ = num_parents
    # lambda is the number of offspring
    lambda_ = num_offspring
    #tau is the learning rate used for the sigma values
    # set tau according to the recommendation of Schweffel
    tau =  1.0 / np.sqrt(dimension) 
    # initial sigma can be specified
    initial_sig = initial_sigma

    #lower/upper bound set to -1 and 1 for easier sigmoid comparison
    lowerbound = -1
    upperbound = 1

    # Since we are trying to find the highest fitness value the initial optimum is set to 0
    f_opt = 0
    x_opt = None

    # initialize the parents for the number of parents according to the set upper and lowerbound
    # also creates binary values with the sigmoid function that can be used to evaluate the fitness of the function
    parents = []
    parents_binary = []
    parents_sigma = []
    parents_f = []
    
    for i in range(mu_):
        # parent values are normally distributed between -1 and 1
        parents.append(np.random.uniform(low = lowerbound,high = upperbound, size = problem.meta_data.n_variables))
        if mutation_input == 1:
            parents_sigma.append([initial_sig])
        else:
            parents_sigma.append([initial_sig for _ in range(dimension)])
        # parents converted to binary so that fitness evaluation can be done
        parents_binary.append(sig(parents[i]))
        parents_f.append(problem(parents_binary[i]))

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.

    while problem.state.evaluations < budget:

        offsprings_binary = []
        offsprings_f = []
        # Recombination
        offsprings, offsprings_sigma = recombination(recombination_input, parents,parents_sigma, lambda_)
        # mutate the offspring
        offsprings, offsprings_sigma = mutation(mutation_input,offsprings,offsprings_sigma,tau, upperbound, lowerbound)

        # makes the offspring binary so it can be evaluated
        for entry in offsprings:
            offsprings_binary.append(sig(entry))
        # Evaluation
        offsprings_f = problem(offsprings_binary)
        for i in range(lambda_):
            if offsprings_f[i] > f_opt:
                    f_opt = offsprings_f[i]
                    x_opt = offsprings[i].copy()
        # selects and sets new parents
        parents, parents_sigma, parents_f = mating_selection(selection_input, offsprings, offsprings_f, offsprings_sigma, parents, parents_sigma, parents_f,lambda_,mu_)

    return f_opt, x_opt

def create_problem(fid: int, Fname: str, Folname:str):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root=Folname,  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name= Fname,  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    F18, _logger = create_problem(18, 'F18ES', 'F18ES')
    Average_list = []
    np.random.seed(2023)
    for run in range(20):
        # The parameters in the function correspond to Individual Sigma mutation,
        # Plus Selection, Discrete Recombination, an initial sigma of 0.1, mu of 8 and lambda of 56
        f_opt, x_opt = s3663841_s2960710_ES(F18, 2,2,2,0.1,8,56)
        Average_list.append(f_opt)
        F18.reset()
    _logger.close()
    print(sum(Average_list)/len(Average_list))



    F19, _logger = create_problem(19, 'F19ES', 'F19ES')
    Average_list = []
    np.random.seed(2023)
    for run in range(20):
        # The parameters in the function correspond to Individual Sigma mutation,
        # Plus Selection, Discrete Recombination, an initial sigma of 0.1, mu of 14 and lambda of 98 
        f_opt, x_opt = s3663841_s2960710_ES(F19, 2,2,2,0.1,14,98)
        Average_list.append(f_opt)
        F19.reset()
    _logger.close()
    print(sum(Average_list)/len(Average_list))


