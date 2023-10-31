import numpy as np

# Function to calculate the autocorrelation of a binary sequence
def autocorrelation(sequence):
    n = len(sequence)
    autocorr = np.correlate(sequence, sequence, mode='full')
    return autocorr[n - 1:]

# Function to generate a random binary sequence of a given length
def generate_random_sequence(length):
    return np.random.randint(2, size=length)

# Function to evaluate the fitness of a binary sequence (lower peak autocorrelation is better)
def fitness(sequence):
    autocorr = autocorrelation(sequence)
    return -max(autocorr)  # We negate it because genetic algorithms maximize fitness

# Function to perform one-point crossover between two parent sequences
def one_point_crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

# Function to perform mutation by flipping a random bit in the sequence
def mutate(sequence, mutation_rate):
    for i in range(len(sequence)):
        if np.random.rand() < mutation_rate:
            sequence[i] = 1 - sequence[i]
    return sequence

# Genetic Algorithm to find a low autocorrelation binary sequence
def genetic_algorithm(sequence_length, population_size, generations, mutation_rate):
    population = [generate_random_sequence(sequence_length) for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate the fitness of each sequence
        fitness_scores = [fitness(seq) for seq in population]

        # Select parents for crossover (higher fitness is better)
        selected_indices = np.argsort(fitness_scores)[:population_size // 2]
        parents = [population[i] for i in selected_indices]

        # Create the next generation through crossover and mutation
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            child1, child2 = one_point_crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            next_generation.extend([child1, child2])

        population = next_generation

    # Find the best sequence after the specified number of generations
    best_sequence = max(population, key=fitness)
    lowest_peak = -fitness(best_sequence)

    return best_sequence, lowest_peak

# Example usage
sequence_length = 20
population_size = 100
generations = 100
mutation_rate = 0.01

best_sequence, lowest_peak = genetic_algorithm(sequence_length, population_size, generations, mutation_rate)
print("Best Sequence:", best_sequence)
print("Lowest Peak Autocorrelation:", lowest_peak)