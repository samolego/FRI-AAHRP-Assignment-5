import math
import random
import numpy as np
import pymoo.problems.dynamic.df as dfs

n_dimensions = 75
time = 50
problems = [dfs.DF1(time=time, n_var=n_dimensions),
            dfs.DF2(time=time, n_var=n_dimensions),
            dfs.DF3(time=time, n_var=n_dimensions),
            dfs.DF4(time=time, n_var=n_dimensions),
            dfs.DF5(time=time, n_var=n_dimensions),
            dfs.DF6(time=time, n_var=n_dimensions),
            dfs.DF7(time=time, n_var=n_dimensions),
            dfs.DF8(time=time, n_var=n_dimensions),
            dfs.DF9(time=time, n_var=n_dimensions),
            dfs.DF10(time=time, n_var=n_dimensions),
            dfs.DF11(time=time, n_var=n_dimensions),
            dfs.DF12(time=time, n_var=n_dimensions),
            dfs.DF13(time=time, n_var=n_dimensions),
            dfs.DF14(time=time, n_var=n_dimensions)]


# Simmulated Annealing
def simulated_annealing(DF, bounds, num_iters=10, initial_temperature=100, cooling_rate=0.99, max_iter=1000, neighbor_range=0.1):
    """
    Runs a simulated annealing algorithm on the provided function.
    :param DF: Function to be optimized
    :param bounds: The bounds of the function
    :param num_iters: Number of iterations to run the algorithm
    :param initial_temperature: The initial temperature
    :param cooling_rate: Cooling rate of the temperature
    :param max_iter: Maximum number of iterations of the algorithm
    :param neighbor_range: Range of the neighborhood
    :return: The best point found
    """

    def acceptance_probability(old_cost, new_cost, temp):
        if new_cost < old_cost:
            return 1.0
        return math.exp((old_cost - new_cost) / temp)


    x0 = np.random.uniform(low=bounds[0], high=bounds[1], size=n_dimensions)
    best_run_x = x0.copy()

    for _ in range(num_iters):
        x = x0.copy()
        best_x = x0.copy()
        best_f = np.sum(DF.evaluate(x, time=time))
        temperature = initial_temperature

        for i in range(max_iter):
            # randomly perturb the current solution within the bounds
            perturbation = np.random.uniform(-neighbor_range, neighbor_range, size=x.shape)
            x_new = x + perturbation
            x_perturbed = np.clip(x_new, bounds[0], bounds[1])

            # evaluate the perturbed solution
            f_perturbed = np.sum(DF.evaluate(x_perturbed))
            f = np.sum(DF.evaluate(x))

            # accept if the chance for new solution is high enough
            if acceptance_probability(f, f_perturbed, temperature) > np.random.rand():
                x = x_perturbed.copy()
                if f_perturbed < best_f:
                    best_x = x_perturbed.copy()
                    best_f = f_perturbed

            # lower the temperature
            temperature *= cooling_rate

        if np.sum(DF.evaluate(best_x)) < np.sum(DF.evaluate(best_run_x)):
            best_run_x = best_x.copy()

        x0 = np.random.uniform(low=bounds[0], high=bounds[1], size=n_dimensions)

    return best_run_x

# Genetic Algorithm

# Function to generate an individual
def generate_individual(length, xl, xu):
    return np.random.uniform(low=xl, high=xu, size=length)

# Function to compute fitness
def compute_fitness(individual, problem):
    return np.sum(problem.evaluate(np.array([individual])))

# Function to select parents using tournament selection
def selection(population, fitnesses, num_parents, tour_size):
    selected = []
    for _ in range(num_parents):
        tournament = random.sample(range(len(population)), tour_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament]
        winner = tournament[np.argmin(tournament_fitnesses)]
        selected.append(population[winner])
    return selected

# Function for crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

# Function for mutation using problem-specific bounds
def mutate(individual, mutation_rate, xl, xu):
    if random.random() < mutation_rate:
        mutation_point = random.randint(0, len(individual) - 1)
        individual[mutation_point] = np.random.uniform(low=xl[mutation_point], high=xu[mutation_point])
    return individual

# Genetic Algorithm
def genetic_algorithm(problem, num_iterations, chromosome_len, population_size, num_generations, mutation_rate, tournament_size):
    xl, xu = problem.xl, problem.xu
    population = [generate_individual(chromosome_len, xl, xu) for _ in range(population_size)]
    best_individual = None
    best_fitness = float('inf')

    for generation in range(num_generations):
        fitnesses = [compute_fitness(ind, problem) for ind in population]
        
        min_fitness = min(fitnesses)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_individual = population[fitnesses.index(min_fitness)]

        parents = selection(population, fitnesses, population_size, tournament_size)

        # Create next generation
        next_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, xl, xu)
            child2 = mutate(child2, mutation_rate, xl, xu)
            next_population.extend([child1, child2])

        population = next_population

    return best_individual, best_fitness

def main():
    points_sim_an = []
    points_gen_al = []
    for i, p in enumerate(problems):
        xl, xu = p.bounds()
        x_best_sim_an = simulated_annealing(p, np.array([xl, xu]), max_iter=1000)
        evald = np.sum(p.evaluate(x_best_sim_an))
        evald = np.round(evald, 2)
        points_sim_an.append(x_best_sim_an)
        print(f"Problem DF{i+1}: Simulated Annealing {evald}")
        best_individual, best_fitness = genetic_algorithm(p, num_iterations=1000, chromosome_len=n_dimensions, population_size=100, num_generations=10000, mutation_rate=0.01, tournament_size=5)
        points_gen_al.append(best_individual)
        print(f"Problem DF{i+1}: Genetic Algorithm = {best_fitness}")

    # Save points file
    with open('simulated_annealing_points.txt', 'w') as f:
        for point in points:
            f.write('\t'.join([str(coord) for coord in point]) + '\n')

    with open("genetic_algorithm_points.txt", "w") as f:
        for item in points_gen_al:
            f.write("\t".join(map(str, item)) + "\n")

if __name__ == '__main__':
    main()
