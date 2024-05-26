import math

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

# Choose a sample test point (Note that this point is outside of bounds for some functions!)
"""test_point = np.array([0.5] * n_dimensions)
for p in problems:
    print(p.name)
    print("Bounds from ", p.xl, " to ", p.xu, ".")
    print(p.evaluate(test_point))
    print(sum(p.evaluate(test_point)))"""


def main():
    points = []
    for p in problems:
        xl, xu = p.bounds()
        x_best_sim_an = simulated_annealing(p, np.array([xl, xu]), max_iter=1000)
        evald = np.sum(p.evaluate(x_best_sim_an))
        evald = np.round(evald, 2)
        points.append(x_best_sim_an)
        # Print evald without newline
        print(evald, end='\t')
    print()

    # Save points file
    with open('simulated_annealing_points.txt', 'w') as f:
        for point in points:
            # Separate point coordinates by tab
            f.write('\t'.join([str(coord) for coord in point]) + '\n')




# Visualization -----------------------------------------------------
# Calculates a 2d slice of a n_dimensions-dimensional space
#def sum_of_pareto_functions(DF, x):
#    if len(DF.xl) == 2:
#        return [sum(z) for z in DF.evaluate(np.array(x))]
#    else:
#        xm = list((DF.xl + DF.xu) / 2)
#        x = [[a, b, *xm[2:]] for a, b in x]
#        return [sum(z) for z in DF.evaluate(np.array(x))]
#

# Plots a 2d graph of a function (slice)
#def plot_function(DF):
#    d = 400
#    x = np.linspace(DF.xl[0], DF.xu[0], d)
#    y = np.linspace(DF.xl[1], DF.xu[1], d)
#    X, Y = np.meshgrid(x, y)
#    points = [[x, y] for x, y in zip(X.flatten(), Y.flatten())]
#    Z = sum_of_pareto_functions(DF, points)
#    Z = np.array(Z).reshape((d, d))
#    print(Z)
#
#    # Plotting the functions
#    fig, axs = plt.subplots(1, 1, figsize=(15, 15))
#    axs.contourf(X, Y, Z, levels=50, cmap='viridis')
#    axs.set_title(DF.name)
#    axs.set_xlabel('x')
#    axs.set_ylabel('y')
#    plt.show()
#

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


if __name__ == '__main__':
    main()
