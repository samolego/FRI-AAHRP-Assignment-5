import matplotlib.pyplot as plt
import numpy as np
import pymoo.problems.dynamic.df as dfs

n = 75
time = 50
problems = [dfs.DF1(time=time, n_var=n),
            dfs.DF2(time=time, n_var=n),
            dfs.DF3(time=time, n_var=n),
            dfs.DF4(time=time, n_var=n),
            dfs.DF5(time=time, n_var=n),
            dfs.DF6(time=time, n_var=n),
            dfs.DF7(time=time, n_var=n),
            dfs.DF8(time=time, n_var=n),
            dfs.DF9(time=time, n_var=n),
            dfs.DF10(time=time, n_var=n),
            dfs.DF11(time=time, n_var=n),
            dfs.DF12(time=time, n_var=n),
            dfs.DF13(time=time, n_var=n),
            dfs.DF14(time=time, n_var=n)]

# Choose a sample test point (Note that this point is outside of bounds for some functions!)
test_point = np.array([0.5] * n)
for p in problems:
    print(p.name)
    print("Bounds from ", p.xl, " to ", p.xu, ".")
    print(p.evaluate(test_point))
    print(sum(p.evaluate(test_point)))


# Visualization -----------------------------------------------------
# Calculates a 2d slice of a n-dimensional space
def sum_of_paretno_functions(DF, x):
    if len(DF.xl) == 2:
        return [sum(z) for z in DF.evaluate(np.array(x))]
    else:
        xm = list((DF.xl + DF.xu) / 2)
        x = [[a, b, *xm[2:]] for a, b in x]
        return [sum(z) for z in DF.evaluate(np.array(x))]


# Plots a 2d graph of a function (slice)
def plot_function(DF):
    d = 400
    x = np.linspace(DF.xl[0], DF.xu[0], d)
    y = np.linspace(DF.xl[1], DF.xu[1], d)
    X, Y = np.meshgrid(x, y)
    points = [[x, y] for x, y in zip(X.flatten(), Y.flatten())]
    Z = sum_of_paretno_functions(DF, points)
    Z = np.array(Z).reshape((d, d))
    print(Z)

    # Plotting the functions
    fig, axs = plt.subplots(1, 1, figsize=(15, 15))
    axs.contourf(X, Y, Z, levels=50, cmap='viridis')
    axs.set_title(DF.name)
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    plt.show()


for p in problems:
    plot_function(p)
