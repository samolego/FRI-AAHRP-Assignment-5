#include "cec2018_DF.h"
#include <vector>
#include <cstdlib>

using namespace std;

vector<double> simulated_annealing(char* DF, double lbound, double ubound) {
    int n_dims = 75;
    int n_iter = 1000;
    int max_evals = 10000;

    // Choose a random initial solution
    vector<double> x(n_dims);
    for (int i = 0; i < n_dims; i++) {
        x[i] = lbound + (ubound - lbound) * rand() / RAND_MAX;
    }

    for (int i = 0; i < n_iter; i++) {
        // Choose a random neighbour
        vector<double> y(x);
        int j = rand() % n_dims;
        y[j] = lbound + (ubound - lbound) * rand() / RAND_MAX;

        // Calculate the objective function
        vector<double> f_x = cec2018_DF(DF, x, i);
        vector<double> f_y = cec2018_DF(DF, y, i);

        // Calculate the acceptance probability
        double delta = f_y[0] - f_x[0];
        double p = exp(-delta / i);

        // Accept the new solution with probability p
        if (rand() / RAND_MAX < p) {
            x = y;
        }
    }
}