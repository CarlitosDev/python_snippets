'''
From William Koehrsen 
(https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0)

Bayesian optimization is a probabilistic model based approach for finding the minimum of any function that returns a real-value metric.

Bayesian optimization, also called Sequential Model-Based Optimization (SMBO), implements this idea by building a probability model 
of the objective function that maps input values to a probability of a loss: p (loss | input values). The probability model, also called 
the surrogate or response surface, is easier to optimize than the actual objective function. Bayesian methods select the next values to 
evaluate by applying a criteria (usually Expected Improvement) to the surrogate. The concept is to limit evals of the objective 
function by spending more time choosing the next values to try.

Recent results suggest Bayesian hyperparameter optimization of machine learning models is more efficient than manual, 
random, or grid search with:
    - Better overall performance on the test set
    - Less time required for optimization

Grid or random search might be adequeate as long as evaluations of the objective function are 'cheap'. Random search is actually
 more efficient than grid search for problems with high dimensions, but is still an uniformed method where the search does not
  use previous results to pick the next input values to try.


HyperOpt

https://hyperopt.github.io/hyperopt/
https://github.com/hyperopt/hyperopt/wiki


To read:
https://arimo.com/data-science/2016/bayesian-optimization-hyperparameter-tuning/
https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f



Installation notes
------------------

    pip3 install hyperopt
    As there is a conflict with networkx, 
    Option A:
        Either downgrading networkx to 1.11 with pip3 install networkx==1.11
    Option B:    
        pip3 install networkx --upgrade
        pip3 install --upgrade git+git://github.com/hyperopt/hyperopt.git


Alternatives to HyperOpt
(1) https://github.com/HIPS/Spearmint
(2) pip3 install bayesian-optimization


# Have a look at this tutorial
https://www.kaggle.com/fanvacoolt/tutorial-on-hyperopt


# Advanced tour
https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb

'''




import numpy as np
import pandas as pd
from hyperopt import hp, tpe, fmin, Trials, rand, STATUS_OK
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer


# We have to provide with a 'space' where to optimise ('search space')
opt_space = hp.normal('x', 4.9, 0.5)


# Bayesian Optimization methods differ in how they construct the surrogate function: common choices 
# include Gaussian Processes, Random Forest Regression and the Tree Parzen Estimator (TPE).
# Currently two algorithms are implemented in hyperopt:
#   - Random Search
#   - Tree of Parzen Estimators (TPE)
opt_algorithm  = tpe.suggest
#opt_algorithm  = rand.suggest

# objective
opt_objective = lambda x: np.poly1d([1, -2, -28, 28, 12, -26, 100])(x)


x = np.linspace(-5, 6, 10000)
y = opt_objective(x)
miny = min(y)
minx = x[np.argmin(y)]



# inspect the progression of the optimisation
tpe_trials = Trials()

# Single line bayesian optimization of polynomial function
best = fmin(fn = opt_objective,
            space = opt_space, 
            algo = opt_algorithm,
            trials = tpe_trials,
            max_evals = 2000)

print(best)


# Dataframe of results from optimization
tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results], 
                            'iteration': tpe_trials.idxs_vals[0]['x'],
                            'x': tpe_trials.idxs_vals[1]['x']})
                            
tpe_results.head()


plt.figure(figsize = (10, 8))
plt.plot(tpe_results['iteration'], tpe_results['x'],  'bo', alpha = 0.25);
plt.xlabel('Iteration', size = 22); 
plt.ylabel('x value', size = 22); 
plt.title('TPE Sequence of Values', size = 24);
plt.hlines(minx, 0, 2000, linestyles = '--', colors = 'r');
plt.show()


# histogram of the values
plt.figure(figsize = (8, 6))
plt.hist(tpe_results['x'], bins = 50, edgecolor = 'k');
plt.title('Histogram of TPE Values'); plt.xlabel('Value of x'); plt.ylabel('Count');
plt.show()



# Modify the cost function
def objective(x):
    """Objective function to minimize with smarter return values"""
    
    # Create the polynomial object
    f = np.poly1d([1, -2, -28, 28, 12, -26, 100])

    # Evaluate the function
    start = timer()
    loss = f(x) * 0.05
    end = timer()
    
    # Calculate time to evaluate
    time_elapsed = end - start
    
    results = {'loss': loss, 'status': STATUS_OK, 'x': x, 'time': time_elapsed}
    
    # Return dictionary
    return results



tpe_trials_v2 = Trials()
best_v2 = fmin(fn = objective,
            space = opt_space, 
            algo = opt_algorithm,
            trials = tpe_trials_v2,
            max_evals = 2000)

print(best_v2)


# Results into a dataframe
results_df = pd.DataFrame({'time': [x['time'] for x in tpe_trials_v2.results], 
                           'loss': [x['loss'] for x in tpe_trials_v2.results],
                           'x': [x['x'] for x in tpe_trials_v2.results],
                            'iteration': list(range(len(tpe_trials_v2.results)))})

# Sort with lowest loss on top
results_df = results_df.sort_values('loss', ascending = True)
results_df.head()

sns.kdeplot(results_df['x'], label = 'Uniform Domain')
plt.legend(); plt.xlabel('Value of x'); 
plt.ylabel('Density'); plt.title('Comparison of Domain Choice using TPE');
plt.show()


'''

Take home messages from WillKoehrsen

- The differences between random search and and Sequential Model-Based Optimization is clear: random search is uninformed 
and therefore requires more trials to minimize the objective function. The Tree Parzen Estimator, an algorithm used
for SMBO, spends more time choosing the next values, but overall requires fewer evaluations of the objective function
because it is able to reason about the next values to evaluate. Over many iterations, SMBO algorithms concentrate the 
search around the most promising values, yielding:

    - Lower scores on the objective function
    - Faster optimization
    - Bayesian model-based optimiziation means construction a probability model $p(y | x)$ of the objective function and updating 
    this model as more information is collected. As the number of evaluations increases, the model (also called a surrogate function) 
    becomes a more accurate respresentation of the objective function and the algorithm spends more time evaluating promising values.

'''




## Updates June 2019
'''
    Quick review of the two main libraries
'''
import pandas as pd
import numpy as np
import shap
from bayes_opt import BayesianOptimization

# Fake sales
num_samples = 500

x3_values = np.random.randint(100,200, size=(num_samples))
print(f'Mean value {x3_values.mean()} \pm {x3_values.std()}')

x4_values = np.random.randint(50,100, size=(num_samples))

noisy_sales = 10 * np.random.random(size=(num_samples)) + x3_values - x4_values

df = pd.DataFrame({
    "x1" : np.random.randint(0, 100, size=(num_samples)),
    "x2" : np.random.randint(0, 200, size=(num_samples)),
    "x3" : x3_values,
    "x4" : x4_values,
    "sales": noisy_sales
})


def black_box_function(x_1, x_2, x_3, x_4):
    """Function with unknown internals we wish to *maximize*.
    """
    y = 10*np.random.random(1)[0] + x_3 - x_4
    y_hat = np.sum([x_1, x_2, x_3, x_4])
    return -1*np.abs(y-y_hat)


# Bounded region of parameter space
pbounds = {'x_1': (0, 200), 'x_2': (0, 200), 'x_3': (0, 500), 'x_4': (0, 500)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=4,
    n_iter=8,
)

print(optimizer.max)


######
# Compare to HyperOpt
######
import hyperopt as hpo


# Search space

opt_space = {'x_1': hpo.hp.normal('x_1', 0, 5),
'x_2': hpo.hp.normal('x_2', 0, 50),
'x_3': hpo.hp.normal('x_3', 0, 500),
'x_4': hpo.hp.normal('x_4', 0, 500)}



#   - Random Search
#   - Tree of Parzen Estimators (TPE)
opt_algorithm  = hpo.tpe.suggest
#opt_algorithm  = hpo.rand.suggest


# inspect the progression of the optimisation
tpe_trials = hpo.Trials()

results_hopt = fmin(fn = lambda x_in: -black_box_function(x_in['x_1'], x_in['x_2'], x_in['x_3'], x_in['x_4']),
            space = opt_space, 
            algo = opt_algorithm,
            trials = tpe_trials,
            max_evals = 100)


# Explore the results
print(results_hopt)



####
# Classic optimisation
###

from scipy.optimize import minimize
# Plotting limits
x_limits = (-10, 10)
y_limits = (-10, 10)

# Initial guess and list of traversed points
xy_M = [(0.0, 0.0, 10.0, 10.0)]
n_pts = 10

# Function to dump current point in minimization algorithm to list
def f_current_point(x):
    xy_M.append(x)
    return None

# Starting maximization (minus sign since most optimization problems are 
# formulated in terms of finding minimum)
res_M = minimize(lambda x: -black_box_function(*x), x0=xy_M[0], method='L-BFGS-B', 
                   bounds=[x_limits, y_limits, x_limits, y_limits], 
                   options={'maxiter': n_pts**2}, 
                   callback=f_current_point)

print(res_M)


### Updates March 2020
#
# Same problem but using SKOpt
# Have a read: https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html
from skopt import gp_minimize
bounds = [x_limits, y_limits, x_limits, y_limits] 
fnc_min = lambda x: -black_box_function(*x)

res = gp_minimize(fnc_min, # the function to minimize
                  bounds,       # the bounds on each dimension of x
                  acq_func="EI",       # the acquisition function
                  n_calls=15,          # the number of evaluations of f
                  n_random_starts=5,   # the number of random initialization points
                  noise=0.1**2,        # the noise level (optional)
                  random_state=1234)   # the random seed

print(res)

from skopt.plots import plot_convergence
plot_convergence(res)
plt.show()

