'''

	CVXPY is an open source Python-embedded modeling language for convex optimization problems. 
	It lets you express your problem in a natural way that follows the math, 
	rather than in the restrictive standard form required by solvers.

	https://www.cvxpy.org/examples/

 	python3 -m pip install cvxpy
'''


'''

Non-negative matrix factorization using alternate minimization
https://www.cvxpy.org/examples/applications/nonneg_matrix_fact.html

This example generates a random matrix A and obtains an
approximate solution to the above problem by first generating
a random initial guess for  and then alternatively minimizing 
over X and Y for a fixed number of iterations.


'''

import cvxpy as cp
import numpy as np

# Ensure repeatably random problem data.
np.random.seed(0)

# Generate random data matrix A.
m = 10
n = 10
k = 3
A = np.random.rand(m, k).dot(np.random.rand(k, n))

# Initialize Y randomly.
Y_init = np.random.rand(m, k)


# Ensure same initial random Y, rather than generate new one
# when executing this cell.
Y = Y_init

# Perform alternating minimization.
MAX_ITERS = 30
residual = np.zeros(MAX_ITERS)
for iter_num in range(1, 1+MAX_ITERS):
    # At the beginning of an iteration, X and Y are NumPy
    # array types, NOT CVXPY variables.

    # For odd iterations, treat Y constant, optimize over X.
    if iter_num % 2 == 1:
        X = cp.Variable(shape=(k, n))
        constraint = [X >= 0]
    # For even iterations, treat X constant, optimize over Y.
    else:
        Y = cp.Variable(shape=(m, k))
        constraint = [Y >= 0]

    # Solve the problem.
    # increase max iters otherwise, a few iterations are "OPTIMAL_INACCURATE"
    # (eg a few of the entries in X or Y are negative beyond standard tolerances)
    obj = cp.Minimize(cp.norm(A - Y@X, 'fro'))
    prob = cp.Problem(obj, constraint)
    prob.solve(solver=cp.SCS, max_iters=10000)

    if prob.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")

    print('Iteration {}, residual norm {}'.format(iter_num, prob.value))
    residual[iter_num-1] = prob.value

    # Convert variable to NumPy array constant for next iteration.
    if iter_num % 2 == 1:
        X = X.value
    else:
        Y = Y.value