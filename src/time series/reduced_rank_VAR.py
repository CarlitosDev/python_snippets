'''
reduced_rank_VAR.py


https://towardsdatascience.com/reduced-rank-vector-autoregressive-model-for-high-dimensional-time-series-forecasting-bdd17df6c5ab

(This is a bit similar to Sergio's thesis but for time series)

We have way more variables than time observations.

So this author argues that by reducing the rank of the coefficients variable, the issue of over-parameterization is overcome. The coeeficients 
matrix is expressed as product of two other matrices.


'''


import numpy as np
def rrvar(data, R, pred_step, maxiter = 100):
    """
    Reduced-rank VAR algorithm.
    from this guy: https://xinychen.github.io
    """
    
    N, T = data.shape
    X1 = data[:, : -1]
    X2 = data[:, 1 :]
    V = np.random.randn(R, N)
    for it in range(maxiter):
        W = X2 @ np.linalg.pinv(V @ X1)
        V = np.linalg.pinv(W) @ X2 @ np.linalg.pinv(X1)
    mat = np.append(data, np.zeros((N, pred_step)), axis = 1)
    for s in range(pred_step):
        mat[:, T + s] = W @ V @ mat[:, T + s - 1]
    return mat[:, - pred_step :]



# demo
X = np.zeros((20, 10))
for i in range(20):
    X[i, :] = np.arange(i + 1, i + 11)
# this is the predicted time steps
pred_step = 3
# this is the desired rank
# np.linalg.matrix_rank(X)
R = 2
mat_hat = rrvar(X, R, pred_step)
print(mat_hat)