

'''
Let's have a look at the handcrafted solution to NNLS
From the example in scipy, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
'''

import numpy as np
A = np.array([[1, 0], [1, 0], [0, 1]])
b = np.array([2, 1, 1])

from lsqnonneg import  lsqnonneg

#nnls(A, b) 
# (array([1.5, 1. ]), 0.7071067811865475)


print('Expected: (array([1.5, 1. ]), 0.7071067811865475)')


C = A
d = b
res = lsqnonneg(C, d, x0=None, tol=None, itmax_factor=3)
print(f'Getting: {res}')



A = np.array([[1, 0], [1, 0], [0, 1]])
b = np.array([-1, -1, -1])
#nnls(A, b)
expected_result = '(array([0., 0.]), 1.7320508075688772)'
print(f'Expected: {expected_result}')
res = lsqnonneg(A, b, x0=None, tol=None, itmax_factor=3)
print(f'Getting: {res}')



# let's try the faster approach
from fnnls import fnnls
AtA = np.dot(A.T, A)
Aty = np.dot(A.T, b)

fast_res = fnnls(AtA, Aty, epsilon=None, iter_max=None)
expected_result = '(array([0., 0.]), 1.7320508075688772)'
print(f'Expected: {expected_result}')
print(f'Fast NNLS: {fast_res}')


