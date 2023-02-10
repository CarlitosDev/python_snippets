'''
	linear_regression_manually_GD.py
	
	Just to play around. Approximate the coefficient w by using LMS/gradient descent.

'''

import numpy as np

# Set a simple linear relationship and use gradient descent to calculate the weights w
_w = 7.98
x = np.random.rand(8)
y = x*_w

w = 0.0
# Training
learning_rate = 0.05
n_iters = 1000

for epoch in range(n_iters):
    # this is the derivative dJ/dw, being J = MSE= 1/N*np.mean(np.power(y-wx,2))
    dw = np.mean(2*x*(x*w-y))
    # update weights
    w -= learning_rate * dw
    print(w)


    