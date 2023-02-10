'''

    To do this manually just using Numpy:
        .../PythonSnippets/Optimisation/linear_regression_manually_GD.py


    From:

	https://www.youtube.com/watch?v=c36lUUr864M
	
	Code:
	https://github.com/python-engineer/pytorchTutorial
        

'''


# 1- Using Pytorch's autograd (instead of calculating it by hand)

import torch
import numpy as np

# Set a simple linear relationship and 
# use gradient descent to calculate the weights w
_w = 7.98
x = np.random.rand(4).astype(np.float32)
y = x*_w

X = torch.from_numpy(x)
Y = torch.from_numpy(y)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Training
learning_rate = 0.05
n_iters = 100

for epoch in range(n_iters):
    # calculate the MSE directly
    l = ((w*X - Y)**2).mean()
    l.backward()
    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # zero the gradients after updating
    w.grad.zero_()
    print(w.item())




# 2 - Now, use a more PyTorch approach.
import torch.nn as nn

# reshape as column vectors 
# Pytorch expects samples x features
x_tensor = torch.from_numpy(x).view(-1,1)
y_tensor = torch.from_numpy(y).view(-1,1)

x_tensor.shape

n_samples, n_features = x_tensor.shape

# given that we're modelling regression,
# we can represent it as a linear layer
# mind that the dims are the features and it will take 
# in any samples
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)


# PyTorch fnc for MSE loss
loss = nn.MSELoss()

# instead of optimising the weights manually, let's use the SGD function
learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_iters = 250
for epoch in range(n_iters):
 
    # predict = forward pass with our model
    y_predicted = model(x_tensor)
    # loss
    l = loss(y_tensor, y_predicted)
    # calculate gradients = backward pass
    l.backward()
    # update weights
    optimizer.step()
    # zero the gradients after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters() # unpack parameters
        print('epoch ', epoch, ': w = ', w[0][0].item(), ' loss = ', l)

# the model outputs weight and bias
[w, b] = model.parameters()
print(f'Model after {n_iters} iterations w={w.item():3.2f} and b={b.item():3.2f}')