'''

    To do this manually just using Numpy:
        .../PythonSnippets/Optimisation/linear_regression_manually_GD.py

        

'''


# 2 - Now, use a more PyTorch approach.
import torch
import torch.nn as nn
import numpy as np

# Set a simple linear relationship and 
# use gradient descent to calculate the weights w
num_samples = 500
num_test_samples = 100
num_features = 30
threshold = 0.58

x = np.random.rand(num_samples+num_test_samples, num_features).astype(np.float32)
y = (x[:,7] >= threshold).astype(np.float32)



# reshape as column vectors 
# Pytorch expects samples x features
x_tensor = torch.from_numpy(x[0:num_samples])
y_tensor = torch.from_numpy(y[0:num_samples]).view(num_samples,1)

x_tensor.shape
y_tensor.shape


##
# from sklearn import datasets
# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# x_tensor = torch.from_numpy(X_train.astype(np.float32))
# y_tensor = torch.from_numpy(y_train.astype(np.float32)).view(y_train.shape[0], 1)
# ##

n_samples, n_features = x_tensor.shape


# Now we do a trick
# We create a class to represent the model. It inherits 
# from nn.Module and needs a forward method.
# The second trick is to represent logistic reg by 
# a composition of sigmoid and linear layer.
class log_reg(nn.Module):
    def __init__(self, n_input_features):
        super(log_reg, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = log_reg(n_features)


# PyTorch fnc for binary cross-entropy, which is just the loss defined for log reg
# (the one that comes from d_likelihood/dw)
loss = nn.BCELoss()

# instead of optimising the weights manually, let's use the SGD function
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_iters = 250
for epoch in range(n_iters):

  # predict = forward pass with our model
  y_predicted = model(x_tensor)
  # loss
  l = loss(y_predicted, y_tensor)
  # calculate gradients = backward pass
  l.backward()
  # update weights
  optimizer.step()
  # zero the gradients after updating
  optimizer.zero_grad()

  if epoch % 10 == 0:
    print(f'epoch: {epoch}, loss = {l.item():.4f}')


x_test = torch.from_numpy(x[num_samples::]).requires_grad_(False)
y_test = torch.from_numpy(y[num_samples::])

y_predicted = model(x_test).detach()
y_predicted_cls = y_predicted.round()
acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
print(f'accuracy: {acc.item():.4f}')




## the same with a neural net.
# kind of two logistic regressions back to back
# (the first layer uses RELU instead of sigmoid)

# Binary classification
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)  
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred


nn_model = NeuralNet1(input_size=n_features, hidden_size=4)
loss = nn.BCELoss()

# instead of optimising the weights manually, let's use the SGD function
learning_rate = 0.01
optimizer_nn = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)

n_iters = 250
for epoch in range(n_iters):

  # predict = forward pass with our model
  y_predicted = nn_model(x_tensor)
  # loss
  l = loss(y_predicted, y_tensor)
  # calculate gradients = backward pass
  l.backward()
  # update weights
  optimizer_nn.step()
  # zero the gradients after updating
  optimizer_nn.zero_grad()

  if epoch % 10 == 0:
    print(f'epoch: {epoch}, loss = {l.item():.4f}')


y_predicted = nn_model(x_test).detach()
y_predicted_cls = y_predicted.round()
acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
print(f'accuracy: {acc.item():.4f}')