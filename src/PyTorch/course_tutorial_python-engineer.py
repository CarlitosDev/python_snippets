'''

	https://www.youtube.com/watch?v=c36lUUr864M
	
	Code:
	https://github.com/python-engineer/pytorchTutorial


'''

# FROM here:

# view reshapes the tensor

# Reshape with torch.view()
x = torch.randn(4, 4)
# concatenate row-wise
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# if -1 it pytorch will automatically determine the necessary size
print(x.size(), y.size(), z.size())





# requires_grad = True -> tracks all operations on the tensor. 
x = torch.randn(3, requires_grad=True)
y = x + 2

# Generally speaking, torch.autograd is an engine for computing vector-Jacobian product
# It computes partial derivates while applying the chain rule



# Differentiation
# 'v': It should be a tensor of matching type and location, that contains the gradient of the differentiated function w.r.t. self.
x = torch.tensor([1.0, 3.0, 7.0], requires_grad=True)
# add a + node to the graph
y = x+2
# add a mult node to the graph
z = y*8
# backward propagation (towards x)
v = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
z.backward(v)
x.grad


# A few considerations:
#
# (1)
# backward() accumulates the gradient for this tensor into .grad attribute.
# !!! We need to be careful during optimization !!!
# Use .zero_() to empty the gradients before a new optimization step!
#
# (2)
# Stop a tensor from tracking history:
# For example during our training loop when we want to update our weights
# then this update operation should not be part of the gradient computation
# - x.requires_grad_(False)
# - x.detach()
# - wrap in 'with torch.no_grad():'
