'''
  pytorch_basics.py
  
'''

import torch


vector_A = [4, 2, 1]
vector_B = [2, 9, 1]

tensor_A = torch.Tensor(vector_A)
tensor_B = torch.Tensor(vector_B)

dot_product = torch.dot(tensor_A, tensor_B)

mat_product = tensor_A*tensor_B

# check if the reshape method is still called view??
# called reshape now
mat_product.reshape(3, 1)