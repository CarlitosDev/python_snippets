'''
positional_encoding_transformers.py

https://www.tensorflow.org/text/tutorials/transformer

'''


import numpy as np
position = 2048
d_model = 512

# this is 0...2047 size (2048, 1)
pos_vector = np.arange(position)[:, np.newaxis]
# i is every point within the depth 
# (1, 512)
i_vector =  np.arange(d_model)[np.newaxis, :]


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)