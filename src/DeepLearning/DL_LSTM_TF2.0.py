'''

  https://towardsdatascience.com/generating-text-with-tensorflow-2-0-6a65c7bdc568

'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import datetime
from random import randint
import pandas as pd
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('TensorFlow version: ', tf.__version__)


# From here https://www.kaggle.com/shivamb/how-autoencoders-work-intro-and-usecases

n_unique = 51
inp_size = 6
out_size = 3

# [3, 40, 11, 26, 46, 32]
inp_seq = [randint(1, n_unique-1) for _ in range(inp_size)]

## create target sequence
# [3, 40, 11]
target = inp_seq[:out_size]

## create padded sequence / seed sequence 
target_seq = list(reversed(target))
seed_seq = [0] + target_seq[:-1] 
# [0, 11, 40]