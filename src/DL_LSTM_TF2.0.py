'''

  https://towardsdatascience.com/generating-text-with-tensorflow-2-0-6a65c7bdc568

'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import datetime
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('TensorFlow version: ', tf.__version__)