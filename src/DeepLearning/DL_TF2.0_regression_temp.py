'''

    From here:
    https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_regression.ipynb

    Example of basic regression using TF20/Keras and 


'''

from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# keras is a core part of TF2.0
import tensorflow as tf
print(tf.__version__)


'''
    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)
'''

# option A
data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original",
                   delim_whitespace = True, header=None,
                   names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin', 'Car name'])
print(data.shape)
data = data.dropna()
data.head()


inputVars = ['Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year']
responseVar = 'MPG'



model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(data[inputVars].shape[1],)),
  tf.keras.layers.Dense(64, activation=tf.nn.relu),
  tf.keras.layers.Dense(1)])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mean_absolute_error', 'mean_squared_error'])

model.summary()



model.fit(data[inputVars], data[responseVar], epochs=5)

model.predict(data[inputVars])