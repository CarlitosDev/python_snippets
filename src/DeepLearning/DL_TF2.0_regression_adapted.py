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

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']


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
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original",
                   delim_whitespace = True, header=None,
                   names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin', 'Car name'])
print(df.shape)
df.dropna(inplace=True)
df.head()


origin = df.pop('Origin')
df['USA'] = (origin == 1)*1.0
df['Europe'] = (origin == 2)*1.0
df['Japan'] = (origin == 3)*1.0
df.tail()




responseVar = 'MPG'
inputVars   = ['Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'USA', 'Europe', 'Japan']

numRecords = df.shape[0]

# checkpoint


# standard scaling
basic_stats = df[inputVars].describe().transpose()
df_norm = (df[inputVars] - basic_stats['mean']) / basic_stats['std']
df_norm[responseVar] = df[responseVar]
df_norm.head()




# get datasets
# training
X = df_norm.iloc[0:320][inputVars].values
y = df_norm.iloc[0:320][responseVar].values

# test
X_test = df_norm.iloc[320::][inputVars].values
y_test = df_norm.iloc[320::][responseVar].values



model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation=tf.nn.relu, input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error', 'mean_squared_error'])



'''
model_b = tf.keras.Sequential()
model_b.add(tf.keras.layers.InputLayer(batch_input_shape=(X.shape[1],)))
# this one fails...
model_b.add(tf.keras.layers.Dense(8, input_shape=(X.shape[1],), activation=tf.nn.relu))

'''

# summary
model.summary()

# Train
history = model.fit(X, y, epochs=100)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# Get forecast parameters
loss, mae, mse = model.evaluate(X_test, y_test, verbose=1)

#model.to_json()

y_hat = model.predict(X_test)
y_test

# Scatter plot
plt.scatter(y_hat, y_test)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show(block=False)



# Get the weights of the 1st hidden layer?
weights = model.get_weights()[0].T
model.get_weights()[0].shape

# Evaluate the layers - why do we have 2x intermediate ones?
a_weights = model.get_weights()
for idx in range(len(a_weights)):
  print(f'Dims {idx} - {len(a_weights[idx])}')


'''
for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())
'''


## Not sure about this...
# I guess the activation is the input times the weights of the layer
weights = model.get_weights()
numLayers = len(weights)


one_row = X_test[0]


# LAYER 0
w_layer_0 = weights[0]
# 9x8
w_layer_0.shape
# activation of the layer 0
activation_layer_0 = np.dot(one_row, w_layer_0)

# LAYER 1
w_layer_1 = weights[1]
# 9x8
w_layer_1.shape

activation_layer_1 = np.dot(activation_layer_0, w_layer_1)