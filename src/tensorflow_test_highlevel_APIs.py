'''

https://www.tensorflow.org/guide/premade_estimators

We strongly recommend writing TensorFlow programs with the following APIs:

Estimators, which represent a complete model. The Estimator API provides methods to train the model, 
to judge the model's accuracy, and to generate predictions. An Estimator is any class derived from 
tf.estimator.Estimator. TensorFlow provides a collection of tf.estimator (for example, LinearRegressor)
to implement common ML algorithms. Beyond those, you may write your own custom Estimators. 
We recommend using pre-made Estimators when just getting started.



Datasets for Estimators, which build a data input pipeline. The Dataset API has methods to load 
and manipulate data, and feed it into your model. The Dataset API meshes well with the Estimators API.

'''


# To avoid problems with version 2 (not recommended)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def forward(x):
  return W * x + b

out_a = forward([1,0])
print(out_a)


import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.04),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])

train_data = tf.ones(shape=(1, 28, 28, 1))
test_data = tf.ones(shape=(1, 28, 28, 1))

train_out = model(train_data, training=True)
print(train_out)