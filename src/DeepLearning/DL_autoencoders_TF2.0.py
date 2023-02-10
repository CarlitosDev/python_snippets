'''

    Also, have a look at this notebook: https://github.com/serengil/tensorflow-101/blob/master/python/Autoencoder.ipynb

'''

from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.regularizers import l1
from keras.optimizers import Adam
from keras.utils import to_categorical

import tensorflow as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# reshape 784 = 28x28
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)


# Train the autoencoder

input_size = 784
hidden_size = 128
code_size = 32

input_img = Input(shape=(input_size,))
# In TF2.0 this is done through 'layers'
input_img = tf.keras.Input(shape=(input_size,))

hidden_1 = Dense(hidden_size, activation='relu')(input_img)
# In TF2.0 this is done through 'layers'
hidden_1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.relu)(input_img)

code = Dense(code_size, activation='relu')(hidden_1)
# In TF2.0
code = tf.keras.layers.Dense(units=code_size, activation='relu')(hidden_1)

hidden_2 = Dense(hidden_size, activation='relu')(code)
# In TF2.0
hidden_2 = tf.keras.layers.Dense(units=hidden_size, activation='relu')(code)

output_img = Dense(input_size, activation='sigmoid')(hidden_2)
# In TF2.0
output_img = tf.keras.layers.Dense(input_size, activation='sigmoid')(hidden_2)


autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# the model is trained as y_train=x_train (targets are the same as the inputs)
autoencoder_train = autoencoder.fit(x_train, x_train, epochs=3)

# summary
autoencoder.summary()


# check the loss per epoch
loss = autoencoder_train.history['loss']
#val_loss = autoencoder_train.history['val_loss']



# predict
decoded_imgs = autoencoder.predict(x_test)
print(decoded_imgs.shape)

# To save the model
# autoencoder.save_weights('autoencoder.h5')


# Since the coding layer has a lower dimensionality than
# the input data, the autoencoder is said to be undercomplete.


# There is another way to force the autoencoder to learn useful
# features, which is adding random noise to its inputs and
# making it recover the original noise-free data.

# Get the weights of the 1st hidden layer?
weights = autoencoder.get_weights()[0].T
autoencoder.get_weights()[0].shape

# Evaluate the layers - why do we have 2x intermediate ones?
a_weights = autoencoder.get_weights()
for idx in range(len(a_weights)):
  print(f'Dims {idx} - {len(a_weights[idx])}')


n = 10
plt.figure(figsize=(20, 5))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(weights[i+0].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()









#### Another example where we use a method to
'''
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
'''
from keras.layers.normalization import BatchNormalization
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose

num_classes = 10

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out

encode = encoder(input_img)
full_model = Model(input_img,fc(encode))

for l1,l2 in zip(full_model.layers[:19],autoencoder.layers[0:19]):
    l1.set_weights(l2.get_weights())

for layer in full_model.layers[0:19]:
    layer.trainable = False

train_X,valid_X,train_label,valid_label = train_test_split(train_data,train_Y_one_hot,test_size=0.2,random_state=13)


#train_label -> (48000, 10)

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
classify_train = full_model.fit(train_X, train_label, batch_size=64,epochs=100,verbose=1,validation_data=(valid_X, valid_label))




##### Another example

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

encoding_dim = 8


input_dim = X_train.shape[1] #features
hidden_dim = int(encoding_dim / 2)

nb_epoch = 30
batch_size = 128
learning_rate = 0.1

input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh", 
    activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(hidden_dim, activation='relu')(encoder)
decoder = Dense(encoding_dim, activation='relu')(encoder)
decoder = Dense(input_dim, activation='tanh')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)


autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

autoencoder.fit(X_train, X_train,
        epochs=nb_epoch,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_test, X_test),
        verbose=1)