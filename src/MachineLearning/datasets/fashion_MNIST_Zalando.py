'''
bash
git clone git@github.com:zalandoresearch/fashion-mnist.git
cd '/Users/carlos.aguilar/Documents/DS_repos/fashion-mnist/utils'
python3
'''


import numpy as np
import mnist_reader
data_folder = './../data/fashion'
X_train, y_train = mnist_reader.load_mnist(data_folder, kind='train')
X_test, y_test = mnist_reader.load_mnist(data_folder, kind='t10k')


data = np.array(np.vstack([X_train, X_test]), dtype=np.float64) / 255.0

# The images are 28x28, so 784
# (70000, 784)
# data.shape


target = np.hstack([y_train, y_test])
# (70000,)
#target.shape
np.unique(target)
# [0,10]

'''
Some tutorials using this dataset:

# Siamese networks
https://github.com/adambielski/siamese-triplet/blob/master/Experiments_FashionMNIST.ipynb
# UMAP: https://umap-learn.readthedocs.io/en/latest/supervised.html#training-with-labels-and-embedding-unlabelled-test-data-metric-learning-with-umap
'''