'''

    To do this manually just using Numpy:
        .../PythonSnippets/Optimisation/linear_regression_manually_GD.py


    From:

	https://www.youtube.com/watch?v=c36lUUr864M
	
	Code:
	https://github.com/python-engineer/pytorchTutorial
        

'''


# Notes:
# copied from here: https://github.com/python-engineer/pytorchTutorial/blob/master/09_dataloader.py
# Gist -> divide dataset into small batches to make the training efficient
#
# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas as pd
import carlos_utils.data_utils as du

# So any datafeed can be instanciated as a Dataset by just building a class that 
# inherits from Dataset. The nice features come from combining Dataset with DataLoader 
# as we get parallelisation, iterators, etc off the shelf.
class PandasDataset(Dataset):

    def __init__(self, df, input_vars, target_var):
        # Initialize data, download, etc.
        # read with numpy or pandas
        self.n_samples = df.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(df[input_vars].values) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(df[target_var].values) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples



# This code is just to fetch info from Snowflake and see if I can read it 
# as a Pytorch.Dataset
current_database= 'EDTECH_DS_DEV'
system_prefix = 'ENGLISHLIVE'

# number of days to consider for the bucket
number_of_days = 30

prefix_days = f'F{number_of_days}D'
features_table = f'{prefix_days}_STUDENT_FEATURE_STORE'

current_environment = f'{system_prefix}_LEARNER_PROFILE'
sfk_connection = du.edtech_ds_dev_get_connection_snowflake(_schema=current_environment)
sqlQuery = f'''SELECT * FROM {current_database}.{current_environment}.{features_table} 
where product_category is not null
and region_group != 'zzz_Unknown' 
'''
df_student_features = pd.read_sql(sqlQuery, sfk_connection)
sfk_connection.close()

df_student_features.iloc[2]

df_student_features.columns

input_vars = ['first_day_since_enrollment', 'latest_active_day_step',
       'how_frequent_step', 'num_student_items_step',
       'total_student_minutes_step', 'avg_student_grade_step',
       'num_active_days_class', 'first_day_since_enrollment_class',
       'latest_active_day_class']
target_var = 'active_beyond_min_days'




# Instanciate ;)
student_features = PandasDataset(df_student_features, \
  input_vars, target_var)



features, labels = student_features[2]
print(features, labels)

# Turn it into a DataLoader object
batch_size=780
train_loader = DataLoader(dataset=student_features,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# convert to an iterator and look at one random sample
dataiter = iter(train_loader)
data = dataiter.next()
features, labels = data
print(features, labels)

student_features.__len__()
# this is the same as len(student_features)
# 26874



# Dummy Training loop
num_epochs = 2
total_samples = len(student_features)
n_iterations = math.ceil(total_samples/batch_size)

print(total_samples, n_iterations)
for epoch in range(num_epochs):
  for i, (inputs, labels) in enumerate(train_loader):  
    if i % 5 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')

'''

From here: https://github.com/python-engineer/pytorchTutorial/blob/master/10_transformers.py

Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''


class PandasDatasetWithTransform(Dataset):

    def __init__(self, df, input_vars, target_var, transform=None):
        # Initialize data, download, etc.
        # read with numpy or pandas
        self.n_samples = df.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = df[input_vars].values
        self.y_data = df[target_var].values

        self.transform = transform


    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        if self.transform:
          sample = self.transform(sample)
        return sample 

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# Transforms: implement the call method

# Custom Transforms
# implement __call__(self, sample)
class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets.astype(float))

print('Without Transform')
features, labels = student_features[0]
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor Transform')
student_features_transform = PandasDatasetWithTransform(df_student_features, \
  input_vars, target_var, transform=ToTensor())

features, labels = student_features_transform[1]
print(type(features), type(labels))
print(features, labels)        




# Softmax applies the exponential function to each element, and normalizes
# by dividing by the sum of all these exponentials
# -> squashes the output to be between 0 and 1 = probability
# sum of all probabilities is 1
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)


# Cross entropy
# Cross-entropy loss, or log loss, measures the performance of a classification model 
# whose output is a probability value between 0 (better) and 1. 
# -> loss increases as the predicted probability diverges from the actual label
def cross_entropy(actual, predicted):
    EPS = 1e-15
    predicted = np.clip(predicted, EPS, 1 - EPS)
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

# y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')



import torch.nn as nn
# CrossEntropyLoss in PyTorch (applies Softmax)
# nn.LogSoftmax + nn.NLLLoss
# NLLLoss = negative log likelihood loss
loss = nn.CrossEntropyLoss()
# loss(input, target)

# target is of size nSamples = 1
# each element has class label: 0, 1, 2 or 3 
# Y (=target) contains class labels, NOT one-hot
Y = torch.tensor([3])

# input is of size nSamples x nClasses = 1 x 4
# y_pred (=input) must be raw, unnormalizes scores (logits) for each class, not softmax
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1, 2.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3, 1.7]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'PyTorch Loss1: {l1.item():.4f}')
print(f'PyTorch Loss2: {l2.item():.4f}')

# get predictions
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y}, Y_pred1: {predictions1}, Y_pred2: {predictions2}')