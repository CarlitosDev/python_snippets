'''

From https://www.microsoft.com/en-us/research/project/mslr/

Each row corresponds to a query-url pair. 
  The first column is relevance label of the pair. From 0 (irrelevant) to 4 (perfectly relevant).
  the second column is query id. 
  and the following columns are 136 features. 

The larger value the relevance label has, the more relevant the query-url pair is.
A query-url pair is represented by a 136-dimensional feature vector.


https://colab.research.google.com/github/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb



Very fast  →  very slow; Simple method  →  complex method; Low quality  →  high quality.

RMSE
QueryRMSE
PairLogit
PairLogitPairwise
YetiRank
YetiRankPairwise


'''

from catboost import CatBoostRanker, Pool, MetricVisualizer
from copy import deepcopy
import numpy as np
import os
import pandas as pd

from catboost.datasets import msrank_10k
train_df, test_df = msrank_10k()

# each document (user query?) has got 136 features
# column 0 is the label (from 0 to 4)
# column 1 is the query (87 different queries)
# >> basically what I see here is that there are 87 different 
# queries that have returned documents with 136 features and a relevance from 0 to 4.
train_df.head()

X_train = train_df.drop([0, 1], axis=1).values
y_train = train_df[0].values
queries_train = train_df[1].values

X_test = test_df.drop([0, 1], axis=1).values
y_test = test_df[0].values
queries_test = test_df[1].values


# I guess what CBoost will do now is to try to learn 
# the relationship between a query + the features of a document in order to 
# retrieve the highest ranked documents ???


train_df[1].unique()
train_df[1].nunique()

test_df[1].unique()
test_df[1].nunique()

# the majority of queries are irrelevant...
from collections import Counter
Counter(y_train).items()

# normalise 0,1
max_relevance = np.max(y_train)
y_train /= max_relevance
y_test /= max_relevance

train = Pool(
    data=X_train,
    label=y_train,
    group_id=queries_train
)

test = Pool(
    data=X_test,
    label=y_test,
    group_id=queries_test
)

# Attention: all objects in dataset must be grouped by group_id

# For EF Hello??
# groups Q. Types of user interests ("queries"): travelling, business, ...
# documents/items D_q from Q: course1, course2, etc, ...
# labels L_q for D_q. This is the 'relevance', so I guess for EFHello should be
# something like a top number if the user upgraded and then something related to engagement??
# In the input data, the features might be something like age range, etc, ...

# What CBoost does is
'''
  The first and simplest idea is to try predicting document relevance $l_q$ minimizing RMSE.
  $$\frac{1}{N}\sqrt{ \sum_q \sum_{d_{qk}} \left(f(d_{qk}) - l_{qk} \right)^2 }$$
'''

default_parameters = {
    'iterations': 300,
    'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=10'],
    'verbose': True,
    'random_seed': 0,
}

parameters = {}

def fit_model(loss_function, additional_params=None, train_pool=train, test_pool=test):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function
    
    if additional_params is not None:
        parameters.update(additional_params)
        
    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)
    
    return model

model = fit_model('RMSE', {'custom_metric': ['PrecisionAt:top=10', 'RecallAt:top=10', 'MAP:top=10']})



'''
'''
queries = queries_train
query_set = np.unique(queries)
query_weights = np.random.uniform(size=query_set.shape[0])
weights = np.zeros(shape=queries.shape)
for i, query_id in enumerate(query_set):
    weights[queries == query_id] = query_weights[i]
###
# some queries are more important than others for us.
# The word "importance" used here in terms of accuracy or quality of CatBoostRanker prediction for given queries.
# You can pass this additional information for learner using a group_weights parameter.

# Under the hood, CatBoostRanker uses this weights in loss function simply multiplying it on a group summand.
# So the bigger weight → the more attention for query.
# Let's show an example of training procedure with random query weights. 
# >> This must be just to show the possibility of adding weights.

def create_weights(queries):
    query_set = np.unique(queries)
    query_weights = np.random.uniform(size=query_set.shape[0])
    weights = np.zeros(shape=queries.shape)
    
    for i, query_id in enumerate(query_set):
        weights[queries == query_id] = query_weights[i]
    
    return weights
    

train_with_weights = Pool(
    data=X_train,
    label=y_train,
    group_weight=create_weights(queries_train),
    group_id=queries_train
)

test_with_weights = Pool(
    data=X_test,
    label=y_test,
    group_weight=create_weights(queries_test),
    group_id=queries_test
)

fit_model(
    'RMSE', 
    additional_params={'train_dir': 'RMSE_weigths'}, 
    train_pool=train_with_weights,
    test_pool=test_with_weights
)



####
'''

Predicting the top one most relevant object for a given query.
For this purpose CatBoostRanker has a mode called QuerySoftMax.

Suppose our dataset contain a binary target: 1 − mean best document for a query, 0 − others.
We will maximize the probability of being the best document for given query.
MSRANK dataset doesn't contain binary labels, but for example of method QuerySoftMax we convert it to that format,
choosing a best document for every query.

'''

def get_best_documents(labels, queries):
    query_set = np.unique(queries)
    num_queries = query_set.shape[0]
    by_query_arg_max = {query: -1 for query in query_set}
    
    for i, query in enumerate(queries):
        best_idx = by_query_arg_max[query]
        if best_idx == -1 or labels[best_idx] < labels[i]:
            by_query_arg_max[query] = i
    
    binary_best_docs = np.zeros(shape=labels.shape)
    for arg_max in by_query_arg_max.values():
        binary_best_docs[arg_max] = 1.
        
    return binary_best_docs


best_docs_train = get_best_documents(y_train, queries_train)
best_docs_test = get_best_documents(y_test, queries_test)

train_with_weights = Pool(
    data=X_train,
    label=best_docs_train,
    group_id=queries_train,
    group_weight=create_weights(queries_train)
)

test_with_weights = Pool(
    data=X_test,
    label=best_docs_test,
    group_id=queries_test,
    group_weight=create_weights(queries_test)
)

softmax_model = fit_model(
    'QuerySoftMax',
    additional_params={'custom_metric': 'AverageGain:top=1'},
    train_pool=train_with_weights,
    test_pool=test_with_weights
)


x_user = train_df.loc[3:10, 2::]
q_user = queries_test[2:10]
input_data = x_user.values#.reshape(1,-1)
input_group_weight = create_weights(queries_test)[3:11]
# group_id=np.array(q_user).reshape(1,-1),

random_test = Pool(
    data=input_data,
    group_id=q_user,
    group_weight=input_group_weight,
)
# ???
softmax_model.predict(random_test)

model.predict(random_test)