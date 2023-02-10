
'''
https://autogluon.mxnet.io/tutorials/tabular_prediction/tabular-indepth.html


Installation

brew install libomp
python3 -m pip install --upgrade "mxnet<2.0.0"
python3 -m pip install --pre autogluon

python3 -m pip install autogluon --upgrade


AutoGluon is accessible as open source but also free of cost through AWS Sagemaker marketplace.

This video contains good explanations from the creator about the no need for hyper-opt and how they get around with model ensembling. 
https://www.youtube.com/watch?v=BzVg7zMSwNY (Around here https://youtu.be/BzVg7zMSwNY?t=1205)


Updates:
    12.03.2021 - Reviewing AutoGluon.

'''

import autogluon as ag
from autogluon import TabularPrediction as task


import pandas as pd
import numpy as np




experiment_label = 'linear_model'
num_samples = 500
num_features = 5
input_vars = [f'x_{idx}' for idx in range(1,num_features+1)]
input_data = np.random.rand(num_samples, num_features)

weights = np.array([13,9,6,1,0])
y_train = np.dot(input_data, weights.T)

df = pd.DataFrame(input_data, columns=input_vars)
df['target'] = y_train

# AutoGluon format
train_data = task.Dataset(df=df)




# Ad-hoc test set to see the influence of the variables
df_test = pd.DataFrame([{'x_1': 0.1, 'x_2': 0.5, 'x_3': 0.5, 'x_4': 0.5, 'x_5': 0.5},
{'x_1': 0.9, 'x_2': 0.5, 'x_3': 0.5, 'x_4': 0.5, 'x_5':0.5 },
{'x_1': 0.5, 'x_2': 0.9, 'x_3': 0.5, 'x_4': 0.5, 'x_5':0.5 },
{'x_1': 0.5, 'x_2': 0.1, 'x_3': 0.5, 'x_4': 0.5, 'x_5':0.5 }])
y_test = np.dot(df_test.values, weights.T)

df_test['target'] = y_test

test_data = task.Dataset(df=df_test)


model = task.fit(train_data=train_data, output_directory="auto_gluon", label='target', hyperparameter_tune=False)

results = model.fit_summary()

model.leaderboard()

model.feature_types

y_pred = model.predict(df_test)
print("Predictions:  ", y_pred)
perf = model.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

dir(model)




##### 
# Try the hyperparameter optimisation method
gbm_options = { # specifies non-default hyperparameter values for lightGBM gradient boosted trees
    'num_boost_round': 100, # number of boosting rounds (controls training time of GBM models)
    'num_leaves': ag.space.Int(lower=26, upper=30, default=28), # number of leaves in trees (integer hyperparameter)
}

hyperparameters = {'GBM': gbm_options}  # hyperparameters of each model type

search_strategy = 'skopt'  # to tune hyperparameters using SKopt Bayesian optimization routine


# If one of these keys is missing from hyperparameters dict, then no models of that type are trained.
model = task.fit(train_data=train_data, output_directory="auto_gluon", 
label='target', hyperparameter_tune=True,
hyperparameters=hyperparameters,
search_strategy=search_strategy)
results = model.fit_summary()

model.leaderboard()



### Load model from folder
import autogluon as ag
from autogluon import TabularPrediction as task
experiment_label = 'grocery'
autogluon_model_folder = '.auto_gluon_' + experiment_label


model = task.load(autogluon_model_folder)
model.model_names


for varName in categoricalVars:
    df_train[varName] = df_train[varName].astype(str)
    df_test[varName] = df_test[varName].astype(str)

test_data = task.Dataset(df=df_test)
results = model.leaderboard(test_data)

autogluon_frc = model.predict(test_data)

model_to_use = model.model_names[1]
model_pred = model.predict(test_data, model=model_to_use)
d = {model_to_use: model.predict(test_data, model=model_to_use) for model_to_use in model.model_names}


perf = model.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)



###

/usr/local/lib/python3.7/site-packages/autogluon/utils/tabular/features/auto_ml_feature_generator.py

from autogluon.utils.tabular.features.auto_ml_feature_generator import AutoMLFeatureGenerator

#??
auto_feats = AutoMLFeatureGenerator()
df_feats = auto_feats.generate_features(df_train)

#df_feats = auto_feats.generate_features(task.Dataset(df=df_train))
df_2 = auto_feats.preprocess(df_train)


from .abstract_learner import AbstractLearner
DefaultLearner()

x_train_path = '/Users/carlosAguilar/Google Drive/order/Machine Learning Part/contrastive explanations/auto_gluon/grocery/utils/data/X_train.pkl'
fhelp.



### Notes
Details regarding the hyperparameters you can specify for each model are provided in the following files:
    NN: `autogluon/utils/tabular/ml/models/tabular_nn/hyperparameters/parameters.py`
        Note: certain hyperparameter settings may cause these neural networks to train much slower.
    GBM: `autogluon/utils/tabular/ml/models/lgb/hyperparameters/parameters.py`
         See also the lightGBM docs: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    CAT: `autogluon/utils/tabular/ml/models/catboost/hyperparameters/parameters.py`
         See also the CatBoost docs: https://catboost.ai/docs/concepts/parameter-tuning.html
    RF: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        Note: Hyperparameter tuning is disabled for this model.
        Note: 'criterion' parameter will be overridden. Both 'gini' and 'entropy' are used automatically, training two models.
    XT: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        Note: Hyperparameter tuning is disabled for this model.
        Note: 'criterion' parameter will be overridden. Both 'gini' and 'entropy' are used automatically, training two models.
    KNN: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Note: Hyperparameter tuning is disabled for this model.
        Note: 'weights' parameter will be overridden. Both 'distance' and 'uniform' are used automatically, training two models.
    LR: `autogluon/utils/tabular/ml/models/lr/hyperparameters/parameters.py`
        Note: a list of hyper-parameters dicts can be passed; each set will create different version of the model.
        Note: Hyperparameter tuning is disabled for this model.
        Note: 'penalty' parameter can be used for regression to specify regularization method: 'L1' and 'L2' values are supported.



lr_options = hyperparameters.get('LR', None)
gbm_options = hyperparameters.get('GBM', None)
nn_options = hyperparameters.get('NN', None)
cat_options = hyperparameters.get('CAT', None)
rf_options = hyperparameters.get('RF', None)
xt_options = hyperparameters.get('XT', None)
knn_options = hyperparameters.get('KNN', None)
custom_options = hyperparameters.get('custom', None)