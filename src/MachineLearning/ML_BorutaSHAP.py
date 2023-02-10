'''
  pip3 install BorutaShap
  pip3 install shap --upgrade

https://github.com/Ekeany/Boruta-Shap

'''
import pandas as pd
import numpy as np
from scipy.optimize import nnls
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
from sklearn.metrics import explained_variance_score

import xgboost as xgb
import shap

import matplotlib.pyplot as plt



# train XGBoost model
X,y = shap.datasets.boston()
model = xgb.train({"learning_rate": 0.01}, xgb.DMatrix(X, label=y), 100)



# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:], matplotlib=True)


# visualize the training set predictions
shap.force_plot(explainer.expected_value, shap_values, X)


###### Boruta-SHAP
from BorutaShap import BorutaShap

# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(importance_measure='shap',
                              classification=False)
# Also gini impurity
# importance_measure='gini'

Feature_Selector.fit(X=X, y=y, n_trials=5, random_state=0)

# Returns Boxplot of features
Feature_Selector.plot(which_features='all', 
                      X_size=8, figsize=(12,8),
                      y_scale='log')

# Feature_Selector.TentativeRoughFix()

# Returns a subset of the original data with the selected features
subset = Feature_Selector.Subset()
subset.head()


'''
To use CB
model = CatBoostClassifier()
# no model selected default is Random Forest, if classification is False it is a Regression problem
Feature_Selector = BorutaShap(model=model,
                              importance_measure='shap',
                              classification=True)
'''