'''
SHAP (SHapely Additive exPlanations)
https://medium.com/fiddlerlabs/case-study-explaining-credit-modeling-predictions-with-shap-2a7b3f86ec12


Based on the Shapley values from game theory, which are the only 
explanations within a broad class of possibilities that satisfy 
three useful properties, termed by Lundberg et al.:

1. local accuracy (roughly: the explanation model 
matches the real model at the point being explained)

2. missingness (Lundberg: “missingness requires
features missing in the original input to have no impact”)

3. consistency (Lundberg: “Whenever we change a model such
 that it relies more on a feature, then the attributed
 importance for that feature should not decrease.”)


https://github.com/slundberg/shap


More info:
---------

SHapley Additive exPlantions (SHAP)[1]. It is introduced by Lundberg et al. 

https://towardsdatascience.com/interpreting-your-deep-learning-model-by-shap-e69be2b47893

Shapley value which is a solution concept in cooperative game theory.
It is based on Shapley values, a technique used in game theory to determine
how much each player in a collaborative game has contributed to its success

The idea is using game theory to interpret target model. 
All features are “contributor” and trying to predict the task which is 
“game” and the “reward” is actual prediction minus the result from explanation model.


SHAP provides multiple explainers for different kind of models:

TreeExplainer: Support XGBoost, LightGBM, CatBoost and scikit-learn models by Tree SHAP.
DeepExplainer (DEEP SHAP): Support TensorFlow and Keras models by using DeepLIFT and Shapley values.
GradientExplainer: Support TensorFlow and Keras models.
KernelExplainer (Kernel SHAP): Applying to any models by using LIME and Shapley values.

Installing:

brew install snappy
pip3 install python-snappy


Shapley values
https://en.wikipedia.org/wiki/Shapley_value

Explanation of the order of the players: https://math.stackexchange.com/questions/111580/shapley-value-formula



To read: 
https://towardsdatascience.com/one-feature-attribution-method-to-supposedly-rule-them-all-shapley-values-f3e04534983d


https://medium.com/datadriveninvestor/time-step-wise-feature-importance-in-deep-learning-using-shap-e1c46a655455
https://github.com/slundberg/shap

https://www.kaggle.com/learn/machine-learning-explainability


https://github.com/slundberg/shap
https://www.kaggle.com/dansbecker/shap-values
https://www.kaggle.com/scratchpad/kernel67f4e74941/edit
https://towardsdatascience.com/what-70-of-data-science-learners-do-wrong-ac35326219e4
https://shap.readthedocs.io/en/latest/#plots


Microsoft's interpretability library
https://github.com/microsoft/interpret




'''
###


'''
SHapley Additive exPlantions (SHAP)[1]. It is introduced by Lundberg et al. 


https://towardsdatascience.com/interpreting-your-deep-learning-model-by-shap-e69be2b47893

Shapley value which is a solution concept in cooperative game theory.


The idea is using game theory to interpret target model. 
All features are “contributor” and trying to predict the task which is 
“game” and the “reward” is actual prediction minus the result from explanation model.


SHAP provides multiple explainers for different kind of models:

TreeExplainer: Support XGBoost, LightGBM, CatBoost and scikit-learn models by Tree SHAP.
DeepExplainer (DEEP SHAP): Support TensorFlow and Keras models by using DeepLIFT and Shapley values.
GradientExplainer: Support TensorFlow and Keras models.
KernelExplainer (Kernel SHAP): Applying to any models by using LIME and Shapley values.

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
import seaborn as sns; sns.set()

from mpl_toolkits.mplot3d import Axes3D



# load JS visualization code to notebook
shap.initjs()

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






dataRoot  = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/PythonDevs (nextDoorDev)/data'
dataFile  = os.path.join(dataRoot, 'fakeArray.mat')
matData   = sio.loadmat(dataFile)

data     = matData['fakeArray']
varNames = [f[0] for f in matData['varNames'][0]]
df   = pd.DataFrame(data, columns=varNames)

numRecords = df.shape[0]


# input vars
inputVars   = ['num1', 'num2', 'confoundingVar', 
'c1_0', 'c1_1', 'c2_0', 'c2_1']
responseVar = 'responseVar'

X, y = df[inputVars],df[responseVar]


xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

xgb_model.fit(X,y)

explainer = shap.TreeExplainer(xgb_model, data=X)
shap_values = explainer.shap_values(X)

shap_values = explainer.shap_values(X.iloc[1])

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
idx_instance = 1
print(y[idx_instance])
shap.force_plot(explainer.expected_value, shap_values[idx_instance,:],  
X.iloc[idx_instance].values, matplotlib=True, feature_names=inputVars, text_rotation=45)


#shap.DeepExplainer


predictions = xgb.predict(X_test)
print(explained_variance_score(predictions,y_test))




# get datasets
X = df.iloc[0:300][inputVars].values
Y = df.iloc[0:300][responseVar].values

# validate
X_val = df.iloc[300:400][inputVars].values
y_val = df.iloc[300:400][responseVar].values

# test
X_test = df.iloc[400::][inputVars].values
y_test = df.iloc[400::][responseVar].values


###########


# Fake sales
num_samples = 800

x3_values = np.random.randint(100,200, size=(num_samples))
print(f'Mean value {x3_values.mean()} \pm {x3_values.std()}')

x4_values = np.random.randint(50,100, size=(num_samples))

x3_values[-1] = 0.0
x4_values[-1] = 0.0

noisy_sales = 10 * np.random.random(size=(num_samples)) + x3_values - x4_values

df = pd.DataFrame({
    "x1" : np.random.randint(0, 100, size=(num_samples)),
    "x2" : np.random.randint(0, 200, size=(num_samples)),
    "x3" : x3_values,
    "x4" : x4_values,
    "sales": noisy_sales
})


df_correlations = df.corr()
print(df_correlations['sales'])

df['sales'].mean()


cmap = sns.diverging_palette(20, 220, n=200)
# Plot the histograms
def plot_SHAP_histogram(x_values, shap_values, phi_values, varName):
  f, axes = plt.subplots(nrows=1, ncols=3)
  sns.distplot(x_values, ax=axes[0])
  axes[0].set_title(f'x{idx}')
  sns.distplot(shap_values, ax=axes[1])
  axes[1].set_title(f"Shap values for {varName}")
  sns.distplot(phi_values, ax=axes[2])
  axes[2].set_title(f"Phi values for {varName}")
  plt.show(block = False)

#cmap = sns.light_palette((210, 90, 60), input="husl")
ax = sns.heatmap(
    df_correlations, 
    vmin=-1, vmax=1, center=0,
    cmap=cmap,
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()

## Model
# input vars
inputVars   = ['x1', 'x2', 'x3', 'x4']
responseVar = 'sales'


# get datasets
X = df.iloc[0:500][inputVars]
y = df.iloc[0:500][responseVar]

# test
X_test = df.iloc[500::][inputVars]
y_test = df.iloc[500::][responseVar]


xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, 
                             gamma=0, subsample=0.75,
                             colsample_bytree=1, max_depth=7)

xgb_model.fit(X,y)

'''
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['x3'], X_test['x4'], y_test, marker='o')
ax.set_xlabel('x3')
ax.set_ylabel('x4')
ax.set_zlabel('y')
plt.show()
'''


# Has the model picked the relationship between the input variables?
approximate = False
y_hat = xgb_model.predict(X_test)
print(f'Average prediction {y_hat.mean()}')




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['x3'], X_test['x4'], y_test, marker='o')
ax.scatter(X_test['x3'], X_test['x4'], y_hat, marker='o')
ax.set_xlabel('x3')
ax.set_ylabel('x4')
ax.set_zlabel('y')
plt.show()

# Get the type of model
# str(type(xgb_model))
#self.expected_value = self.model.predict(self.data, output=model_output).mean(0)
#xgb_model.predict(X_test)
phi_0  = y_hat.mean()

explainer = shap.TreeExplainer(xgb_model)
#explainer.expected_value = 0.0
shap_values = explainer.shap_values(X_test)
shap_interaction_values = explainer.shap_interaction_values(X_test)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
idx_instance = -1
idx_instance = 8

shap.force_plot(explainer.expected_value, shap_values[idx_instance,:],  
X_test.iloc[idx_instance].values, matplotlib=True, feature_names=inputVars, text_rotation=45)

shap_values[idx_instance,:]
X_test.iloc[idx_instance]
y_test.iloc[idx_instance]

# 'Manual' explanation
# Shap values: A_i = phi_i \cdot x_i
A = shap_values[idx_instance,:]
C = np.sum(A)+explainer.expected_value
print(f'''\tSHAP computation {C:3.2f}
\tActual value {y_test.iloc[idx_instance]:3.2f}
\tPredicted value {y_hat[idx_instance]:3.2f}''')

shap_interaction_values[idx_instance, :, :]


x3_shap = shap_values[:,2]
x4_shap = shap_values[:,3]

phi_3 = x3_shap/X_test['x3'].values
phi_4 = x4_shap/X_test['x4'].values


idx = 3
plot_SHAP_histogram(X_test[f'x{idx}'], shap_values[:,idx-1], 
shap_values[:,idx-1]/(X_test[f'x{idx}'].values+1), f'X{idx}')



# Case where x3 and x4 are 0
x_treme = np.append(X.loc[ idx_instance, ['x1', 'x2']].values, [0.0, 0.0])
shap_values = explainer.shap_values()

shap.force_plot(explainer.expected_value, shap_values[idx_instance,:],  
, matplotlib=True, feature_names=inputVars, text_rotation=45)


############
# pip3 install shap --upgrade
#
# Fake sales
num_samples = 800

x3_values = np.random.randint(100,200, size=(num_samples))
print(f'Mean value {x3_values.mean()} \pm {x3_values.std()}')

x4_values = np.random.randint(50,100, size=(num_samples))

x3_values[-1] = 0.0
x4_values[-1] = 0.0

noisy_sales = 10 * np.random.random(size=(num_samples)) + x3_values - x4_values

df = pd.DataFrame({
    "x1" : np.random.randint(0, 100, size=(num_samples)),
    "x2" : np.random.randint(0, 200, size=(num_samples)),
    "x3" : x3_values,
    "x4" : x4_values,
    "sales": noisy_sales
})


## Model
# input vars
inputVars   = ['x1', 'x2', 'x3', 'x4']
responseVar = 'sales'


# get datasets
X = df.iloc[0:500][inputVars]
y = df.iloc[0:500][responseVar]

# test
X_test = df.iloc[500::][inputVars]
y_test = df.iloc[500::][responseVar]


from catboost import CatBoostRegressor, Pool


# General hyperparameters
idx_categorical_features = None

num_iterations=100
learning_rate=0.08
depth=12
cat_features=idx_categorical_features
loss_function='RMSE'


# CatBoost on SKU A with SKU A data
cb_model= CatBoostRegressor(iterations=num_iterations, learning_rate=learning_rate, \
                             depth=depth, loss_function=loss_function, \
                             cat_features=idx_categorical_features, silent=False)

cb_model.fit(X,y, verbose=50)


y_hat = cb_model.predict(X_test)

phi_0  = y_hat.mean()

explainer = shap.TreeExplainer(cb_model)

shap_values = explainer.shap_values(X_test)
# This doesn't work
shap_interaction_values = explainer.shap_interaction_values(X_test, tree_limit=10)


# 'Manual' explanation
idx_instance = 8
# Shap values: A_i = phi_i \cdot x_i
A = shap_values[idx_instance,:]
C = np.sum(A)+explainer.expected_value
print(f'''\tSHAP computation {C:3.2f}
\tActual value {y_test.iloc[idx_instance]:3.2f}
\tPredicted value {y_hat[idx_instance]:3.2f}''')

# shap_interaction_values[idx_instance, :, :]

# How does CatBoost calculate this internally?
# np.ndarray of shape (n_objects, n_features + 1) with Shap values (float) for (object, feature).
test_pool = Pool(X_test, y_test)
v = cb_model.get_feature_importance(test_pool, type='ShapValues', prettified=True)

# np.array of shape (n_objects, n_features + 1, n_features + 1) with
# Shap interaction values (float) for (object, feature(i), feature(j)).
# 300 x 5 x 5
vi = cb_model.get_feature_importance(test_pool, type='ShapInteractionValues', prettified=True)

#cb_model._object._calc_fstr(type.name, pool, thread_count, 
# verbose, shap_mode, interaction_indices, shap_calc_type)

from . import _catboost
_CatBoost = _catboost._CatBoost


# this is from catboost/_catboost.pyx
if type_name == 'ShapValues' and dereference(self.__model).GetDimensionsCount() > 1:
            with nogil:
                fstr_multi = GetFeatureImportancesMulti(
                    fstr_type,
                    dereference(self.__model),
                    dataProviderPtr,
                    referenceDataProviderPtr,
                    thread_count,
                    shap_mode,
                    verbose,
                    calc_type,
                    model_output
                )
    return _3d_vector_of_double_to_np_array(fstr_multi), native_feature_ids
# GetFeatureImportancesMulti is defined in calc_str.cpp
# and it subsequently calls 
# CalcShapValuesMulti(model, *dataset, referenceDataset, 
# /*fixedFeatureParams*/ Nothing(), logPeriod, mode, &localExecutor,
# calcType, modelOutputType);
#
# Then we get to shap_values.cpp









# Round decimals for the SHAP plot
numericalVars = cannibalisation_test_set[input_varnames].select_dtypes(include=['number']).columns.tolist()
cut_off_decimals = lambda x_in: np.around(x_in, decimals=2)
cannibalisation_test_set[numericalVars] = cannibalisation_test_set[numericalVars].applymap(cut_off_decimals)

X_cannibalisation = cannibalisation_test_set[input_varnames].values

plot_name = 'cannibalisation'
idx = 20
shap.force_plot(explainer.expected_value, shap_values[idx,:], X_cannibalisation[idx,:], 
               feature_names=input_varnames, text_rotation=25,show=False,matplotlib=True)#.savefig()

plt_filename = 'shap_plots/' + plot_name + '.pdf'
plt.savefig(plt_filename, format='pdf', dpi=300, bbox_inches='tight')

