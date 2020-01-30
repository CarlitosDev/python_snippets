'''

	https://github.com/stanfordmlgroup/ngboost
	https://stanfordmlgroup.github.io/projects/ngboost/
	https://towardsdatascience.com/interpreting-the-probabilistic-predictions-from-ngboost-868d6f3770b2

	Paper: https://arxiv.org/abs/1910.03225

	pip3 install --upgrade git+https://github.com/stanfordmlgroup/ngboost.git


	* NGBoost does not return confidence intervals. The confusion lies in the difference between a confidence interval and a prediction interval.
	* NGBoost just tells you the relative likelihood of observing each outcome, according to the model.
'''

from ngboost import NGBRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np

X, Y = load_boston(True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ngb = NGBRegressor().fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)

# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test.flatten()).mean()
print('Test NLL', test_NLL)



## Try to work this stuff out
idx = 30
one_case = X_test[idx]
Y_test[idx]
# ??
#Y_preds = ngb.predict(one_case)
Y_preds[idx]
#Y_preds.item(idx)

Y_dists.loc[idx]
Y_dists.var[idx]

len(Y_preds)

import matplotlib.pyplot as plt
plt.scatter(Y_preds, Y_test, alpha=0.20, color='red')
plt.set_title('Probabilistic forecast')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()

# pred_dist function. This function enables to show the results of probabilistic predictions.

# see the probability distributions by visualising
Y_dists = ngb.pred_dist(X_test, 1)
Y_dists = ngb.pred_dist(X_test)

y_range = np.linspace(min(Y_test), max(Y_test), len(Y_test))
dist_values = Y_dists.pdf(y_range).transpose()
# plot index 0 and 114
idx = 13
#plt.plot(y_range, dist_values[idx])
plt.plot(y_range, dist_values)
plt.title(f"idx: {idx}")
plt.tight_layout()
plt.show()



##############

# NGBoost
from ngboost import NGBRegressor
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

#ngb = NGBoost(Base=default_tree_learner, Dist=Normal, Score=MLE(), natural_gradient=True, verbose=True)


##
ngb = NGBRegressor(Dist=Normal, Score=MLE, Base=default_tree_learner,
natural_gradient=True, n_estimators=100, learning_rate=0.01,
minibatch_frac=1.0, verbose=True, verbose_eval=100, tol=1e-4)


X, Y = load_boston(True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ngb.fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

a = ngb.feature_importances_
a.shape


# Feature importance plot
feature_importance = pd.DataFrame({'feature':load_boston()['feature_names'], 
                                   'importance':ngb.feature_importances_[0]})\
    .sort_values('importance',ascending=False).reset_index().drop(columns='index')
fig, ax = plt.subplots()
plt.title('Feature Importance Plot')
sns.barplot(x='importance',y='feature',ax=ax,data=feature_importance)

##

# Copied from the method

base_models = ngb.base_models
params_trees = zip(*base_models)

 # Get the feature_importances_ for all the params and all the trees
all_params_importances = [[getattr(tree, 'feature_importances_') 
		for tree in trees if tree.tree_.node_count > 1]  for trees in params_trees]
# Weighted average of importance by tree scaling factors
all_params_importances = np.average(all_params_importances, axis=1, weights=ngb.scalings)

all_params_importances /= np.sum(all_params_importances,axis=1,keepdims=True)


import sklearn
sklearn.__version__

import ngboost
ngboost.__version__


'0.22'

# Report issue
str(type(ngb.base_models[0][0]))

"<class 'sklearn.tree._classes.DecisionTreeRegressor'>"


isinstance(type(self.base_models[0][0]), sklearn.tree.DecisionTreeRegressor)

isinstance(ngb.base_models[0][0], sklearn.tree.DecisionTreeRegressor)



####
base_models = ngb.base_models
params_trees = zip(*base_models)
 # Get the feature_importances_ for all the params and all the trees
all_params_importances = [[getattr(tree, 'feature_importances_') 
		for tree in trees if tree.tree_.node_count > 1]  for trees in params_trees]

len(all_params_importances)
# Weighted average of importance by tree scaling factors
all_params_importances = np.average(all_params_importances, axis=1, weights=ngb.scalings)

base_models = ngb.base_models
params_trees = zip(*base_models)
all_params_importances = []
for trees in params_trees:
	for tree in trees:
		if tree.tree_.node_count > 1:
			all_params_importances.append(getattr(tree, 'feature_importances_'))

len(all_params_importances)