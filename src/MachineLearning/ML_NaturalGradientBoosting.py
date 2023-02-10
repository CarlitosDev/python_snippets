'''

	https://github.com/stanfordmlgroup/ngboost
	https://stanfordmlgroup.github.io/projects/ngboost/
	https://towardsdatascience.com/interpreting-the-probabilistic-predictions-from-ngboost-868d6f3770b2

	Paper: https://arxiv.org/abs/1910.03225

	pip3 install --upgrade git+https://github.com/stanfordmlgroup/ngboost.git


	* NGBoost does not return confidence intervals. The confusion lies in the difference between a confidence interval and a prediction interval.
	* NGBoost just tells you the relative likelihood of observing each outcome, according to the model.


https://dkopczyk.quantee.co.uk/ngboost-explained/

'''

from ngboost import NGBRegressor, NGBoost
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt
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
idx = 3
one_case = X_test[idx].reshape(1,-1)
y_actual = Y_test[idx]

Y_preds = ngb.predict(one_case)
Y_dists = ngb.pred_dist(one_case)
mu_Y_dists = Y_dists.loc[0]
sigma_Y_dists = Y_dists.scale[0]

# From ngboost/normal.py
# The normal distribution has two parameters, loc and scale, which are the mean and standard deviation, respectively.
print(f'''Predicted value {Y_preds[0]:3.2f} belongs to a normal distribution with mu={mu_Y_dists:3.2f}  and sigma={sigma_Y_dists:3.2f}''')
print(f'Actual value {y_actual:3.2f}')



# let's integrate the pdf to see if it matches the
# actual value
std_distance = 3*sigma_Y_dists
y_range = np.linspace(mu_Y_dists-std_distance, mu_Y_dists+std_distance, 50)
dist_values = Y_dists.pdf(y_range).transpose()
plt.plot(y_range, dist_values)
plt.plot(y_actual, 0.25, 'ro')
plt.show()






## ?
print('P(y_0|x_0) is normally distributed with loc={:.2f} and scale={:.2f}'.format(Y_dists.loc[0], Y_dists.scale[0]))
print('P(y_1|x_1) is normally distributed with loc={:.2f} and scale={:.2f}'.format(Y_dists.loc[1], Y_dists.scale[1]))


Y_dists.var

len(Y_preds)




# Another example using a LogNormal
from ngboost.learners import default_tree_learner
from ngboost.scores import CRPS, MLE
from ngboost.distns import LogNormal, Normal
ngb_ln =  NGBRegressor(n_estimators=100, learning_rate=0.1,
              Dist=LogNormal,
              Base=default_tree_learner,
              natural_gradient=False,
              minibatch_frac=1.0,
              Score=CRPS)

ngb_ln.fit(X_train, Y_train-min(Y_train)+0.001)
Y_preds = ngb_ln.predict(X_test)
Y_dists = ngb_ln.pred_dist(X_test)
print(f'Mean of lognormal scale = {Y_dists.scale.mean():3.2f}')
print(f'Standard deviation of lognormal scale = {Y_dists.scale.std():3.2f}')

# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
test_R2 = r2_score(Y_test, Y_preds)
print(f'Test MSE (lognormal) {test_MSE:3.2f}')
print(f'Test R2 (lognormal) {test_R2:3.2f}')


# Normal dist
ngb_n =  NGBRegressor(n_estimators=100, learning_rate=0.1,
              Dist=Normal,
              Base=default_tree_learner,
              natural_gradient=True,
              Score=MLE)
ngb_n.fit(X_train, Y_train-min(Y_train)+0.001)
Y_preds = ngb_n.predict(X_test)
Y_dists = ngb_n.pred_dist(X_test)

test_MSE = mean_squared_error(Y_preds, Y_test)
test_R2 = r2_score(Y_test, Y_preds)
print(f'Test MSE (lognormal) {test_MSE:3.2f}')
print(f'Test R2 (lognormal) {test_R2:3.2f}')

print(f'Mean of lognormal scale = {Y_dists.scale.mean():3.2f}')
print(f'Standard deviation of lognormal scale = {Y_dists.scale.std():3.2f}')





# Probablity Integral transform
from ngboost.evaluation import *
pctles, observed, slope, intercept = calibration_regression(Y_dists, Y_test-min(Y_test)+0.001)
plt.subplot(1, 1, 1)
plot_pit_histogram(pctles, observed, label="CRPS", linestyle = "-")
plt.show()














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