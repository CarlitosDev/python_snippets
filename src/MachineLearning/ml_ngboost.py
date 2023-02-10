from ngboost import NGBRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X, Y = load_boston(True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ngb = NGBRegressor().fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# If using evaluation set
ngb.fit(X_reg_train, Y_reg_train, X_val=X_reg_test, Y_val=Y_reg_test)


# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)


features = ngb.feature_importances_
## Feature importance for loc trees
feature_importance_loc = ngb.feature_importances_[0]

## Feature importance for scale trees
feature_importance_scale = ngb.feature_importances_[1]



# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test.flatten()).mean()
print('Test NLL', test_NLL)


###

# Reshape the base_models
base_models = ngb.base_models
# create an iterator using zip
params_trees = zip(*ngb.base_models)
# Get the feature_importances_ for all the params and all the trees
all_params_importances = [[getattr(tree, 'feature_importances_') 
    for tree in trees if tree.tree_.node_count > 1] 
        for trees in params_trees]

params_trees = zip(*ngb.base_models)
[trees for trees in params_trees]

params_trees = zip(*ngb.base_models)
for trees in params_trees:
    trees
    trees.tree_.node_count

tr = base_models[0]
idx = 0
tr[idx].tree_.node_count



if not all_params_importances:
    return np.zeros(len(self.base_models[0]),self.base_models[0][0].n_features_, dtype=np.float64)
# Weighted average of importance by tree scaling factors
all_params_importances = np.average(all_params_importances,
                            axis=1, weights=self.scalings)
return all_params_importances / np.sum(all_params_importances,axis=1,keepdims=True)