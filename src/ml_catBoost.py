CatBoost
--------------------------
https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/


Focuses on categorical features
Symmetrical trees
Do not use any OHE as the algo deals with it
TaskType=gpu and we can use the gpu to speed up the processing
Evaluate feature interaction
SHAP library
Influencial documents: which objects in your dataset are the most important ones
New features evaluations (through p-value)
Catboost viewer (Jupyter ntb) [npm install catboost-viewer]
Missing values support
Bagging temperature: the observations are given a weight drawn 
from an exponential dist, controlled by this temperature parameter.
Nips: paper_11

End of CatBoost
--------------------------



Classifier (CatBoostClassifier)
#####
# CatBoost
####
# specify the training parameters for CatBoost
cb_iterations = 100
cb_depth = 8
cb_learning_rate = 1  
cb_loss_function = 'Logloss'
#cb_loss_function='CrossEntropy'

cb_model = CatBoostClassifier(iterations=cb_iterations, depth=cb_depth,
learning_rate=cb_learning_rate, loss_function=cb_loss_function, class_weights=[100, 1], logging_level='Verbose')

cb_model.fit(X_train[inputVars], y_train, cat_features=idx_categorical_features)




Regressor



### Create a CatbOOst.Pool from a Pandas DF

cb_pool = Pool(X_train, label=y_train, 
  cat_features = idx_categorical_features)



------

from catboost import CatBoostRegressor
# Initialize data

train_data = [[1, 4, 5, 6],
              [4, 5, 6, 7],
              [30, 40, 50, 60]]

eval_data = [[2, 4, 6, 8],
             [1, 4, 50, 60]]

train_labels = [10, 20, 30]
# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=2,
                          learning_rate=1,
                          depth=2)
# Fit model
model.fit(train_data, train_labels)

# Get feature importance
 model.get_feature_importance()

# Get predictions
preds = model.predict(eval_data)



-----
# Better regressor example with categorical variables

from catboost import CatBoostRegressor, Pool
idx_categorical_features = [0, 4]

cb_model = CatBoostRegressor(iterations=250, learning_rate=0.05, depth=12, 
loss_function='RMSE',
cat_features=idx_categorical_features, silent=False)

# Trick: do not pass a DF. Instead pass a numpy array to avoid issues when predicting
cb_model.fit(X_train.values,y_train)

y_test_hat = cb_model.predict(X_test.values)
# TO-DO: Check the variable importance here
# Get most important vars
cb_pool = Pool(X_train[inputVars], label=y_train, cat_features = idx_categorical_features)

df_feats = pd.DataFrame(cb_model.get_feature_importance(cb_pool, prettified=True))

print(df_feats)

