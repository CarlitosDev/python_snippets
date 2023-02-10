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



---


from catboost import CatBoostClassifier, Pool

# The labels cannot be booleans!
train_data = Pool(data=[[1, 4, 5, 6],
                        [4, 5, 6, 7],
                        [30, 40, 50, 60]],
                  label=[True,True, False],
                  weight=[0.1, 0.2, 0.3])

train_data.get_label()

model = CatBoostClassifier(iterations=10)
model.fit(train_data)
preds_class = model.predict(train_data)



train_data.get_label()



# The labels cannot be booleans!
eval_pool = Pool(data=[[1, 4, 5, 6],
                        [4, 5, 6, 7],
                        [30, 40, 50, 60]],
                  label=[0,0, 1],
                  weight=[0.1, 0.2, 0.3])


import numpy as np
from sklearn.metrics import classification_report, roc_curve, precision_score, recall_score, f1_score

y_eval = np.array(eval_pool.get_label())

y_eval_bin = y_eval > 0
#y_pred = self.classifier.predict(eval_pool).astype('bool')
y_pred = np.array([0,1,1])
y_pred_bin = y_pred > 0

plain_f1_score = f1_score(y_eval, y_pred)
plain_precision_score = precision_score(y_eval, y_pred)


plain_f1_score = f1_score(y_eval_bin, y_pred_bin)
plain_precision_score = precision_score(y_eval_bin, y_pred_bin)
plain_recall_score = recall_score(y_eval_bin, y_pred_bin)
print(classification_report(y_eval_bin, y_pred_bin))


y_eval = [0,0, 1,1]
prob_class_1 = np.array([0.3, 0.25,0.7,0.35])
fpr, tpr, thresholds = roc_curve(y_eval, prob_class_1, pos_label=1)
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]


y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
fpr

tpr

thresholds

y_eval = [0,0, 1,1,1,0]
prob_class_1 = np.array([0.3, 0.25,0.7,0.35,0.28,0.6])
fpr, tpr, thresholds = roc_curve(y_eval, prob_class_1, pos_label=1)
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]
best_thresh