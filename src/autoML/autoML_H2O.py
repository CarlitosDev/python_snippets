'''

The current version of AutoML trains and cross-validates the following algorithms (in the following order):
  three pre-specified XGBoost GBM (Gradient Boosting Machine) models, 
  a fixed grid of GLMs, 
  a default Random Forest (DRF), 
  five pre-specified H2O GBMs, 
  a near-default Deep Neural Net, 
  an Extremely Randomized Forest (XRT), 
  a random grid of XGBoost GBMs, 
  a random grid of H2O GBMs, 
  and a random grid of Deep Neural Nets. 


From here:
http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

Check if XGBoost is avaiable
h2o.estimators.xgboost.H2OXGBoostEstimator.available()

'''


import h2o
from h2o.automl import H2OAutoML, get_leaderboard

h2o.init()

# Import a sample binary outcome train/test set into H2O
train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
# a = train.as_data_frame()
test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

# Identify predictors and response
x = train.columns
y = "response"
x.remove(y)

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# AutoML Leaderboard
lb = aml.leaderboard

# Optionally edd extra model information to the leaderboard
lb = get_leaderboard(aml, extra_columns='ALL')

# Print all rows (instead of default 10 rows)
lb.head(rows=lb.nrows)

# The leader model is stored here
aml.leader

# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly

preds = aml.predict(test)

# or:
preds = aml.leader.predict(test)