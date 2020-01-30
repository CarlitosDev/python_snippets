Isolation forest distance-less regressors/classifiers available in sklearn
The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.


Paper claiming that statistical forecasting outmaths ML
http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194889

Article on CatBoost, LGBM and XGboost
https://www.kdnuggets.com/2018/03/catboost-vs-light-gbm-vs-xgboost.html

Skater library for ML explanation
https://datascienceinc.github.io/Skater/overview.html

Multiclass is exclusive, multilabel is not. Check that out.

Ideas to test out if your feature selection is doing the job: detach the response from the variables, just reshuffle the y but keep the order of the x
abd run your FS code. If the score is still decently high, something doesn't work.

CVBagging:

On each run, select the average of the hyperparameters in each fold > leads to a very robust model. (That's what we did in the paper)


https://en.wikipedia.org/wiki/Knapsack_problem
https://en.wikipedia.org/wiki/Anscombe%27s_quartet





One Hot encoding (OHE will be superseeded soon. Keep an eye)
	pd.get_dummies(obj_df, columns=["body_style", "drive_wheels"], pdrefix=["body", "drive"]).head()



# >> x must be a pandas series
# Outlier detection based on MAD
mad_score = lambda x: np.abs((0.6745*(x-x.median()))/x.mad())


# One Hot Encoding (Just go to PANDAS)
oheTarget = pd.get_dummies(df1['target'])



# get the confussion matrix
confusion_matrix = pd.DataFrame(
    confusion_matrix(df_test["Churn"], predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"]
)
confusion_matrix.head()



# Get the ROC
fpr, tpr, threshold = roc_curve(df_test["Churn"], probs[:,1])



# Use multi-output decision trees (y is 2-dimensional)
# ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += (0.5 - rng.rand(20, 2))
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
# Predict
X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)

(2)
Embedding LightGBM as a multi-ouput

from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from tqdm import tqdm
y_pred = 0
N = 2
for i in tqdm(range(N)):
    model = MultiOutputRegressor(lgb.LGBMRegressor(random_state=i*101), n_jobs=-1)
    model.fit(X, two_Y)
    y_pred += model.predict(test)
y_pred /= N




# ---