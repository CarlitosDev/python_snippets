'''
	sysidentpy_ts.py

	pip3 install sysidentpy


From here: http://sysidentpy.org/introduction_to_narmax.html

For nonlinear scenarios we have the NARMAX class (Non-linear Autoregressive 
Models with Moving Average and Exogenous Input). 
As reported by Billings (the creator of NARMAX model) in the book Nonlinear System Identification: 
NARMAX Methods in the Time, Frequency, and Spatio-Temporal Domains, 
NARMAX started out as a model name, but soon became a philosophy when it comes 
to identifying nonlinear systems. 

Obtaining NARMAX models consists of performing the following steps:

        - Dynamical tests and collecting data;
        - Choice of mathematical representation;
        - Detection of the model structure;
        - Estimation of parameters;
        - Validation;
        - Analysis of the model.


'''


from sysidentpy.general_estimators import NARX
from catboost import CatBoostRegressor

catboost_narx = NARX(
        base_estimator=CatBoostRegressor(
                iterations=300,
                learning_rate=0.1,
                depth=6),
        xlag=[2,2,2],
        ylag=2,
        fit_params={'verbose': False}
)



from utils.simulation_utils import generate_students_data
import pandas as pd
# Get all the weeks since the 2nd of Sept
valid_from = pd.to_datetime('18-03-2018', format='%d-%m-%Y')
valid_to = pd.to_datetime('19-01-2021', format='%d-%m-%Y')
idxDates = pd.date_range(valid_from, valid_to, freq='W')

number_students = len(idxDates)
df_students = generate_students_data(number_students)
# df_students['dates'] = idxDates

inputVars = df_students.columns.tolist()
responseVar = 'score'
inputVars.remove(responseVar)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
  train_test_split(df_students[inputVars].values, df_students[responseVar].values, \
  test_size=0.1)


# build
catboost_narx.fit(X_train, y_train.reshape(-1, 1))
# predict
y_hat = catboost_narx.predict(X_test, y_test.reshape(-1,1))

ee, ex, extras, lam = catboost_narx.residuals(X_test, y_test.reshape(-1,1), y_hat)

catboost_narx.plot_result(y_test.reshape(-1,1), y_hat, ee, ex)








# Generate a dataset of a simulated dynamical system
from sysidentpy.utils.generate_data import get_siso_data

x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000,
        colored_noise=False,
sigma=0.001,
train_percentage=80
)

y_train.shape