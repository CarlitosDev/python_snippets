'''

	source ~/.bash_profile && python3 -m pip install kats

	It contains a decent set of algos for UNIVARIATE data.

	https://towardsdatascience.com/kats-a-generalizable-framework-to-analyze-time-series-data-in-python-3c8d21efe057


	Kats supports the following 10 forecasting models:
		Linear
		Quadratic
		ARIMA
		SARIMA
		Holt-Winters
		Prophet
		AR-Net
		LSTM
		Theta
		VAR


Documentation.
https://facebookresearch.github.io/Kats/


'''


import os
import pandas as pd
import numpy as np

from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams

from utils.simulation_utils import generate_students_data
import matplotlib.pyplot as plt

# Get all the weeks since the 2nd of Sept
valid_from = pd.to_datetime('18-03-2018', format='%d-%m-%Y')
valid_to = pd.to_datetime('19-01-2021', format='%d-%m-%Y')
idxDates = pd.date_range(valid_from, valid_to, freq='W')

number_students = len(idxDates)
df_students = generate_students_data(number_students)
df_students['dates'] = idxDates


# Kats >> Only support univariate time series

# Construct TimeSeriesData object
ts = TimeSeriesData(df_students[['score', 'dates']], time_col_name='dates')


# (A) Use Prophet
params = ProphetParams(seasonality_mode='additive')

# Create a model instance
m = ProphetModel(ts, params)

# Fit mode
m.fit()

# Forecast
fcst = m.predict(steps=10, freq="W")
fcst

m.plot()
plt.show()



# (B) Using CUSUM detection algorithm on simulated data set.

from kats.consts import TimeSeriesData
from kats.detectors.cusum_detection import CUSUMDetector

# simulate time series with increase
np.random.seed(10)
df_increase = pd.DataFrame(
    {
        'time': pd.date_range('2019-01-01', '2019-03-01'),
        'increase':np.concatenate([np.random.normal(1,0.2,30), np.random.normal(2,0.2,30)]),
    }
)

# convert to TimeSeriesData object
timeseries = TimeSeriesData(df_increase)

# run detector and find change points
change_points = CUSUMDetector(timeseries).detector()