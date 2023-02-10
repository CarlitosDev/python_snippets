'''

  NeuralProphet.py

  cd DS_repos
  git clone https://github.com/ourownstory/neural_prophet
  cd neural_prophet
    python3 setup.py install



Feautures:

Autocorrelation modelling through AR-Net
Piecewise linear trend
Fourier term Seasonality at different periods such as yearly, daily, weekly, hourly.
Lagged regressors
Future regressors
Holidays & special events
Sparsity of coefficients through regularization
Plotting for forecast components, model coefficients as well as final forecasts


Example:

https://github.com/ourownstory/neural_prophet/blob/master/example_notebooks/events_holidays_peyton_manning.ipynb

'''

from neuralprophet import NeuralProphet
model = NeuralProphet()
