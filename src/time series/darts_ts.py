
'''


TS library from a Swiss company. It integrates classic approaches and some TCN (Temporal ConvNet)

"Darts attempts to be a scikit-learn for time series, and its primary goal is to 
simplify the whole time series machine learning experience."


https://medium.com/unit8-machine-learning-publication/darts-time-series-made-easy-in-python-5ac2947a8878
pip3 install 'u8darts[all]' --ignore-installed


Further examples
https://github.com/unit8co/darts/tree/master/examples
TCN (Temporal ConvNet)
https://github.com/unit8co/darts/blob/master/examples/TCN-examples.ipynb

'''




import pandas as pd
from darts import TimeSeries
df = pd.read_csv('/Users/carlos.aguilar/Downloads/AirPassengers.csv')
series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')
train, val = series.split_before(pd.Timestamp('19580101'))



from darts.models import AutoARIMA

model_aarima = AutoARIMA()
model_aarima.fit(train)
prediction_aarima = model_aarima.predict(len(val))


import matplotlib.pyplot as plt
series.plot(label='actual')
prediction_aarima.plot(label='forecast', lw=3)
plt.legend()
plt.show()



from darts.models import ExponentialSmoothing
from darts.models import Prophet
models = [ExponentialSmoothing(), Prophet()]
backtests = [model.backtest(series,
                            start=pd.Timestamp('19550101'),
                            forecast_horizon=3)
             for model in models]



from darts.metrics import mape
series.plot(label='data')
for i, m in enumerate(models):
    err = mape(backtests[i], series)
    backtests[i].plot(lw=3, label='{}, MAPE={:.2f}%'.format(m, err))
plt.title('Backtests with 3-months forecast horizon')
plt.legend()
plt.show()


# Also
# https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4