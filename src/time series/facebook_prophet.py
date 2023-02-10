# Based on https://facebook.github.io/prophet/docs/quick_start.html#python-api
# https://github.com/facebook/prophet/blob/master/notebooks/quick_start.ipynb
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt # For ploting


# Create a seasonal TS in here
fs = 1000 # sample rate 1KHz
f  = 180 # the frequency of the signal

t  = 

x = np.arange(fs) # the points on the x axis for plotting
# compute the value (amplitude) of the sin wave at the for each sample
y = [ np.sin(2*np.pi*f * (i/fs)) for i in x]


% dfSignal = 100; %Hz
% fs = 1000;
% t = 0:1/fs:5-1/fs;
% inputSignal = cos(2*pi*dfSignal*t)+randn(size(t));


# showing the exact location of the smaples
plt.stem(x,y, 'r', )
plt.plot(x,y)
plt.show()

import os

filePath = os.path.join('/Users/carlos.aguilar/Documents/BeamlyRepos/prophet/examples', 'example_wp_peyton_manning.csv')
df = pd.read_csv(filePath)
df['y'] = np.log(df['y'])
df.head()


# Python
m = Prophet()
m.fit(df);

# Python
future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(forecast)
plt.show()


m.plot_components(forecast)
plt.show()

plt.plot(df.ds,df.y)
plt.show()