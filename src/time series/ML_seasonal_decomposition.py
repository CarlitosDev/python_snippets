from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import Series

series = pd.Series(data_between20and30['final_sales']).reset_index().drop(columns=['index'])
result = seasonal_decompose(series, model='additive', freq = 6) # 6 is an estimate of the frequency
result.plot()