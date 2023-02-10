'''
  Select the data using the map/coordinates or city name
  https://power.larc.nasa.gov/data-access-viewer/

  Then download the csv. I have cut off the headers
'''


import pandas as pd
temp_filepath = '/Users/carlos.aguilar/Documents/Kaggle/Grocery Sales Forecasting/ext_data/daily_weather_in_Quito2017_no_header.csv'
df_temp = pd.read_csv(temp_filepath)
df_temp['date_str'] = df_temp[['DY', 'MO', 'YEAR']].apply(lambda dv: f'{dv[0]}-{dv[1]}-{dv[2]}', axis=1)
df_temp['date'] = pd.to_datetime(df_temp['date_str'], format='%d-%m-%Y')
df_temp.drop(columns='date_str', inplace=True)

cols_to_rename = {'T2M': 'avg_temp', 'PS': 'pressure', 'PRECTOT': 'total_precipitation', 'WS50M': 'wind_speed'}
df_temp.rename(columns=cols_to_rename, inplace=True)




df_temp_monthly_agg = df_temp.groupby(['YEAR', 'MO']).agg({'avg_temp': ['mean'], 'total_precipitation': 'mean', 'T2M_MAX': 'mean', 'T2M_MIN': 'mean'})

jan = df_temp_monthly_agg.loc[(2017,1)]


## Something is fishy about the 2m temperature...
'''
Air temperature at 2 m above the land surface is a key variable used to assess climate change. However, observations of air temperature are typically only available from a limited number of weather stations distributed mainly in developed countries, which in turn may often report time series with missing values. As a consequence, the record of air temperature observations is patchy in both space and time. Satellites, on the other hand, measure land surface temperature continuously in both space and time.
'''
# Compare to this other dataset
df_monthly_weather = pd.read_csv('/Users/carlos.aguilar/Documents/Kaggle/Grocery Sales Forecasting/ext_data/terraclimate_-0.1807N_78.4678W_monthly_aggregates_no_header.csv')

monthly_weather_max = df_monthly_weather[0:12]['tmax(degC)']
monthly_weather_min = df_monthly_weather[0:12]['tmin(degC)']

aggregated_2m = df_temp_monthly_agg.loc[(2017)]


import numpy as np
B = monthly_weather_max.values
A_0 = aggregated_2m_max['T2M_MAX'].values.squeeze()
A    = np.vstack([A_0, np.ones(len(A_0))]).T
m, c = np.linalg.lstsq(A, B, rcond=None)[0]
df_temp['T2M_MAX_adj'] = df_temp['T2M_MAX']*m + c




B = monthly_weather_min.values
A_0 = aggregated_2m_max['T2M_MIN'].values.squeeze()
A    = np.vstack([A_0, np.ones(len(A_0))]).T
m, c = np.linalg.lstsq(A, B, rcond=None)[0]
df_temp['T2M_MIN_adj'] = df_temp['T2M_MIN']*m + c

# save the dataset with the adjusted values
df_temp.to_pickle(temp_filepath.replace('csv', 'pickle'))

alpha = 0.5*(df_temp['T2M_MAX']-df_temp['T2M_MIN'])
df_temp['T2M_MIN'] + alpha