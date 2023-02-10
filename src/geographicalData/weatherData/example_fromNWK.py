#=========================================================
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
#=========================================================
#            SET TARGET DATA 
#=========================================================
#day=1
lat_target=45.0
lon_target=360-117.0
#=========================================================
#             SET OPENDAP PATH 
#=========================================================
pathname = 'http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_macav2metdata_huss_BNU-ESM_r1i1p1_historical_1950_2005_CONUS_daily.nc'
#=========================================================
#             GET DATA HANDLES
#=========================================================
filehandle=Dataset(pathname,'r',format="NETCDF4")
lathandle=filehandle.variables['lat']
lonhandle=filehandle.variables['lon']
timehandle=filehandle.variables['time']
datahandle=filehandle.variables['specific_humidity']
#=========================================================
#             GET DATA 
#=========================================================
#get data
time_num=365
timeindex=range(0,time_num,1)  #python starts arrays at 0
time=timehandle[timeindex]
lat = lathandle[:]
lon = lonhandle[:]
#=========================================================
#find indices of target lat/lon/day
lat_index = (np.abs(lat-lat_target)).argmin()
lon_index = (np.abs(lon-lon_target)).argmin()
#check final is in right bounds
if(lat[lat_index]>lat_target):
	if(lat_index!=0):
		lat_index = lat_index - 1
if(lat[lat_index]<lat_target):
	if(lat_index!=len(lat)):
		lat_index =lat_index +1
if(lon[lon_index]>lon_target):
	if(lon_index!=0):
		lon_index = lon_index - 1
if(lon[lon_index]<lon_target):
	if(lon_index!=len(lon)):
		lon_index = lon_index + 1
lat=lat[lat_index]
lon=lon[lon_index]
#=========================================================
#get data
data = datahandle[timeindex,lat_index,lon_index]
#=========================================================
#              MAKE A PLOT
#=========================================================
days = np.arange(0,len(time))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel(u'Day of Year')
ax.set_ylabel(u'Specific Humidity(kg/kg)')
ax.set_title(u'1950 Daily Specific Humidity(BNU-ESM) ,\n %4.2f\u00b0N, %4.2f\u00b0W' % (lat, abs(360-lon)))
ax.ticklabel_format(style='plain')
ax.plot(days,data,'b-')
plt.savefig("myPythonGraph.png")
plt.show()



#########
pip3 install climetlab



import pandas as pd
temp_filepath = '/Users/carlos.aguilar/Documents/Kaggle/Grocery Sales Forecasting/ext_data/daily_weather_in_Quito2017_no_header.csv'
df_temp = pd.read_csv(temp_filepath)
df_temp['date_str'] = df_temp[['DY', 'MO', 'YEAR']].apply(lambda dv: f'{dv[0]}-{dv[1]}-{dv[2]}', axis=1)
df_temp['date'] = pd.to_datetime(df_temp['date_str'], format='%d-%m-%Y')
df_temp.drop(columns='date_str', inplace=True)

cols_to_rename = {'T2M': 'avg_temp', 'PS': 'pressure', 'PRECTOT': 'total_precipitation', 'WS50M': 'wind_speed'}
df_temp.rename(columns=cols_to_rename, inplace=True)

df_temp.to_pickle(temp_filepath.replace('csv', 'pickle'))