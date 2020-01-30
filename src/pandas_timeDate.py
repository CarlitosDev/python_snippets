
# TO FINISH ONE DAY
'''
df = pd.io.json.json_normalize(subDict)
df = df.infer_objects()
obj_columns = df.select_dtypes(include=['object'])

for iCol in df.select_dtypes(include=['object']):
    if any(df[iCol].str.contains('-|/|:')):
        try:
            result = parse(df[iCol][0], fuzzy_with_tokens=True)
        except ValueError:
            pass

contains('expert_model|challenger_model')

from dateutil.parser import parse


result = parse(df.iloc[-1].lesson_start_time, fuzzy_with_tokens=True)
result = parse("manolo", fuzzy_with_tokens=True)
'''



# Parser for ISO 8601
df = pd.io.json.json_normalize(subDict)
df = df.infer_objects()
# double-check we are not missing ISO8601 dates
this_regex = r'(\d+)-(\d+)-(\d+)T(\d+):(\d+):(\d+)Z'
regEx      = re.compile(this_regex)
for iCol in df.columns.tolist():
    if regEx.match(df[iCol].iloc[0]) != None:
        df[iCol] = pd.to_datetime(df[iCol], format='%Y-%m-%dT%H:%M:%SZ')




# Define dates from strings
start_UT = pd.Timestamp('2019-09-20').date()
end_UT = pd.Timestamp('2019-10-17').date()


Time operations. Get hour and month from timestamp

df = pd.DataFrame([{'Timestamp': pd.tslib.Timestamp.now()}]);
df['month']       = df.Timestamp.dt.month;
df['hour']        = df.Timestamp.dt.hour;
df['day']         = df.Timestamp.dt.day;
df['day_of_week'] = df.Timestamp.dt.dayofweek;


# Filter a column based on a timestamp
idx = df['startTimestamp'] >= pd.Timestamp('2019-09-19')


# Get the minutes(seconds) of a timedelta - Time differences
df_adhoc_ext['block_total_minutes'] = (df_adhoc_ext['ts_end'] - df_adhoc_ext['ts_begin']).apply(lambda x: x.total_seconds()/60.0)

# Example of step difference (ie; session duration)
df = pd.DataFrame({'Col1': [10, 20, 35, 42, 49]})
df['step_time'] = df.shift(-1)-df

# In minutes
higueputa = (df_activity['startTimestamp'].shift(-1) - df_activity['startTimestamp']).apply(lambda x: x.total_seconds()/60.0)




#Date differences in Pandas. Get a column with the date of a lagged day:
df = pd.DataFrame({ 'A' : 1., 'date' : pd.Timestamp('20130102') }, index=[0])
currentLag = 1;
df['dayLagged_{}'.format(currentLag)] = df.date - timedelta(days=currentLag);


# Get all the weeks since the 2nd of Sept
valid_from = pd.to_datetime('02-09-2019', format='%d-%m-%Y')
valid_to = pd.to_datetime(datetime.now())
idxDates = pd.date_range(valid_from, valid_to, freq='W')


# Matlab's datenum
from datetime import date
print(date.toordinal(date(1970,1,1)))
df.date.apply(date.toordinal)


# Give it a go with dates
minDate  = min(dfPurchasesFD.date)
maxDate  = max(dfPurchasesFD.date)
idxDates = pd.date_range(minDate, maxDate)

s       = pd.Series({minDate: 0, maxDate: 0})
s.index = pd.DatetimeIndex(s.index)
s       = s.reindex(idxDates, fill_value=0).to_frame('sales')



# Snippet
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import date, timedelta


df = pd.DataFrame([{'unit_sales' :1, 'date' : pd.Timestamp('20130103'), 'store_nbr' : 2, 'onpromo': True }, 
				  {'unit_sales' : 7, 'date' : pd.Timestamp('20130106'), 'store_nbr' : 2, 'onpromo': False }, 
				  {'unit_sales' : 2, 'date' : pd.Timestamp('20130102'), 'store_nbr' : 1, 'onpromo': True  }])

varsToKeep = ['unit_sales', 'date', 'store_nbr', 'onpromo'];
leftKeys   = ['date', 'store_nbr']

for currentLag in range(1,28+1):
	currentLagVar     = 'dLag_{}'.format(currentLag);
	df[currentLagVar] = df.date - timedelta(days=currentLag);
	varsToKeepRightDF = [x + '_' + currentLagVar for x in ['unit_sales', 'onpromo']]
	allVarsToKeep     = varsToKeep
	for iVar in varsToKeepRightDF:
		allVarsToKeep.append(iVar)
	# Set the current rightKeys	
	rightKeys = [currentLagVar, 'store_nbr'];
	dfTemp    = pd.merge(df, df, how='left', left_on=leftKeys, right_on=rightKeys,  suffixes=('', '_' + currentLagVar), copy=True);
	# Update the DF
	df = dfTemp[allVarsToKeep].copy();
	varsToKeep = allVarsToKeep



# get the following 16 days...
t2017 = date(2017, 5, 31)
pd.date_range(t2017, periods=16)



#From timestamp to string:
dateAsStr = dfTest['date'].dt.strftime('%d_%m_%Y');

# Must be another way without calling datetime...
from datetime import datetime
datetime.strptime('02-07-2017', '%d-%m-%Y').date()

#From timestamp to string:
stocks.index[-1].strftime('%Y-%m-%d')



#From string to date (PANDAS)):
pd.to_datetime(thisCookieID_Clicks['Timestamp'], format='%Y-%m-%d %H:%M:%S')


# a bit more complex. Dates to string in a required format and then unique and to a list:
idxValid = dfCITemp['retrieveRow'] > 0
listOfDates = dfCITemp['date'].ix[idxValid].dt.strftime('%Y_%m_%d').unique().tolist()



# Normal Python from string to date
datetime.strptime(war_start, '%Y-%m-%d')




minDate  = min(dfPurchasesFD.date)
maxDate  = max(dfPurchasesFD.date)
idxDates = pd.date_range(minDate, maxDate)

s       = pd.Series({minDate: 0, maxDate: 0})
s.index = pd.DatetimeIndex(s.index)
s       = s.reindex(idxDates, fill_value=0).to_frame('sales')


From string to date:
pd.to_datetime(thisCookieID_Clicks['Timestamp'], format='%Y-%m-%d %H:%M:%S')



From timestamp to date:
	df_activities['date'] = df_activities['startTimestamp'].apply(lambda ts: ts.date())

a bit more complex. Dates to string in a required format and then unique and to a list:
	idxValid = dfCITemp['retrieveRow'] > 0
	listOfDates = dfCITemp['date'].ix[idxValid].dt.strftime('%Y_%m_%d').unique().tolist()


# today
import datetime
now = datetime.datetime.now()


# Go back 6 months from today
from datetime import date
from dateutil.relativedelta import relativedelta
six_months = date.today() - relativedelta(months=6)
