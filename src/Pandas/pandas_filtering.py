Filter a dataframe based on two conditions:
	world = world[(world.pop_est>0) & (world.continent=="Europe")]


Select by dates:
	minDate    = datetime(2016, 7, 1);
	maxDate    = datetime(2016, 9, 1);
	rangeDates = pd.date_range(minDate, maxDate);
	idxPeriodA = dfTrain['date'].isin(rangeDates);
	# directly...
	dfA = dfTrain[rangeDates]


Pandas timestamps as strings:
#timestamp   
df['thisTimeStamp'] = df['yyyy_mm_dd'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'));

Strings to Pandas timestamp:
pd.to_datetime(thisCookieID_Clicks['Timestamp'], format='%Y-%m-%d %H:%M:%S')

# extract the date from a timestamp
dfSnowplow['date'] = dfSnowplow.dvce_tstamp.apply(lambda x: x.date());



Time differences with Pandas:
Pandas series less than a minute
	timeDiffBetweenCookies < np.timedelta64(1, 'm')



# Get the indices of the min and max
idx_max = df_pull_requests['bodyTextLenght'].idxmax()
idx_min = df_pull_requests['bodyTextLenght'].idxmin()
