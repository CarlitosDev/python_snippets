Pandas DF’s indexes are immutable so each time we append a new chunk the index is recalculated. Work around: numpy indexes through dictionaries or namedtuples.

Avoid pd.append and DF.apply


# Extract list column into columns
# expand list
df = pd.DataFrame({'url_path': [['a', 'ff', 'c']]})
df_urls = pd.DataFrame(df['url_path'].values.tolist())





# Check if a DF is empty
dfTemp.empty


# Find some words in a pandas column
import pandas as pd
text = ['samples|fragrance|free-gift', 'free-gift|fragrances', 'free-gift']
df = pd.DataFrame({'A': text})
searchfor = ['sample', 'fragrance']
a = df.A.str.contains('|'.join(searchfor))
print(a)



Remove duplicates:
	dfPurchasesGBRFD.drop_duplicates(keep='first', inplace=True)


Column to lowercase
	Inbuilt:
	geoMeta['CityName'] = geoMeta.CityName.str.lower()
	Manually:
	geoMeta['lowerCityId'] = geoMeta.CityName.apply(lambda x: str.lower(x))


# Concatenate DF's with different datatypes
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
   					'B': ['B0', 'B1', 'B2', 'B3']})

df2 = pd.DataFrame({'A': [4, 6, 2, 3],
   					'B': ['B0', 'B1', 'B2', 'B3']})

result = pd.concat([df1,df2])

result.info()
result.A is a string...

typeDF1 = df1.dtypes.to_dict()
typeDF2 = df2.dtypes.to_dict()

thisType=typeDF1.keys()

for thisType in typeDF1.keys():
	a = typeDF1[thisType]
	b = typeDF2[thisType]
	if not (a == b):
		print('Crazy')

for iVar, iType in df1.dtypes.iteritems():
	print(iType)


# CAST: When a column mixes strings and floats...
#  There are some orders that were captured as strings instead of numbers, let's get rid of them
dfTrackingPoints.OrderId = pd.to_numeric(dfTrackingPoints.OrderId, errors='coerce')
idxValid                 = ~dfTrackingPoints.OrderId.isnull();
dfTrackingPoints         = dfTrackingPoints[idxValid]



To force the objects (strings) to become numbers:
df.convert_objects(convert_numeric=True) 



Cast large floats to strings:
a = dfTrackingPoints['OrderId'].astype('int64').astype(str)


# See type of variables grouped
df1.columns.to_series().groupby(df1.dtypes).groups


# Map values
# Use a dictionary to store the unique values of a variable,
# map a DF column to get the mapped values as a new colummn
unqCityId      = dfTrackingPoints.CityId.unique().tolist()
idxGeoMeta     = dfGeoMeta.CityId.isin(unqCityId)
dfGeoMeta      = dfGeoMeta[idxGeoMeta]
uniqueTypeDict = dict(zip(dfGeoMeta.CityId, dfGeoMeta.countryCodeISO3))
dfTrackingPoints['countryCodeISO3'] = dfTrackingPoints['CityId'].map(uniqueTypeDict)



# Convert a variable into categorical and then into a categorical but just with numbers
a = promoData[targetVariable].astype('category').cat.codes


Count the values (histogram-like):
	df["num_cylinders"].value_counts()
	# Easier to cast as a DF...
	cookieCounts = dfPurchasesGBRTPS['CookieID'].value_counts().to_frame()
	cookieCounts.head()



# MIND pivot table and crosstab
Create a pivot table with Pandas:
	dfPurchasesFiltPerDay = pd.pivot_table(dfPurchasesFilt, values='productCount', 
		index=['productName'], columns=['hour'], aggfunc=np.mean)


# Dataframe to text...
d = df_active[0:8].to_dict()
print(f'''df = pd.DataFrame.from_dict({d})''')

df = pd.DataFrame.from_dict({'origin': 
	{1170: 'mqtt:type:shuffleDefault', 1171: 'mqtt:type:shuffleDefault', 1527: 'mqtt:type:1_blue_songs', 1528: 'mqtt:type:2_green_stories', 1529: 'mqtt:type:3_green_mix',
 1530: 'mqtt:type:4_yellow_teeth', 1531: 'mqtt:type:6_purple_practice', 1532: 'mqtt:type:6_purple_practice'}, 
 'startTimestamp': {1170: pd.Timestamp('2019-09-05 14:16:15.630000'), 1171: pd.Timestamp('2019-09-05 14:16:58.107000'), 
 1527: pd.Timestamp('2019-09-03 07:49:21.981000'), 1528: pd.Timestamp('2019-09-03 07:49:45.745000'), 1529: pd.Timestamp('2019-09-03 07:50:12.277000'), 
 1530: pd.Timestamp('2019-09-03 07:51:28.660000'), 1531: pd.Timestamp('2019-09-03 07:53:03.288000'), 1532: pd.Timestamp('2019-09-03 07:53:03.288000')}, 
 'thingName': {1170: 'yd_50a2fa522ad52120', 1171: 'yd_50a2fa522ad52120', 1527: 'yd_cf3eeed4bde0a27f', 1528: 'yd_cf3eeed4bde0a27f', 1529: 'yd_cf3eeed4bde0a27f', 
 1530: 'yd_cf3eeed4bde0a27f', 1531: 'yd_cf3eeed4bde0a27f', 1532: 'yd_cf3eeed4bde0a27f'}, 'receiveTimestamp': {1170: pd.Timestamp('2019-09-05 14:16:55.245000'), 
 1171: pd.Timestamp('2019-09-05 14:16:58.378000'), 1527: pd.Timestamp('2019-09-03 07:49:36.372000'), 1528: pd.Timestamp('2019-09-03 07:49:46.131000'), 
 1529: pd.Timestamp('2019-09-03 07:50:14.951000'), 1530: pd.Timestamp('2019-09-03 07:51:29.509000'), 1531: pd.Timestamp('2019-09-03 07:53:05.376000'), 
 1532: pd.Timestamp('2019-09-03 07:53:14.050000')}, 'eventType': {1170: 'onetime', 1171: 'onetime', 1527: 'onetime', 1528: 'onetime', 1529: 'onetime', 
 1530: 'onetime', 1531: 'onetime', 1532: 'onetime'}, 'activityId': {1170: '188', 1171: '197', 1527: '249', 1528: '263', 1529: '265', 1530: '266', 1531: '270', 1532: '270'}, 
 'duration': {1170: 37.0, 1171: 0.0, 1527: 13.0, 1528: 0.0, 1529: 0.0, 1530: 0.0, 1531: 1.0, 1532: 10.0}, 
 'eventValue1': {1170: 'user', 1171: 'normal', 1527: 'normal', 1528: 'normal', 1529: nan, 1530: nan, 1531: nan, 1532: 'user'}, 
 'eventValue2': {1170: nan, 1171: nan, 1527: nan, 1528: nan, 1529: nan, 1530: nan, 1531: nan, 1532: nan}, 
 'eventValue3': {1170: nan, 1171: nan, 1527: nan, 1528: nan, 1529: nan, 1530: nan, 1531: nan, 1532: nan}, 
 'qty': {1170: 1.0, 1171: 1.0, 1527: 1.0, 1528: 1.0, 1529: 1.0, 1530: 1.0, 1531: 1.0, 1532: 1.0}})

df_active_breakdown = pd.pivot_table(df_active, values=['qty'], 
    index=['eventType'], columns=['eventValue1','eventValue2','eventValue3'], aggfunc='count')
file_path = os.path.join(dataFolder, current_table_name + '.xlsx')
fu.dataFrameToXLSv2(df_active_breakdown, file_path, writeIndex=True)

#
df = df_active
indexes           = [df.eventType, df.eventValue1, df.eventValue2, df.eventValue3];
contingency       = pd.crosstab(indexes, df.qty, margins=True,  margins_name='Totals')
fu.dataFrameToXLSv2(contingency, file_path, writeIndex=True)
fu.osOpenFile(file_path)



# Convert dataframe to text for reproducibility
d = df_active[0:15:90].to_dict()
print(f'''df = pd.DataFrame.from_dict({d})''')



# A bit more like Excel
# A good graphical explanation in http://pbpython.com/images/pivot-table-datasheet.png
df3 = pd.pivot_table(df, index=["widget_text", "dvce_type",'ue_widget_selection'],
                     columns = ['date'], values = ['hour'], 
					 aggfunc='count', fill_value = 0, margins = True)

df3.query('widget_text == ["What’s your preferred matte lipstick formula?"]')


Homemade aggregations (get stats for a column):
	dfPurchasesFiltStats = dfPurchasesFilt.groupby(['productName']).agg({'productCount': ['min', 'max', 'mean', 'std', 'sum']});
	dfPurchasesFiltStats = dfPurchasesFiltStats['productCount'].sort_values('sum', ascending=[0]);

Aggregations w/out creating a multiindex DF
	dfJoopPurchasesStats = dfTrackingPoints.groupby('CookieID').agg({'orderID': 'count'})


Access a multiindex DF by the value of one of the indeces:
dfToPlot.loc('0002638')


Aggregations and indexes:

# Do not use as index
dfToPlot = dfPurchasesGBRFDCln.groupby(['productid', 'productname', seriesBy, xAxis], as_index=False).agg({yAxis: 'sum'});
# use the aggregations as index
dfToPlot = dfPurchasesGBRFDCln.groupby(['productid', 'productname', seriesBy, xAxis], as_index=True).agg({yAxis: 'sum'});


[Added to the cookbook]
Time operations. Get hour and month from timestamp

df = pd.DataFrame([{'Timestamp': pd.tslib.Timestamp.now()}]);
df['month']       = df.Timestamp.dt.month;
df['hour']        = df.Timestamp.dt.hour;
df['day']         = df.Timestamp.dt.day;
df['day_of_week'] = df.Timestamp.dt.dayofweek;


Indexing a value in a DF (same result):
	idx = 0;
	dfTPminimal.CookieID.iloc[idx]
	dfTPminimal.iloc[idx].CookieID
	dfTPminimal.CookieID[idx]


Memory footprint of a DF:
	df.info();

To visualise dataframes a bit better:
	pd.set_option('expand_frame_repr', False)


Equal size DF:
	df = pd.DataFrame([1,2,3,4,5,6,7,8,9,10,11], columns=['TEST'])
	df_split = np.array_split(df, 3)
	Gives a list with the chunked DF








Get a percentage in one go:
	cookieRatio = dfCookies.value_counts()/dfCookies.count();

Find and replace backslashes:
(Backslashes use backslash so if we are dealing with functions that can accept a regex as input, let's make sure 
we explicitily pass the object as a regex one)
	b = r'\\'
	a = df.PublisherURL.str.contains(b)
	df.PublisherURL.str.replace(b, '', case=False);


Drop/Remove nulls from a dataset (based on a column):
	df = df[~df['Market'].isnull()]


Select rows and cols based on some condition:
	validColNames = ['Market', 'Year'];
	idxValid = ~df['Market'].isnull();
	df = df.ix[idxValid, validColNames];

Multiple indexing - must be a better way:
	listOfVars = ['numAggregations', 'numAggregations_imp', 'numAggregations_tp'];
	idxVals    = dfCITemp[listOfVars] > 0;
	for idx in [0,1,2]:
		dfCITemp.ix[idxVals.ix[:, idx], listOfVars[idx]] = 1.0;



Filter a dataframe by a list of values (all the unique ones):
	uniqueBPNS     = dfTest['item_nbr'].unique();
	idxItemsInList = dfTest['item_nbr'].isin(uniqueBPNS);

Find values in the columns of a dataframe:

	dfTest = pd.DataFrame();
	dfTest['item_nbr'] = np.array([1,4,7,9])
	dfTest['item_nbrList'] = [1,4,7,9]
	idx = np.array([4,7])
	idxA = dfTest['item_nbr'].isin(idx)
	idxB = dfTest['item_nbrList'].isin(idx)

Drop some rows of a DF (must be another way):
	dfPurchasesExt.drop(['customvars'], axis = 1, inplace = True)

idxList = list(range(1,60));
df.drop(df.index[idxList], inplace=True)

DropNaN's:
	df2.dropna(axis=1, how='any', inplace=True)

isnan:
	np.isnan(a)
(Pandas' isnull)
df.isnull().values.any()


To iterate through all the rows:
>> iterrows returns a Series for each row, it does not preserve dtypes across the rows
for index, row in df.iterrows():
	print(row['name'], row['score'])


One Hot encoding (OHE will be superseeded soon. Keep an eye)
	pd.get_dummies(obj_df, columns=["body_style", "drive_wheels"], prefix=["body", "drive"]).head()


Iterate through datatypes
	for iVar, iType in df.dtypes.iteritems():
		print(iType)

Also, a list with the selected dtype:	
	objTypes = df.select_dtypes(include=['object']).keys().tolist()


SORT OUT A DF
	grpDF.sort_values(['total'], ascending=[0], inplace = True)

Get a row: df.ix[0]
	.iloc is primarily integer position based
	.ix supports mixed integer and label based access

Indexing a column:
	idxPresentCVars  = ~df['customvars'].isnull();
	dfCustomVarsTemp = df.customvars.ix[idxPresentCVars]

Make a percentage count based on an index:
	# How many customvars are not null
	idxPresentCVars   = ~df['customvars'].isnull();
	print(idxPresentCVars.value_counts()/idxPresentCVars.count())

histogram-like counts (normalised):
pd.value_counts(df.isFraud, normalize = True)


Convert the row of a DF into a list:
	avgValues     = df.ix[0].values.tolist();
	
Reshuffle a DF (avoid time/reading effects):
	import numpy as np
	np.random.seed(0)
	df = df.reindex(np.random.permutation(df.index))

Apply a function to a DF column:
	p = lambda x: x*2
	df['newVar'] = df['sentiment'].apply(p)

Apply a function to a DF:
	cl1 = lambda x: str.replace(x, "___","_none_none_" )
	cl2 = lambda x: str.replace(x, "__","_none_" )
	df.applymap(cl1)
	df.applymap(cl2)
	# same idea but inline
	df['age_segment'] = df['age'].apply(lambda age: get_age_segment(age))



Create vars on-the-fly
 for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]


# Use shift to compare sorted ids
idx_last_record_teacher = (df_teachers.DimOnlineTeacherKey!=df_teachers.DimOnlineTeacherKey.shift(-1))


Concatenate columns of a DF into a new one:

	dfTrainWithItems['base_product_number_std'] = \
		dfTrainWithItems['item_nbr'].astype(str) + \
		'_' + dfTrainWithItems['store_nbr'].astype(str);


Get the column names of a DF:
	Either 'keys()' or tickers = df.columns.values.tolist() to get them on a list


Add columns in a DF where we want to keep the date index.In this case, 
we read every 'df' and add it to main_df keeping the dates as an index. The gaps will be filled with Nan
Replace the Nan's with zeroes
	df.fillna(0, inplace=True)


Add cols to dataframe:
	dfClicksCI['iNodeID'] = [ 'c' + str(idx) for idx in range(0, numClicks)]
	
Size of a dataframe: DataFrame.shape
nR, nC = df.shape

Set up PANDAS options at the beginning of the code
	import pandas as pd
	pd.options.display.max_rows = 10
	pd.options.display.float_format = '{:.1f}'.format
	# Prevent the cell to be truncated
	pd.set_option('display.max_colwidth', -1)

Group values bigger than zero and group them:
	a = data['A']>0
	b = data['B']>0
	data.groupby([a,b]).count() 


Filter by timestamp:
idxValidTimes = (dfTPoints['Timestamp'] > '2017-12-12 14:00') & (dfTPoints['Timestamp'] < '2017-12-12 17:00')


# create a column of percentages
df['percCity'] = 100*df['numtimes']/df['numtimes'].sum();

Set one column as index:
	browserMeta = browserMeta.set_index(['id'])

Rename columns:
	bannersMeta.rename(columns={'id': 'BannerId', 'name': 'BannerName'}, inplace=True);

Rename all columns to lowercase:
renameDict = dict(zip(df_temp.columns.tolist(), 
[str.lower(iCol) for iCol in df_temp.columns.tolist()]))
df_temp.rename(columns=renameDict, inplace=True);



Df to json:
df_clean.to_json(orient='records')


Fill nans in dataframe:
df.fillna(0)


# Find string data
import pandas as pd
text = ['samples|fragrance|free-gift', 'free-gift|fragrances', 'free-gift']
df = pd.DataFrame({'A': text})
searchfor = ['sample', 'fragrance']
a = df.A.str.contains('|'.join(searchfor))
print(a)



# rename columns in multiindex DFs
current_columns = dfGrouped.columns.levels[1].to_list()
d = {'max': 'latest', 'min': 'earliest', 'nunique': 'developers', 'count': 'number_PRs'}
new_columns = stu.replace_words_in_list(d, current_columns)

dfGrouped.columns.set_levels(new_columns, level=1, inplace=True)

