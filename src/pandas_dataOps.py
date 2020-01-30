Find variables within a range:
df = pd.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 3]},index=['falcon', 'dog'])
df.num_legs.between(0,2)
df.num_wings.between(0,2)


# Take the rows that contain the maximum per group
# In this example, retain the stores with the largest forecast
df = pd.DataFrame([ 
	{'store_id': 1, 'forecast': 35},  \
	{'store_id': 1, 'forecast': 28},  \
	{'store_id': 1, 'forecast': 18},  \
	{'store_id': 2, 'forecast': 315}, \
	{'store_id': 2, 'forecast': 268}, \
	{'store_id': 2, 'forecast': 518}
	])

idx = df.groupby(['store_id'])['forecast'].transform(max) == df['forecast']
df[idx]




A primer to groupings:
	groupedDF = df.groupby(df['campaign_id'])
	for name, group in groupedDF:
		groupClientName = group['clientsname'].iloc[0]
		print('Current campaign {} id {} got {} clicks'.format(groupClientName, name, group['totalclicks'].sum()))
	# Also, you can extract into a dictionary and index v b y the key value
	v = dict(list(groupedDF))

A bit more on grouping:

	df = pd.DataFrame([{'A': 'foo', 'B': 'green', 'C': 11, 'D': 's1'}, \
					{'A':'bar', 'B':'blue', 'C': 20, 'D': 's2'}, \
					{'A':'foo', 'B':'blue', 'C': 20, 'D': 's3'}])
	# group by one of the fields (this one is a bit confussing as I rather group by a variable: df.groupby(['A', 'B']);)
	groupedDF = df.groupby(df['A']);
	# get the list of groups
	groupNames = list(groupedDF.groups.keys());
	# access one of the groups
	thisGroup = groupedDF.get_group(groupNames[0])



# Aggregate and calculate the mean in one go
df_products_mu  = df_productExt.groupby('product_ean', as_index=True)[inputFieldNames].mean()


# Get cumsum per row by aggregating, indexing and set index to column
varsForAgg = ['DimOnlineTeacherKey','total_dayhour_10', 'total_dayhour_15']
df_tempA = df_teachers[varsForAgg].iloc[0:60].copy()
df_tempA['DimOnlineTeacherKey'] = df_tempA['DimOnlineTeacherKey'].astype(int).astype(str)
df_tempA.set_index('DimOnlineTeacherKey', inplace=True)

df_tempB = df_tempA.groupby('DimOnlineTeacherKey', as_index=False).cumsum()
df_tempB.reset_index(level=0, inplace=True)




# Use OHE to explode a categorical variable and also, set the fields to 1 when the value is None
ohe_responses = []
userQuestions_OHE = pd.DataFrame()
for iQuestion in userQuestions:
    df_temp = pd.get_dummies(df_product[iQuestion], \
        columns=iQuestion, prefix=iQuestion, dummy_na=True)
    allSelected = df_temp[iQuestion + '_nan'].values
    df_temp.drop(columns=[iQuestion + '_nan'], inplace = True)
    for iCol in df_temp.columns.tolist():
        df_temp[iCol] = df_temp[iCol] + allSelected
    ohe_responses.append(df_temp)

userQuestions_OHE.append(ohe_responses, ignore_index=True)




# Averages using groups
df = pd.DataFrame([{'A': 'foo', 'B': 'green', 'C': 11, 'D': 40}, \
                {'A':'bar', 'B':'blue', 'C': 20, 'D': 12}, \
                {'A':'foo', 'B':'blue', 'C': 20, 'D': 87}])
# option 1
aggregations  = {'D': 'mean', 'C': 'max'}
dfGrouped     = df.groupby('A', as_index=False).agg(aggregations).copy();
# option 2
dfGrouped     = df.groupby('A', as_index=False).mean()
# (just some vars)
dfGrouped     = df.groupby('A', as_index=False)[['D']].mean()



# Some more tricks
aggregations  = {'createdAt': ['min', 'max', 'count'], 
'author': pd.Series.nunique}
dfGrouped     = df_pull_requests.groupby('baseRepository', as_index=False).agg(aggregations).copy()


Inner joins with Pandas:
# Code this SQL query using Pandas
sqlQuery = '''select
A.ID, A.Site,A.Value2,A.Random,
C.minSD,C.maxSD,
sum(A.Value)     as totalValue
from df as A
inner join (select B.ID,     
            min(B.StartDate) as minSD,
            max(B.EndDate)   as maxSD 
            from df as B
            group by 1) as C
    on A.ID = C.ID    
group by 1,2,3,4,5,6 
'''

# Same with pandas (more on groupby)
varA      = 'ID';
dfGrouped = df.groupby(varA, as_index=False).agg({'StartDate': 'min', 'EndDate': 'max'}).copy();

# merge (inner join, left join)
varsToKeep = ['ID', 'Value', 'Site', 'Value2', 'Random', 'StartDate_grp', 'EndDate_grp'];
dfTemp = pd.merge(df, dfGrouped, how='inner', on='ID', suffixes=(' ', '_grp'), copy=True)[varsToKeep];

dfBreakDown = dfTemp.groupby(['ID', 'Site', 'Value2', 'Random', 'StartDate_grp',
       'EndDate_grp']).sum()



#Aggregations (sql-like):

df = pd.DataFrame([{'A': 'foo', 'B': 'green', 'C': 11, 'D': 's1'}, \
				{'A':'bar', 'B':'blue', 'C': 20, 'D': 's2'}, \
				{'A':'foo', 'B':'blue', 'C': 20, 'D': 's3'}])

aggregations  = {'C': 'sum', 'B': 'count'}
dfGrouped     = df.groupby(['A','D'], as_index=False).agg(aggregations).copy();



To use count-distinct:
dfGrouped = dfPurchasesExt.groupby(varA, as_index=False).agg({'yyyymmdd_str': pd.Series.nunique}).copy();



Quick count and return as DF:
	
my_tab = pd.crosstab(index=titanic_train["Survived"], columns="count")




Append dataframes with different nuymber of columns:

df = pd.DataFrame([{'A': 'foo', 'B': 'green', 'C': 11}, \
				{'A':'bar', 'B':'blue', 'C': 20}])
df2 = pd.DataFrame([{'A': 'foo', 'B': 'green', 'C': 11, 'D': 'cojones'}])
df = df.append(df2, ignore_index=True);



# Percentage change
this_series = pd.Series([43,41,90,None,23, -19])
print(this_series.pct_change(periods=1))


from sklearn.preprocessing import normalize, scale
this_series.fillna(method="ffill", inplace=True)
df = pd.DataFrame();
df['data']       = this_series
df['scaled']     = scale(this_series.values)  # zero mean and unit sigma 
df['normalised'] = normalize(this_series.values.reshape(1,-1))[0]  # scale between 0 and 1.
df.head()



MinMaxScaler().fit_transform(X)

MinMaxScaler().fit_transform(this_series.values.reshape(-1, 1))


df['data'].sum()


from sklearn.preprocessing import StandardScaler
lambda x: StandardScaler().fit_transform(x)

x.reshape(1, -1)

df['data'].apply(lambda x: StandardScaler().fit_transform(x), index=1)


StandardScaler().fit_transform(df['data'])

x_prime = df['data'].values.reshape(-1,1)

StandardScaler().fit_transform(x_prime)
MinMaxScaler().fit_transform(x_prime)



column_transformation
df_2 = column_transformation(df, ['data'], type_transformation='MaxAbsScaler');
df_3 = column_transformation(df, ['data', 'normalised'], type_transformation='Normalizer');

StandardScaler
MinMaxScaler
MaxAbsScaler
Normalizer





# Compare two dataframes
df_1 = pd.DataFrame({'cost': [250, 150, 100], 'revenue': [100, 250, 300]}, index=['A', 'B', 'C'])
df_2 = pd.DataFrame({'cost': [150, 250, 100,30], 'revenue': [100, 250, 300,30]}, index=['A', 'B', 'C', 'D'])

idx = ~df_1.eq(df_2, axis = 1).all(axis=1)
# Rows from df_2 not present in df_1
df_2[idx]



df_A = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
"B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
"C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
"D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
"E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})

table = pd.pivot_table(df_A, values=['D', 'E'], index=['A', 'C'], columns = ['B'], 
aggfunc={'D': [np.min, np.max], 'E': np.mean})



#### Combine groupby and pivot table

# pivot table
table = pd.pivot_table(df, index=indexes, columns=column, aggfunc=[np.sum], \
  values = values, fill_value = 0, margins=False)

# group by
dfGrouped = df.groupby(indexes, as_index=True).agg(aggregations).copy()

# merge
df_session_breakdown = pd.merge(table, dfGrouped, how='right', left_index=True, right_index=True, copy=True)
df_session_breakdown.columns.tolist()
df_session_breakdown.sort_values([('startTimestamp', 'to')], inplace=True)