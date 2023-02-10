# Missings
# See the variables with missing data
possible_nan = df_train.isna().any()
possible_nan[possible_nan.values]
# See the number of rows with missing data per variable
possible_nan = df_train.isna().sum().sort_values(ascending=False)

# Show the missings as a DF including percentage
possible_nan = df_all_conversions.isna().sum().sort_values(ascending=False).to_frame()
possible_nan.columns = ['count']
possible_nan['percentage'] = 100*possible_nan['count']/df_all_conversions.shape[0]
possible_nan.head(20)



# gives some infomation on columns types and number of null values
tab_info=pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))
print ('-' * 10 + " Display information about column types and number of null values " + '-' * 10 )
print 
display(tab_info)







# Duplicated rows
print('Duplicate data entries: {}'.format(df_initial.duplicated().sum()))
df_initial.drop_duplicates(inplace = True)

# Reomve duplicates based on the columns
varsDuplication = ['startTimestamp', 'thingName', 'receiveTimestamp', 'eventType']
df_activity.drop_duplicates(subset=varsDuplication, inplace=True)



# Product counts
pd.DataFrame([{'products': len(df_initial['StockCode'].value_counts()),    
               'transactions': len(df_initial['InvoiceNo'].value_counts()),
               'customers': len(df_initial['CustomerID'].value_counts()),  
              }], columns = ['products', 'transactions', 'customers'], 
              index = ['quantity'])



# Check the contents of this variable by looking for the set of codes that would contain only letters:
list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()


# Quick trick to visualise a histogram
df['device_name'].value_counts().plot(kind='bar', figsize = (7,7))


# 2 - Quick trick to visualise a histogram
df['avg_price'].hist(bins=40)
