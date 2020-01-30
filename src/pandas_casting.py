
# Use the infer_object function to make a better match of the data types
df = pd.DataFrame({"A": ["a", 1, 2, 3]})
df = df.iloc[1:]
df.dtypes
a = df.infer_objects()
df.dtypes
a.dtypes


# Select some datatypes and cast them. 
# In the example below, pick the columns with integers and cast them to floats
int_colnames = input_catalogue.select_dtypes(include=['int']).columns
input_catalogue[int_colnames] = input_catalogue[int_colnames].astype(float)

# the whole DF:
userQuestions_OHE = userQuestions_OHE.astype(float)


#  There are some orders that were captured as strings instead of numbers, let's get rid of them
dfTrackingPoints.OrderId = pd.to_numeric(dfTrackingPoints.OrderId, errors='coerce')

#Change from integer to float:
	# clean the item_price column and transform it in a float
	prices = [float(value[1 : -1]) for value in chipo.item_price]
	# reassign the column with the cleaned prices
	chipo.item_price = prices 

#Change from integer to categorical:
	char_cabin = titanic_train["Cabin"].astype(str)    # Convert cabin to str
	new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter
	titanic_train["Cabin"] = pd.Categorical(new_Cabin)  # Save the new cabin var


# ...or cast after reading 
df["age"] = df["age"].astype(np.int16)

# Cast from float to string removing the decimals
dfTPminimal['orderID_str'] = dfTPminimal['orderID'].astype(int).astype(str)


Categorical/Nominal variables:
# (opt A) 
# cast a string into numbers and ignore the nulls
# Using Apply
df['bionic_campaign_id'] = df['bionic_campaign_id'].apply(pd.to_numeric, errors='coerce')
#  Directly with 'to_numeric' and downcasting to integer
df['bionic_campaign_id'] = pd.to_numeric(df['bionic_campaign_id'], downcast='integer', errors='coerce')
# (opt B)
for feature in combined_set.columns: # Loop through all columns in the dataframe
    if combined_set[feature].dtype == 'object': # Only apply for columns with categorical strings
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes # Replace strings with an integer 



Get a PD col as numpy array:
	# My usual....
	a_returns['JPM Returns'].as_matrix()
	# This other approach:
	a_returns.values.T[0]


# Must be another way without calling datetime...
from datetime import datetime
datetime.strptime('02-07-2017', '%d-%m-%Y').date()

# From timestamp to string:
stocks.index[-1].strftime('%Y-%m-%d')

# From timestamp to string:
dateAsStr = dfTest['date'].dt.strftime('%d_%m_%Y');


# From string to date:
pd.to_datetime(thisCookieID_Clicks['Timestamp'], format='%Y-%m-%d %H:%M:%S')


#Convert from standard date format to simple numbers in a dataframe
import matplotlib.dates   as mdates	
df['Date'] = df['Date'].map(mdates.date2num);



# Date conversion
data['date'] = pd.to_datetime(data['date'])