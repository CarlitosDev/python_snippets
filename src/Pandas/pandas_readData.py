# Read and filter a csv file with PD:
# Read the csv file and get the valid dates
dfTrain = pd.read_csv(csvPath,
		parse_dates=['date'],
		low_memory=False, 
		dtype={'id':np.uint32, 'store_nbr':np.uint8, 'item_nbr': np.uint32,
		'onpromotion': np.bool, 'unit_sales': np.float32});

idxTrain = dfTrain.date >= minDate;
dfTrain = dfTrain.ix[idxTrain];


Skip rows when reading a dataframe:
train = pd.read_csv('../input/train.csv', usecols=[1,2,3,4], dtype=dtypes, parse_dates=['date'],
                    skiprows=range(1, 86672217) 



# Read excel selectring sheets and range
# Read excel F1:N7
xlsPath = '/Users/carlos.aguilar/Documents/EF_Study_Buddy/October/Content_organisation.xlsx'
df = pd.read_excel(xlsPath, sheet_name='animals', usecols='F:N')[1:7]


#Read a gz file into a DF:

import gzip
# Couldn't love PANDAS more...
with gzip.open(impressionsPath, 'rb') as fId:
    dfImpressions = pd.read_csv(fId, delimiter='\t');

*This option is even better:
dfImpressions = pd.read_csv(impressionsPath, delimiter='\t', compression='gzip');


# Import data like a pro (csv-read)
# Set up the data types and also, deal with 
# commas delimiting thousands
customerFiles = 'Customer Reviews_GB_carlos.csv'
customerFilePath = os.path.join(baseFolder, customerFiles)

dtype_dict = {'ASIN': object,
'Product Title':object,
'Number of Customer Reviews':np.int64,
'Number of Customer Reviews - Prior Period':np.int64,
'Number of Customer Reviews - Life-to-Date':np.int64,
'Average Customer Rating':np.float64,
'Average Customer Rating - Prior Period':np.float64,
'Average Customer Rating - Life-to-Date':np.float64,
'5 Stars':np.int64,
'4 Stars':np.int64,
'3 Stars':np.int64,
'2 Stars': np.int64,
'1 Star':np.int64}
df_customer_reviews = pd.read_csv(customerFilePath, skiprows=0, dtype=dtype_dict, thousands=',')





# Partial indexing
# let's read this file, set these 3 variables as Indexes
df_test = pd.read_csv(
	"/Users/carlos.aguilar/Documents/Kaggle Competition/Grocery Sales Forecasting/test.csv", usecols=[0, 1, 2, 3, 4],
	dtype={'onpromotion': bool},
	parse_dates=["date"]  # , date_parser=parser
).set_index(['store_nbr', 'item_nbr', 'date'])
# get the index values so we can perform search, etc operations
storeValues = df_test.index.get_level_values(0)
itemValues = df_test.index.get_level_values(1)
dateValues = df_test.index.get_level_values(2)
# get the data for this particular item
idxItem = itemValues == 310671;
df_test = df_test[idxItem]



# Read files and apply functions and filters
df_train = pd.read_csv(
    '/Users/carlos.aguilar/Documents/Kaggle Competition/Grocery Sales Forecasting/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909))




#Read DF from excel:
pd.read_excel(io, sheet_name=0, ...)


# from FEATHER to Pandas
from pmmif import featherpmm
filePath = '/Users/carlos.aguilar/Documents/BeamlyRepos/pyDataAnomaly/oned/data/items.feather'
df       = featherpmm.read_dataframe(filePath).df




# Read PARQUET datasets

# (a) Using parquet Arrow - dataset
import pyarrow.parquet as pq
parquetPath = '/Volumes/CarlosBackU/Beamly/Adform parquet/test/'
dataset  = pq.ParquetDataset(parquetPath)

# (b) One file
fileA = '/Volumes/CarlosBackU/Beamly/Adform parquet/TrackingPoints/Trackingpoint_70375.parquet'
parquet_file      = pq.ParquetFile(fileA)
print(parquet_file.schema)
df = parquet_file.read().to_pandas()

# Using Dask
import dask.dataframe as dd
df = dd.read_parquet(fileA)




Read csv data:
    fileName      = imprName
    csvFile       = os.path.join(dataRoot, fileName);
    df            = pd.read_csv(csvFile, delimiter='\t')
    colsToReplace = {'BannerId-AdGroupId': 'BannerId', 'PlacementId-ActivityId': 'PlacementId'}
    df.rename(columns=colsToReplace, inplace=True)

Read csv data from the web:
    import pandas as pd
    urlSource = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df = pd.read_csv(urlSource, delimiter=';')
    df.head()



