'''

From PARQUET to PANDAS
https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow

Notes:
- Apache Arrow is an in-memory columnar data format that is used in Spark to efficiently transfer data between JVM and Python processes.
- If an error occurs during createDataFrame(), Spark will fall back to create the DataFrame without Arrow.
- Pandas UDFs are user defined functions that are executed by Spark using Arrow to transfer data and Pandas to work with the data. Currently, there are two types of Pandas UDF: Scalar and Grouped Map.
    - Scalar Pandas UDFs are used for vectorizing scalar operations. They can be used with functions such as select and withColumn. (example >  create a scalar Pandas UDF that computes the product of 2 columns)
    - 

Reading Multiples Files and Partitioned Datasets:



Definitions:

* Column chunk: A chunk of the data for a particular column. They live in a particular row group and are guaranteed to be contiguous in the file.
Contiguous data for a single column. Made of data pages and an optional dictionary page.

* Row group: A logical horizontal partitioning of the data into rows. There is no physical structure that is guaranteed for a row group. A row group consists of a column chunk for each column in the dataset. Made of column chunks.
Filtering by row group: Use footer stats and dictionary pages to eliminate row groups. Row group filtering works in parallel.

* Page: Column chunks are divided up into pages. A page is conceptually an indivisible unit (in terms of compression and encoding). There can be multiple page types which are interleaved in a column chunk. 
Use page stats to eliminate pages.

* Parquet dataset: Multiple Parquet files constitute a Parquet dataset. These may present in a number of ways:
    - A list of Parquet absolute file paths (one big folder)
    - A directory name containing nested directories defining a partitioned dataset


Tips:

- Avoid encoding fallback by increasing the dictionary size: parquet.dictionary.page.size (2-3 MB)
- Decrease row group size: parquet.block.size 24,32 or 64MB.
- Brotli compression developed by Google.
- Total memory for writes is approximately the row group size times the number of open files. If this is too high, then processes die with OutOfMemoryExceptions.
- You will have a much faster experience if you store your data so that the grouping column is sorted in parquet.

Resources: 

- Supported datatypes (https://spark.apache.org/docs/latest/sql-programming-guide.html#supported-sql-arrow-types)
- See https://arrow.apache.org/docs/python/parquet.html
- https://www.slideshare.net/RyanBlue3/parquet-performance-tuning-the-missing-guide
- http://ingest.tips/2015/01/31/parquet-row-group-size/


PySpark tips:
- SparkContext tells Spark how and where to access a cluster. 
- Explode: Returns a new row for each element in the given array or map.
- Available functions: http://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html#module-pyspark.sql.functions
- Use withColumn to add a new variable to the dataframe
- Operation on Pyspark DataFrame run parallel on different nodes in cluster but, in case of pandas it is not possible.
- Complex operations in pandas are easier to perform than Pyspark DataFrame


'''

import pandas as pd
import numpy as np
import os
import utils.carlosUtils as cu
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql.utils import require_minimum_pandas_version, require_minimum_pyarrow_version

require_minimum_pandas_version()
require_minimum_pyarrow_version()

# ---------
# READING
# ---------

# Using parquet Arrow
parquetPath = '/Volumes/CarlosBackU/Beamly/Adform parquet/test/'
dataset  = pq.ParquetDataset(parquetPath)

# Using Dask
import dask.dataframe as dd
df = dd.read_parquet('/Volumes/CarlosBackU/Beamly/Adform parquet/TrackingPoints/Trackingpoint_70375.parquet')


parquetPath =  '/Volumes/CarlosBackU/Beamly/Adform parquet/quarantine2/'
dataset     = pq.ParquetDataset(parquetPath)
print(dataset.schema)


parquetPath =  '/Volumes/CarlosBackU/Beamly/Adform parquet/test/'
dataset     = pq.ParquetDataset(parquetPath)
print(dataset.schema)

testerParquetFile = '/Volumes/CarlosBackU/Beamly/Adform parquet/Impressions/Trackingpoint_98795.parquet' 
parquet_file      = pq.ParquetFile(testerParquetFile)
print(parquet_file.schema)
df = parquet_file.read().to_pandas()

df = pd.DataFrame({'A':{'d', 'dr'}, 'B': {'gg', 'ddd'}})
df2 = pd.DataFrame({'B': {'gg', 'ddd'}, 'C':{'d', 'dr'}})

df = df.reindex_axis(sorted(df.columns), axis=1)


# give it a go...
import pyarrow as pa
import pyarrow.parquet as pq


# FROM PANDAS TO PARQUET (PYARROW)
import pyarrow as pa
arrowTable    = pa.Table.from_pandas(df)

engine        = 'pyarrow',
flavor        = 'spark'
pq.write_table(arrowTable, parquetFile, flavor=flavor);
# (ii)
testerParquetFile = '/Volumes/CarlosBackU/Beamly/Adform parquet/quarantine2/' + 'Trackingpoint_103535.parquet' 
parquetFile       = '/Volumes/CarlosBackU/Beamly/Adform parquet/test2/' + 'Trackingpoint_103535.parquet'
parquet_file      = pq.ParquetFile(testerParquetFile)
df = parquet_file.read().to_pandas()
df = df.reindex_axis(sorted(df.columns), axis=1)
arrowTable = pa.Table.from_pandas(df)
engine        = 'pyarrow',
flavor        = 'spark'
pq.write_table(arrowTable, parquetFile, flavor=flavor);





# Tester
testerParquetFile = '/Volumes/CarlosBackU/Beamly/Adform parquet/Impressions/' + 'Impression_86312.parquet'
parquet_file      = pq.ParquetFile(testerParquetFile)

print(parquet_file.schema)
df2 = parquet_file.read().to_pandas()
df = df.reindex_axis(sorted(df.columns), axis=1)


print(parquet_file.metadata)

df = parquet_file.read().to_pandas()
df.head()









# -     -       -       -       -       -
# (i) Tests through PySpark
# -     -       -       -       -       -
spark = SparkSession \
        .builder \
        .appName("Python Arrow-in-Spark example") \
        .getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# Generate a Pandas DataFrame
pdf = pd.DataFrame(np.random.rand(100, 3))

# Create a Spark DataFrame from a Pandas DataFrame using Arrow
df = spark.createDataFrame(pdf)

# Convert the Spark DataFrame back to a Pandas DataFrame using Arrow
result_pdf = df.select("*").toPandas()


# Kaizen Day (23/04/2018)
# >>> Use PySpark to read the parquet files
import pandas as pd
import numpy as np
import os
import utils.carlosUtils as cu
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import col, asc
sc = SparkContext(appName="PythonTester")

sqlContext = SQLContext(sc)

thisDSet  = "/Volumes/CarlosBackU/Beamly/Adform parquet/TrackingPoints/"
#thisDSet  = "/Volumes/CarlosBackU/Beamly/Adform parquet/test2/"
parquetDF = sqlContext.read.parquet(thisDSet)

# Get the schema
parquetDF.printSchema()
# See the head
parquetDF.head()

# Inspect some columns
parquetDF.select('CookieID','DeviceTypeID').show(30)

# Get the count of distinct CookiesId
parquetDF.select('CookieID').distinct().count()

# Pairwise frquency table
parquetDF.crosstab('CookieID','DeviceTypeID').show()


# Basic ETL: drop duplicates, fillnans or drop nans
b = parquetDF.dropDuplicates()
c = parquetDF.select('CookieID').fillna(-1)
d = parquetDF.select('CookieID').dropna()

# More basic ETL's: filtering
# Remove the CookieID with value 0
fromParquet = parquetDF.filter(parquetDF.CookieID != 0)
fromParquet.show()

# Let's try the one that we run in Qubole
tpTPS_1 = "29878294"
tpTPS_2 = "29878294"
tpFD_1  = "27839235"
tpFD_2  = "33126038"
dfAdform = parquetDF.filter((col("TrackingPointId") == tpTPS_1) 
    | (col("TrackingPointId") == tpTPS_2) 
    | (col("TrackingPointId") == tpFD_1) 
    | (col("TrackingPointId") == tpFD_2)).filter(col("IsRobot") == "No").filter(col("CookieID") != 0)
    
dfAdform.head()

dfCrosstab = dfAdform.crosstab('CookieID','DeviceTypeID').show()

# Add a new column
dfAdform = dfAdform.withColumn('sameHour', dfAdform.hour)
dfAdform.printSchema()

# Get the different CrossDeviceData
a = dfAdform.select('CrossDeviceData').distinct().show()

#dfAdform.select("CookieID").filter(col("CrossDeviceData"))
#dfAdform2 = dfAdform.select("CookieID").subtract()

# Apply SQL Queries on DataFrame
dfAdform.registerTempTable("dfAdformSQL")
sqlQuery ="""SELECT 
TO_DATE(CAST(UNIX_TIMESTAMP(dfAdformSQL.date, 'dd-MM-yyyy') AS TIMESTAMP)) as date,
CookieID,
DeviceTypeID
from dfAdformSQL
where CookieID <> 0
group by 1,2,3
order by date asc
"""
dfProfiles = sqlContext.sql(sqlQuery)

dfProfiles.head()

# Convert to PANDAS
dfProfilesPD   = dfProfiles.toPandas()
resultsName    = 'conversionsTPSandFD'
resultsFolder  = '/Volumes/CarlosBackU/Beamly/Adform processed/' + resultsName
cu.createFolder(resultsFolder);
currentResults = os.path.join(resultsFolder, resultsName + '.pickle');
cu.dataFrameToPickle(dfProfilesPD, currentResults);


# Use the normal functionality
devicesMeta = rd.getDevicesMeta(metaRoot);











# -     -       -       -       -       -
# (ii) Tests through PyArrow
# -     -       -       -       -       -
# Another example. From a parquet file to pandas
import pyarrow.parquet as pq
df = pq.read_table('<filename>').to_pandas()
# Only read a subset of the columns
df = pq.read_table('<filename>', columns=['A', 'B']).to_pandas()

# Read table replaces 
table = pq.read_table(parquetPath)


# Another example. From PD to parquet
import pyarrow as pa
import pyarrow.parquet as pq
table = pa.Table.from_pandas(data_frame, timestamps_to_ms=True)
#  Snappy compression by default
pq.write_table(table, '<filename>')



import pyarrow.parquet as pq

pq.write_table(dataset, out_path, use_dictionary=True,
               compression='snappy)
# Parallel reads
table = pq.read_table(file_path, nthreads=4)



# From WesM
from pyarrow.compat import guid
import pyarrow as pa
import pyarrow.parquet as pq
import snappy

def write_to_parquet(df, out_path, use_dictionary=True,
                     compression='SNAPPY'):
    arrow_table = pa.Table.from_pandas(df)
    
    if compression.lower() == 'uncompressed':
        compression = None
    
    pq.write_table(arrow_table, out_path, use_dictionary=use_dictionary,
                   compression=compression)

def read_pyarrow(path, nthreads=1):
    return pq.read_table(path, nthreads=nthreads).to_pandas()




# Another example
dfSPRK = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))

dfSPRK.show(2)



# Examples just using PyArrow
import pyarrow as pa
import pyarrow.parquet as pq

df = pd.DataFrame({'one': [-1, np.nan, 2.5],
                'two': ['foo', 'bar', 'baz'],
                'three': [True, False, True]})


readFromExternalDrive = False

if readFromExternalDrive:
# If using my external drive
    baseFolder    = '/Volumes/CarlosBackU/Beamly/Adform/TrackingPoints'
    summaryFolder = '/Volumes/CarlosBackU/Beamly/Adform/summaryPickle'
else:
    baseFolder    = '/Users/carlos.aguilar/Documents/Beamly/Personalisation/issuesMasterDS/trackingPoints'


baseFolder         = '/Users/carlos.aguilar/Documents/Beamly/Personalisation/issuesMasterDS/trackingPoints'
currentFile        = 'Trackingpoint_85936.csv'
trackingPointsPath = os.path.join(baseFolder, currentFile);
#dfTrackingPoints = pd.read_csv(trackingPointsPath, delimiter='\t', compression='gzip');
dfTrackingPoints = pd.read_csv(trackingPointsPath, delimiter='\t', compression=None, parse_dates=True);

arrowTable = pa.Table.from_pandas(dfTrackingPoints)

pathToParquet = '/Users/carlos.aguilar/Documents/Beamly/Personalisation/issuesMasterDS/parquetTest'
parquetPath   = os.path.join(pathToParquet, 'example.parquet');

# Sanitize field characters unsupported by Spark SQL
pq.write_table(arrowTable, parquetPath, flavor='spark')

# Read the file
parquet_file = pq.ParquetFile(parquetPath)
print(parquet_file.metadata)
print(parquet_file.schema)


# Pandas calling PyArrow (same filesize)
parquetPathPD   = os.path.join(pathToParquet, 'examplePD.parquet');
dfTrackingPoints.to_parquet(parquetPathPD, engine='pyarrow', flavor='spark')




# ---------------------- From S3...
# Using PyArrow fails...
import pandas as pd
import pyarrow.parquet as pq
import s3fs
# It returns...Error: Found files in an intermediate directory: s3://beamly-data-qubole-prod/data/masterdataset/adform/reporting/dailyProfilesTPSandFD.parque
thisDSet = "s3://beamly-data-qubole-prod/data/masterdataset/adform/reporting/dailyProfilesTPSandFD.parquet"
s3       = s3fs.S3FileSystem()
dataset  = pq.ParquetDataset(thisDSet, filesystem=s3)



# ---------------------- From the external drive S3...
# Using PyArrow fails...
import pandas as pd
import pyarrow.parquet as pq
thisDSet = "/Volumes/CarlosBackU/Beamly/Adform parquet/TrackingPoints/"
dataset  = pq.ParquetDataset(thisDSet)


# ---------------------- read one file from the external drive S3...
import pandas as pd
import pyarrow.parquet as pq
thisDSet = '/Volumes/CarlosBackU/Beamly/Adform parquet/TrackingPoints/' + 'Trackingpoint_90647.parquet'
# read the dataset
dataset  = pq.read_table(thisDSet, nthreads = 3).to_pandas()
# read just one column
dataset  = pq.read_table(thisDSet, columns=['customvars', 'CampaignId'], nthreads = 3).to_pandas()






# Arrow error: IOError: An error occurred (AccessDenied) when calling the GetObject operation: Access Denied
thisDSet = "s3://beamly-data-qubole-prod/data/masterdataset/adform parquet/tracking points"
s3       = s3fs.S3FileSystem()
dataset  = pq.ParquetDataset(thisDSet, filesystem=s3)

# This one works...
s3       = s3fs.S3FileSystem()
thisDSet = "s3://beamly-data-qubole-prod/data/masterdataset/adform parquet/tracking points"
bucketList = s3.ls(thisDSet)
bucketList[32]

# FileNotFoundError ....
import dask.dataframe as dd
fileName = "s3://" + bucketList[32]
df = dd.read_parquet(fileName)


# Using fastparquet

import s3fs
import fastparquet as fp
s3 = s3fs.S3FileSystem()
fs = s3fs.core.S3FileSystem()
# Try with the engineering account...
fileName = "s3://beamly-metrics-data-stage/dataScience/Trackingpoint_100086.parquet"
myopen = s3.open
all_paths_from_s3 = fs.glob(path=fileName)
fp_obj = fp.ParquetFile(fileName, open_with=myopen)



from subprocess import call
call(['awsauth', 'datascience'])

# Using fastparquet
import s3fs
import fastparquet as fp
s3 = s3fs.S3FileSystem()
fs = s3fs.core.S3FileSystem()

s3_path = "s3://beamly-data-qubole-prod/data/masterdataset/adform/reporting/dailyProfilesTPSandFD.parquet/*/*.parquet"
all_paths_from_s3 = fs.glob(path=s3_path)
myopen = s3.open
#use s3fs as the filesystem
fp_obj = fp.ParquetFile(all_paths_from_s3, open_with=myopen)
#convert to pandas dataframe
df = fp_obj.to_pandas()


# doesn't work
import os
import dask.dataframe as dd
os.environ['AWS_SHARED_CREDENTIALS_FILE'] = "~/.aws/credentials"
df = dd.read_parquet('s3://' + all_paths_from_s3[4], storage_options={'anon': True, 'use_ssl': False})


a = "s3://beamly-data-qubole-prod/data/masterdataset/adform/reporting/dailyProfilesTPSandFD.parquet/date%3D2017-06-14/part-00000-676b06f9-3d03-48c3-bedf-3ac1d5012894.c000.snappy.parquet"
df = dd.read_parquet(a, storage_options={'anon': True, 'use_ssl': False})
df.head()

#  This one works
# Try with the engineering account...
fileName = "s3://beamly-metrics-data-stage/dataScience/Trackingpoint_100086.parquet"
df = dd.read_parquet(fileName, engine='pyarrow')
df.head()



# Try with the DATA account...
fileName2 = "s3://beamly-data-qubole-prod/data/masterdataset/adform parquet/tracking points/Trackingpoint_100086.parquet"
df2 = dd.read_parquet(fileName2, engine='pyarrow')
df2.head()


# Daskn doesn't work with snappy!
import os
import pyarrow.parquet as pq
testerParquetFile = '/Users/carlos.aguilar/Documents/Beamly/SBEDAP/developers files/Trackingpoint_98795.parquet' 
parquet_file      = pq.ParquetFile(testerParquetFile)
print(parquet_file.metadata)
print(parquet_file.schema)
df = parquet_file.read().to_pandas()
df.head()