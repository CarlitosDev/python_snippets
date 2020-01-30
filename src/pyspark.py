import os
# -     -       -       -       -
#   Initialise PySpark
# -     -       -       -       -
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext

spark = SparkSession \
        .builder \
        .appName("test") \
        .getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")


# get the context
sqlContext = SQLContext(spark.sparkContext)



# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
# get the context
sc = spark.sparkContext
# get the SQL context
sqlContext = SQLContext(sc)

currentFile = 'tpsPurchasesPrices_10_07_2018_14_33.parquet'

outputFolder = '/Users/carlos.aguilar/Documents/Beamly/Personalisation/TPS product database'

parquetFile = os.path.join(outputFolder, currentFile);
tpParquet   = parquetFile 
parquetDF   = sqlContext.read.parquet(tpParquet)

currentFile = 'tpsUnavailable_' + datetime.now().strftime(dtFormat) + '.pickle';
pckFile     = os.path.join(outputFolder, currentFile);






# Create a simple dataframe
browsersMap = spark.createDataFrame([("Explorer 3",  "Explorer"), ("Mobile Safari 11",  "Mobile Safari")],
["browser_name_version", "browser_name"])


Committed to DP-258










# -     -       -       -       -
#   Initialise PySpark
# -     -       -       -       -

# PySpark
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.utils import require_minimum_pandas_version, require_minimum_pyarrow_version
import pyspark.sql.functions as psf
from pyspark.sql.functions import col, asc

spark = SparkSession \
        .builder \
        .appName("Adform TP") \
        .getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
# get the context
sc = spark.sparkContext
# get the SQL context
sqlContext = SQLContext(sc)