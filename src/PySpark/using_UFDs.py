
'''
	This is from the DBricks guys but they mention is does not work, but it's kind of enough to get an idea of UDFs
'''

from pyspark.sql.functions import pandas_udf
from pyspark.sql import Window
import pandas as pd

@pandas_udf("double")
def q025_udf(v: pd.Series) -> float:
    quan = v.mean(q=0.25)
    
    return 0.1
  
  
df = (spark.readStream
  .format('delta') \
  .option('includeTimestamp', 'true')\
  .option('ignoreChanges', 'true')\
  .table(elb2c_silver_tbl) \
  .withColumn("startTime", to_timestamp(col("startTime")))
  .withColumn("completionTime", to_timestamp(col("completionTime")))

  .withWatermark("completionTime", watermark_stride) 
     )

df_agg = df.groupBy("templateId", window("completionTime", window_length, window_slide)).agg(q025_udf("score"))
display(df_agg)