df_unit_completed = spark\
    .readStream\
    .format('delta')\
    .option('includeTimestamp', 'true')\
    .option('ignoreChanges', 'true')\
    .load(silver_table)\
    .withColumn("startTime", to_timestamp(col("startTime")))\
    .withColumn("__occurred", to_timestamp(col("__occurred")))


# The field in withColumn can be a new variable created from the cast operation