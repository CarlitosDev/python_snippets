'''
	polars_tester.py

	Alledgelly faster alternative to Pandas based on Arrow.

'''


pip3 install 'polars[numpy,pandas,pyarrow]'  # install a subset of all optional dependencies




# Read directoly from Deltalake 
# https://delta.io/blog/2022-12-22-reading-delta-lake-tables-polars-dataframe/
import polars as pl
print(pl.read_delta("/tmp/bear_delta_lake"))

import polars as pl
table_path = "/path/to/delta-table/"

# `scan_delta` example
ldf = pl.scan_delta(table_path).collect()  
print(ldf)

# `read_delta` example
df = pl.read_delta(table_path, version=1)  
print(df)