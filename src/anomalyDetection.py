Anomaly Detection workshop
--------------------------
log-log plots to spot outliers (kind of the Bland-Altman plot)
Challenge the data by defining constraints: negative discounts, nulls, nans, etc

The main idea of this talk is to find problems in the data through unit testing principles. By setting up
a file with the constraints that the variables should follow, we can easily find anomalies in the data.
To do that, the guy uses his own library called 'tdda'. The cool thing about this library is that given a file
with the expected data, it is able to work out the present values, the range, etc...

Tester:

import numpy as np
import pandas as pd
from pmmif import featherpmm
from tdda.constraints import discover_df
df = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3']})
constraints = discover_df(df, inc_rex=True)
json_constraints =  constraints.to_json()
print(json_constraints)



End of Anomaly Detection workshop
--------------------------