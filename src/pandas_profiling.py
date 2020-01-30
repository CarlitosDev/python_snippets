Pandas profiler:
pip3 install pandas-profiling
# A play-around with it
import numpy as np
import pandas as pd
import pandas_profiling
df = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], \
				   'B': ['B0', 'B1', 'B2', 'B3'], \
				   'C': [1,5,32,5]})

# introduce a highly correlated 'D' and 'E' variables
df['D'] = df.C + 5
df['E'] = df.C + np.random.random_sample((1,4)).squeeze()

profile = pandas_profiling.ProfileReport(df)

print(profile.get_description())

# To retrieve the list of variables which are rejected due to high correlation
rejected_variables = profile.get_rejected_variables(threshold=0.8)
print(str(rejected_variables) + ' are higly correlated to another variable in the DS')



# It is designed to work in Jupyter
import pandas as pd
import pandas_profiling
df = pd.DataFrame([{'A': 'foo', 'B': 'green', 'C': 11}, \
				   {'A':'bar', 'B':'blue', 'C': 20}])
profile = pandas_profiling.ProfileReport(df)
# but we can output the file and read it from a browser
profile.to_file(outputfile="//Users/carlos.aguilar/Downloads/myoutputfile.html")