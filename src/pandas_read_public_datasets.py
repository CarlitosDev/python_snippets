import pandas as pd

# https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly#
this_dataset = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv'
data = pd.read_csv(this_dataset)





# option A
data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original",
                   delim_whitespace = True, header=None,
                   names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'])
print(data.shape)
data = data.dropna()
data.head()