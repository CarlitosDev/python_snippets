# This is based (mainly the code is copied) from Andreas Mueller

import pandas as pd
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt

# Let's try for classification
data = datasets.load_iris()
df_iris = pd.DataFrame(data.data, columns=data.feature_names)
df_iris['target'] = pd.Series(data.target)


# Get the numerical columns
df_types = df_iris.dtypes
cols = df_iris.columns[df_types!=object].tolist()
df = df_iris.loc[:, cols].melt('target')

# I think this works better with a classification problem...
g = sns.FacetGrid(df, col='variable', hue='target', \
  sharey=False, sharex=False, col_wrap = 2)

g = g.map(sns.kdeplot, 'value', shade=True)
g.axes[0].legend()
plt.show()