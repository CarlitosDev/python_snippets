

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer



from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target

# Take only 2 features to make visualization easier
# Feature of 0 has a long tail distribution.
# Feature 5 has a few but very large outliers.

X = X_full[:, [0, 5]]

distributions = [
    ('Unscaled data', X),
    ('Data after standard scaling',
        StandardScaler().fit_transform(X)),
    ('Data after min-max scaling',
        MinMaxScaler().fit_transform(X)),
    ('Data after max-abs scaling',
        MaxAbsScaler().fit_transform(X)),
    ('Data after robust scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
    ('Data after power transformation (Yeo-Johnson)',
     PowerTransformer(method='yeo-johnson').fit_transform(X)),
    ('Data after power transformation (Box-Cox)',
     PowerTransformer(method='box-cox').fit_transform(X)),
    ('Data after quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(X)),
    ('Data after quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(X)),
    ('Data after sample-wise L2 normalizing',
        Normalizer().fit_transform(X)),
]






# Using a wrapper
import mlToolbox.preprocessingUtils as mlPreproc
import pandas as pd
import numpy as np
dfNorm = mlPreproc.labelEncoding(inputDataset, categoricalVars)

df = pd.DataFrame({
    'x': pd.Series([43,41,90,None,23, -19]),
    'y': pd.Series([54,23,2314,4312,41, -19]),
    'z': [80 +40 * np.random.random() for _  in range(6)]
    })


df_ext = mlPreproc.column_transformation(df, ['x', 'z'], type_transformation='Normalizer');

StandardScaler
MinMaxScaler
MaxAbsScaler
Normalizer

import importlib

prc = importlib.util.find_spec('PowerTransformer', package='sklearn.preprocessing')


# it does not work
prc = importlib.import_module('MinMaxScaler', package='sklearn.preprocessing')


loader = importlib.find_loader('sklearn.preprocessing')
print('Loader:', loader)
m = loader.load_module()
print('Module:', m)


pkg_loader = importlib.find_loader('sklearn.preprocessing')
pkg = pkg_loader.load_module()



loader = importlib.find_loader('MinMaxScaler', pkg.__path__)
print('Loader:', loader)

m = loader.load_module()
print('Module:', m)



module = importlib.import_module('redberry.models.migrations.%s' % migration_file)


from sklearn.preprocessing import MinMaxScaler

importlib.import_module('sklearn.preprocessing')
importlib.import_module('sklearn.preprocessing')


from sklearn import preprocessing
exec('prc=preprocessing.MinMaxScaler()')