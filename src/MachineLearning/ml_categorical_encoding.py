'''

This guy has an article where he goes through the most common categorical encodings (Dec 2019)
https://github.com/DenisVorotyntsev/CategoricalEncodingBenchmark

HelmertEncoder	0.9517
SumEncoder	0.9434
FrequencyEncoder	0.9176

CatBoostEncoder	0.9726
OrdinalEncoder	0.9694
HelmertEncoder	0.9558

JamesSteinEncoder	0.9918
CatBoostEncoder	0.9917
TargetEncoder	0.9916

'''

import category_encoders as ce
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'x_0': ['a'] * 5 + ['b'] * 5,
    'x_1': ['a'] * 9 + ['b'] * 1
})

this_map = {'a': 6, 'b':9}
df['y'] = df['x_0'].map(this_map) + df['x_1'].map(this_map) + 3.5*np.random.random(df.shape[0])


target_var = 'y'
cat_vars = ['x_0', 'x_1']

# JamesSteinEncoder
encoder_js = ce.JamesSteinEncoder(cols=cat_vars, verbose=1)
df_A = encoder_js.fit_transform(df, df[target_var])

# CatBoostEncoder
encoder_cb = ce.CatBoostEncoder(cols=cat_vars)
df_B = encoder_cb.fit_transform(df, df[target_var])

# TargetEncoder
encoder_te = ce.TargetEncoder(cols=cat_vars)
df_C = encoder_te.fit_transform(df, df[target_var])




#pip3 install -U  dirty_cat
