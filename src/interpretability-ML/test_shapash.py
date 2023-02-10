test_shapash.py

'''

Shapash is a Python library which aims to make machine learning interpretable and understandable by everyone. 
It provides several types of visualization that display explicit labels that everyone can understand.
Data Scientists can more easily understand their models and share their results. 
End users can understand the decision proposed by a model using a summary of the most influential criteria


  cd '/Users/carlos.aguilar/Documents/DS_repos'
  git clone https://github.com/MAIF/shapash
  cd shapash
  source ~/.bash_profile && python3 setup.py install

'''





import pandas as pd
import datetime as dt
#from helpers_for_the_paper import run_comparison
from catboost import CatBoostRegressor
from category_encoders import OrdinalEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt


numericalVars = ['total_stores', 'for_price','promoCurrencyDiscount',
'baseline_price', 'total_store_revenue']
categoricalVars = ['meta_store_id', 'offer_type', 'nfor', 'product_sub_group_code']
target = 'wk1_sales_all_stores'
timedate_var='offer_start_date'

input_vars_for_modelling = numericalVars + categoricalVars
all_selected_vars = input_vars_for_modelling + [timedate_var]+ [target]

####
# Grocery
####
experiment_label = 'grocery'
baseFolder = '~/Google Drive/order/Machine Learning Part/data/data From the UK (by folder)'
baseFile   = 'dataGrocery2/dataBTID_Grocery 2_fromSeptToNov2k16.pickle'

data_filepath = os.path.join(baseFolder, baseFile)
df_eng_all = pd.read_pickle(data_filepath)

df_eng_all.shape
df_eng_all.iloc[23]


df_eng_all['offer_start_date'] = pd.to_datetime(df_eng_all['offer_start_date'].astype(str), format='%Y-%m-%d')





from category_encoders import OrdinalEncoder
encoder = OrdinalEncoder(
    cols=categoricalVars,
    handle_unknown='ignore',
    return_df=True).fit(df_eng_all[all_selected_vars])

X_df=encoder.transform(df_eng_all[all_selected_vars])

import mlToolbox.preprocessingUtils as pre
df_train, df_eval, df_test = \
  pre.time_split_datasets(X_df, timedate_var, test_size = 0.20, eval_size = 0.10)


from xgboost import XGBRegressor
reg_xgb = XGBRegressor(max_depth=8,
    learning_rate=0.1,
    n_estimators=1000,
    verbosity=0,
    silent=None,
    objective='reg:linear',
    booster='gbtree',
    n_jobs=-1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None)


reg_xgb.fit(df_train[input_vars_for_modelling], df_train[target],
        eval_set=[(df_eval[input_vars_for_modelling], df_eval[target])],
        early_stopping_rounds=40,
        verbose=10)


y_pred = reg_xgb.predict(df_test[input_vars_for_modelling])
y_hat = pd.DataFrame(y_pred, columns=['y_hat'], index=df_test[target].index)

# Explainable 
from shapash.explainer.smart_explainer import SmartExplainer


# The features must be a dictionary with key and description

features_dict = {k: 'info '+ k for k in input_vars_for_modelling}
xpl = SmartExplainer(features_dict=features_dict)

xpl.compile(
    x=df_test[input_vars_for_modelling],
    model=reg_xgb,
    preprocessing=encoder, # Optional: compile step can use inverse_transform method
    y_pred=y_hat.astype(float) # Optional. Must be a pandas obj. Enforcing data types
)

xpl.plot.features_importance()
plt.show()


xpl.plot.contribution_plot("OverallQual")

xpl.plot.contribution_plot("Second floor square feet")