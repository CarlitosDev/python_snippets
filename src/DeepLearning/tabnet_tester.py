

'''

deep tabular data learning


TabNet: Attentive Interpretable Tabular Learning
https://arxiv.org/pdf/1908.07442.pdf
by Google

python3 tabnet_tester.py

'''

import fcn_helpers as fhelp
#from frc_runner import frc_runner, frc_runner_model_agnostic
import pandas as pd
import datetime as dt
#from helpers_for_the_paper import run_comparison
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from ngboost import NGBRegressor


from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import os



numericalVars = ['total_stores', 'for_price','promoCurrencyDiscount',
'baseline_price', 'total_store_revenue']
categoricalVars = ['meta_store_id', 'offer_type', 'nfor', 'product_sub_group_code']
target = 'wk1_sales_all_stores'

input_vars_for_modelling = numericalVars + categoricalVars

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

timedate_var='offer_start_date'
df_eng_all['offer_start_date'] = pd.to_datetime(df_eng_all['offer_start_date'].astype(str), format='%Y-%m-%d')

# From here
# https://github.com/dreamquark-ai/tabnet/blob/develop/regression_example.ipynb


categorical_columns = []
categorical_dims =  {}
for col in categoricalVars:
    print(col, df_eng_all[col].nunique())
    l_enc = LabelEncoder()
    df_eng_all[col] = df_eng_all[col].fillna("VV_likely")
    df_eng_all[col] = l_enc.fit_transform(df_eng_all[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)



features = [ col for col in input_vars_for_modelling]
cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]




# define your embedding sizes : here just a random choice
cat_emb_dim = [5, 4, 3, 4]
max_epochs = 10

clf = TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)

df_train, df_eval, df_test = \
  fhelp.time_split_datasets(df_eng_all[input_vars_for_modelling + [timedate_var]+ [target]], timedate_var, test_size = 0.20, eval_size = 0.10)


# Pass as NP array
X_train = df_train[input_vars_for_modelling].values
y_train = df_train[target].values.reshape(-1, 1)
X_valid = df_eval[input_vars_for_modelling].values
y_valid = df_eval[target].values.reshape(-1, 1)

X_test = df_test[input_vars_for_modelling].values
y_test = df_test[target]


clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
    max_epochs=max_epochs,
    patience=50,
    batch_size=1024, 
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)


preds = clf.predict(X_test)
y_true = y_test
test_score = mean_squared_error(y_pred=preds, y_true=y_true)
print(f"BEST VALID SCORE FOR {experiment_label} : {clf.best_cost}")
print(f"FINAL TEST SCORE FOR {experiment_label} : {test_score}")

# Is this actually producing interpretable results?
clf.feature_importances_
feature_importance = pd.DataFrame(clf.feature_importances_.reshape(1,-1), columns=input_vars_for_modelling).T
feature_importance.columns = ['importance']
feature_importance.sort_values(by=['importance'], inplace=True, ascending=False)

# 
explain_matrix, masks = clf.explain(X_test)


fig, axs = plt.subplots(1, 3, figsize=(20,20))
for i in range(3):
    axs[i].imshow(masks[i][:50])
    axs[i].set_title(f'mask {i}')

plt.show()



# Compare to xgboost

from xgboost import XGBRegressor

clf_xgb = XGBRegressor(max_depth=8,
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

clf_xgb.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=40,
        verbose=10)


print(clf_xgb.feature_importances_)
xgb_feature_importance = pd.DataFrame(clf_xgb.feature_importances_.reshape(1,-1), columns=input_vars_for_modelling).T
xgb_feature_importance.columns = ['importance']
xgb_feature_importance.sort_values(by=['importance'], inplace=True, ascending=False)


preds = np.array(clf_xgb.predict(X_test))
test_score_xgboost = mean_squared_error(y_pred=preds, y_true=y_test)
print(f'FINAL TEST SCORE FOR XGBoost : {test_score}')