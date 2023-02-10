'''
	Thesis presentation.

  This script runs method 1 on the Lavazza dataset used in the presentation

  >> to run it:
  source ~/.bash_profile && python3 "/Users/carlos.aguilar/Google Drive/PythonSnippets/PhD-Helpers/presentation_run_method_1_Lavazza.py"

  >> to run it:
  cd "/Users/carlos.aguilar/Google Drive/PythonSnippets/PhD-Helpers"
  source ~/.bash_profile && python3 presentation_run_method_1_Lavazza.py


'''


# import datetime as dt
# 
# 
# import numpy as np
# import utils.data_utils as du

# import utils.utils_root as ur

# import utils.string_utils as stru

import os
import utils.file_utils as fu
import mlToolbox.preprocessingUtils as pre
from mlToolbox.nextDoorForecasterV2 import nextDoorForecasterV2
from sklearn.model_selection import train_test_split
import pandas as pd

# import matplotlib.pyplot as plt
# import sklearn.metrics as metrics
# import datetime as dt



output_path = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/ThesisPresentationVIVA/datasets'
output_file = os.path.join(output_path, 'lavazza.pickle')
df_sku_meta_store = fu.readPickleFile(output_file)
df_sku_meta_store.shape



# this is the test promo
idx_promo_test = df_sku_meta_store.promo_id == 'A32071995_50475945'
df_test = df_sku_meta_store[idx_promo_test].copy()
df_test.reset_index(drop=True, inplace=True)
df_train = df_sku_meta_store[~idx_promo_test].copy()
df_train.reset_index(drop=True, inplace=True)


# input vars
numericalVars = ['total_stores', 'for_price', 'baseline_price',
'offer_unit_price', 'total_store_revenue']
categoricalVars = ['area_price_code', 'offer_type', 'nfor', 'feature_space']
# response var
responseVar = 'wk1_sales_all_stores'

# encode categorical variables
df_A_enc, df_test_enc, catVarsMapping = pre.js_encoder(df_train, df_test, \
  categoricalVars, responseVar)


inputVars = numericalVars + [iVar + '_encoded' for iVar in categoricalVars]


X_train, X_val, y_train, y_val = \
  train_test_split(df_A_enc[inputVars].values, df_A_enc[responseVar].values, test_size=0.15, random_state=1)

num_frcs = 50
d_predictions = nextDoorForecasterV2.fit(X_train, y_train, X_val, y_val, df_test_enc[inputVars].values, num_frcs)
y_hat = d_predictions['predictions']
y_test = df_test_enc[responseVar]
errors = nextDoorForecasterV2.get_frc_errors(y_test, y_hat)
#print(f'{num_frcs} forecasters with MAPE {errors["MAPE"]:.2f} and mError {errors["meanError"]:.2f}')

predictions_std = d_predictions['predictions_std'][0]
num_neighbours = round(d_predictions['num_neighbours'])
print(f'NextDoor number of neighbours {num_neighbours}')
print(f'NextDoor forecaster predicts {y_hat[0]:.2f} pm {predictions_std:.2f}. Actual {y_test[0]:.2f}')

print('NextDoor JSON payload')
fu.printJSON(d_predictions)

df_importance = pd.DataFrame(d_predictions['features'], inputVars, columns=['importance'])
df_importance.sort_values('importance', ascending=False, inplace=True)
df_importance['importance_norm'] = df_importance['importance']/df_importance['importance'].sum()
df_importance.reset_index(inplace=True)

fu.to_random_excel_file(df_importance, writeIndex=True)