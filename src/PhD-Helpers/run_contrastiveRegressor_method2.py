'''
cd "/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/contrastive explanations/"

source ~/.bash_profile && python3

'''

'''
  Runner:
  python3 run_comparisons_for_the_paper.py

  python3 '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/contrastive explanations/run_comparisons_for_the_paper_multi_contrastive.py'

'''

import fcn_helpers as fhelp
from frc_runner import frc_runner
import pandas as pd
import os
import datetime as dt
import utils.file_utils as fu

numericalVars = ['total_stores', 'for_price', 'baseline_price',
'offer_unit_price', 'total_store_revenue', 'offer_start_date_num']
categoricalVars = ['area_price_code', 'offer_type', 'nfor', 'feature_space']


# Hyper-parameters
num_neighbours = 4
validation_test_size = 0.1
num_iterations = 500
learning_rate  = 0.1
depth = 12

feat_importance_keyword = 'feature_importances_'


####
# Lavazza pods
####
experiment_label = 'lavazza pods'
baseFolder = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/ThesisPresentationVIVA/datasets'
baseFile = 'm2_lavazza_coffee_pods.pickle'
baseFilePath = os.path.join(baseFolder, baseFile)

base_product_number_std = '76496356'


# Read and filter
df_promos = pd.read_pickle(baseFilePath)


df_promos['offer_start_date_num'] = \
  (pd.to_datetime(df_promos.offer_start_date, format='%Y-%m-%d') - \
    pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')


# map dates
df_mapper = df_promos[['offer_start_date_num', 'offer_start_date']].drop_duplicates()
dct_mapper = dict(zip(df_mapper['offer_start_date_num'].tolist(),
df_mapper['offer_start_date'].tolist()))






sales_threshold = 20
stores_threshold = 10
idx_A = df_promos.wk1_sales_all_stores > sales_threshold 
idx_B = df_promos.total_stores > stores_threshold
df_promos = df_promos[idx_A & idx_B].copy()

responseVar = 'wk1_sales_all_stores'

frc_results = frc_runner(df_promos, base_product_number_std,
  categoricalVars, numericalVars,
  num_neighbours, validation_test_size,
  num_iterations, learning_rate, depth,
  feat_importance_keyword = feat_importance_keyword,
  experiment_label = experiment_label, responseVar=responseVar, 
  doVisualisation=False, doMaskBPNS=True)


data_to_save = {k:v for k,v in frc_results.items() if k not in 'contrastiveRegressor'}
resultsFile = 'lavazza_pods_forecast_2.pickle'
resultsFilePath = os.path.join(baseFolder, resultsFile)
fu.toPickleFile(data_to_save, resultsFilePath)

selected_dates = [1438128000,1430870400,1438128000,1430870400,1474416000]

for this_key in selected_dates:
  print(dct_mapper[this_key].strftime('%d-%m-%Y'))


# As a forecaster, I don't really like that the results are driven by the number of stores.
numericalVars = ['total_stores', 'for_price', 'baseline_price',
'offer_unit_price', 'total_store_revenue']
categoricalVars = ['area_price_code', 'offer_type', 'nfor', 'feature_space']




# For reference
df_promos['promo_identifier_latex'] = df_promos.offer_number + '_' + df_promos.base_product_number_std + \
  ' (' + df_promos.offer_description + ')'

fu.to_random_excel_file(df_promos)



import utils.file_utils as fu
frc_results.keys()
fu.printJSON(frc_results)
fu.printJSON(frc_results['mapObfuscatedVars'])
fu.printJSON(frc_results['catVarsMapping'])

contrastiveReg = frc_results['contrastiveRegressor']
idx_review = 0


frc_results['cold_start_results'].keys()


X_test = frc_results['X_test']
y_test = frc_results['y_test']
contrastiveReg.get_contrastive_explanation(idx_review, X_test, y_test)

X_test: 'DF', y_test: 'pd.Series'


contrastiveReg.get_contrastive_explanation(idx_review, frc_results['df_test'], y_test)

frc_results['df_test'].iloc[0]

# MISSING THE CATEGORICAL ENCODER MAPPING
contrastiveReg.inputVars
contrastiveReg.results['df_feat_importances'][contrastiveReg.inputVars]
contrastiveReg.results['df_feat_importances']
print('\n'.join(contrastiveReg.inputVars))


'''
Passing list-likes to .loc or [] with any missing labels is no longer supported.
The following labels were missing: 
Index(['meta_store_id_encoded', 'offer_type_encoded', 'nfor_encoded'], 
dtype='object'). 
'''

frc_results['cold_start_results']

idx_review = 0
idx_closest_promos = frc_results['cold_start_results']['y_idx_closest_promos'][idx_review]

df_feature_importances = frc_results['cold_start_results']['df_feat_importances']#.loc[self.inputVars]
df_feature_importances.columns = ['var_importance']
df_feature_importances.sort_values(by='var_importance', ascending=False, inplace=True)

sorted_columns = df_feature_importances.index.tolist()

df_forecast = frc_results['df_train'].loc[idx_closest_promos][sorted_columns]
df_forecast['y_train'] = frc_results['df_train'].loc[idx_closest_promos, responseVar]


df_forecast['delta_y_train'] = frc_results['cold_start_results']['y_delta_list'][idx_review]
df_forecast['y_train_plus_delta'] = frc_results['cold_start_results']['y_k_all_list'][idx_review]
df_forecast['y_train_distances'] = frc_results['cold_start_results']['y_distances_closest_promos'][idx_review]
df_forecast['y_train_weights'] = frc_results['cold_start_results']['y_weights'][idx_review]

df_forecast.index = [f'neighbour_{idx}' for idx in range(0, contrastiveReg.num_neighbours)]


y_test = frc_results['y_test']

df_target = frc_results['df_test'].iloc[idx_review][sorted_columns]
df_target['y_actual'] = y_test[idx_review]
df_target['y_forecast'] = frc_results['cold_start_results']['y_hat'][idx_review]
df_target['y_weighted_forecast'] = frc_results['cold_start_results']['y_hat_weighted'][idx_review]
df_target.columns = f'y_test_{idx_review}'

df_feature_importances = frc_results['cold_start_results']['df_feat_importances'].loc[sorted_columns]
df_feature_importances.columns=['var_importance']
df_forecast_ext = df_feature_importances.T.append(df_forecast.append(df_target, sort=False))


import utils.file_utils as fu
fu.to_random_excel_file(df_forecast_ext)



