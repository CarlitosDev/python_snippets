'''
	presentation_example_method_1.py
'''


import datetime as dt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import utils.data_utils as du
import utils.file_utils as fu
import utils.utils_root as ur

import utils.string_utils as stru


from mlToolbox.nextDoorForecasterV2 import nextDoorForecasterV2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import mlToolbox.preprocessingUtils as pre

#import utils.aws_data_utils as awsu
#import utils.nlp_utils as nlpu
#import utils.image_utils as imgu



experiment_label = 'grocery'
baseFolder = '~/Google Drive/order/Machine Learning Part/data/data From the UK (by folder)'


# let's combine the datasets??
baseFile   = 'dataGrocery1/dataBTID_Grocery 1_fromSeptToNov2k16.pickle'
dataFile = os.path.expanduser(os.path.join(baseFolder, baseFile))
df_promos_A = pd.read_pickle(dataFile)

baseFile   = 'dataGrocery1/dataBTID_Grocery 2_NovToDec2k16.pickle'
dataFile = os.path.expanduser(os.path.join(baseFolder, baseFile))
df_promos_B = pd.read_pickle(dataFile)
df_promos = pd.concat([df_promos_A,df_promos_B])

idx_valid_promos = df_promos['wk1_sales_all_stores'] > 50
df_promos = df_promos[idx_valid_promos].copy()

df_promos['promo_id'] = df_promos['offer_number'] + '_' + df_promos['base_product_number_std']
df_promos['product_sub_group_code'] = df_promos['product_sub_group_code'].str.strip()




psgc_coffee_pods = ['G61EN', 'G61ED', 'G61DF']
idx_coffee_pods = df_promos.product_sub_group_code.str.contains('|'.join(psgc_coffee_pods))

# remove nescafe and kenko
this_keyword = 'nescafe|kenco|BEANIES|STARBUCKS COFFEE|TAYLORS COFFEE|PRONTISSIMO|TAYLORS GROUND|TAYLORS OF HARG|NSCAFE|KNCO|TAYLORS CFFE'
idx_to_remove = df_promos.offer_description.str.contains(this_keyword, regex=True, flags=re.IGNORECASE)
idx_to_remove.sum()
idx_coffee_pods[idx_to_remove] = False

df_coffee_pods = df_promos[idx_coffee_pods].copy()

fcn_dates = lambda d: dt.datetime.strptime(d, '%Y-%m-%d')
for iVar in ['offer_start_date', 'offer_end_date']:
  df_coffee_pods[iVar] = df_coffee_pods[iVar].apply(fcn_dates)
  df_coffee_pods[iVar] = df_coffee_pods[iVar].apply(lambda ts: ts.date())


# limit to area price code 2
idx_area_pc = df_coffee_pods.area_price_code==2
df_sku_meta_store = df_coffee_pods[idx_area_pc].copy()

df_sku_meta_store.sort_values('offer_start_date', inplace=True)

idx_valid_dates_A = df_sku_meta_store['offer_start_date'] <= pd.Timestamp('2016-09-21')
idx_valid_dates_B = df_sku_meta_store['offer_start_date'] >= pd.Timestamp('2014-01-01')
idx_valid_dates = idx_valid_dates_A & idx_valid_dates_B

df_sku_meta_store = df_sku_meta_store[idx_valid_dates].copy()


vars_to_keep = ['offer_number',
'base_product_number_std',
'area_price_code',
'feature_space',
'offer_start_date',
'offer_end_date',
'promoLength',
'offer_type',
'nfor',
'offer_description',
'baseline_price',
'offer_unit_price',
'total_stores',
'total_store_revenue',
'wk1_sales_all_stores',
'wk1_sales_all_stores_per_mill',
'avgRevenue']

fu.to_random_excel_file(df_sku_meta_store[vars_to_keep])


# exclude previous offers on lavazza pods
lavazza_target = '76496356'

idx_target = df_sku_meta_store.base_product_number_std == lavazza_target
idx_valid_dates_A = df_sku_meta_store['offer_start_date'] < pd.Timestamp('2016-09-21')
idx_existing_lavazza = (idx_target & idx_valid_dates_A)

df_sku_meta_store = df_sku_meta_store[~idx_existing_lavazza].copy()

output_path = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/ThesisPresentationVIVA/datasets'
output_file = os.path.join(output_path, 'm2_lavazza_coffee_pods.pickle')
fu.toPickleFile(df_sku_meta_store, output_file)
