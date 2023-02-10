'''
	presentation_example_method_2.py
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
import datetime as dt

#import utils.aws_data_utils as awsu
#import utils.nlp_utils as nlpu
#import utils.image_utils as imgu



experiment_label = 'grocery'
baseFolder = '~/Google Drive/order/Machine Learning Part/data/data From the UK (by folder)'
#baseFile   = 'dataHousehold/dataBTID_Household_NovToDec2k16.pickle'
#baseFile   = 'dataGrocery1/dataBTID_Grocery 1_Xmas2k16.pickle'


baseFile   ='dataProvisions/dataBTID_Provisions_Xmas2k16.pickle'
baseFile   = 'dataGrocery2/dataBTID_Grocery 2_Xmas2k16.pickle'

baseFile   = 'dataProvisions/dataBTID_Provisions_NovToDec2k16.pickle'
baseFile   = 'dataGrocery1/dataBTID_Grocery 1_NovToDec2k16.pickle'

dataFile = os.path.expanduser(os.path.join(baseFolder, baseFile))
df_promos = pd.read_pickle(dataFile)


# # let's combine the datasets??
# baseFile   = 'dataGrocery1/dataBTID_Grocery 1_fromSeptToNov2k16.pickle'
# dataFile = os.path.expanduser(os.path.join(baseFolder, baseFile))
# df_promos_A = pd.read_pickle(dataFile)

# baseFile   = 'dataGrocery1/dataBTID_Grocery 1_NovToDec2k16.pickle'
# dataFile = os.path.expanduser(os.path.join(baseFolder, baseFile))
# df_promos_B = pd.read_pickle(dataFile)
# df_promos = pd.concat([df_promos_A,df_promos_B])

idx_valid_promos = df_promos['wk1_sales_all_stores'] > 50
df_promos = df_promos[idx_valid_promos].copy()

df_promos['promo_id'] = df_promos['offer_number'] + '_' + df_promos['base_product_number_std']
df_promos['product_sub_group_code'] = df_promos['product_sub_group_code'].str.strip()


'''
a = df_promos['base_product_number_std'].value_counts().to_frame()
a.iloc[300]

72927084




'''

fcn_dates = lambda d: dt.datetime.strptime(d, '%Y-%m-%d')

this_sku = '50475945'
idx_SKU = df_promos['base_product_number_std'] == this_sku
df_sku = df_promos[idx_SKU].copy()


iVar = 'offer_start_date'
df_sku[iVar] = df_sku[iVar].apply(fcn_dates)
df_sku[iVar] = df_sku[iVar].apply(lambda ts: ts.date())

iVar = 'offer_end_date'
df_sku[iVar] = df_sku[iVar].apply(fcn_dates)
df_sku[iVar] = df_sku[iVar].apply(lambda ts: ts.date())

df_sku.iloc[32]

# aggregate to NTNL
df_sku.area_price_code.value_counts()
idx_area_pc = df_sku.area_price_code==2
df_sku_meta_store = df_sku[idx_area_pc].copy()
df_sku_meta_store.sort_values('offer_start_date', inplace=True)




output_path = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/ThesisPresentationVIVA/datasets'
output_file = os.path.join(output_path, 'lavazza.pickle')
fu.toPickleFile(df_sku_meta_store, output_file)


numericalVars = ['total_stores', 'for_price', 'baseline_price',
'offer_unit_price', 'total_store_revenue']
categoricalVars = ['meta_store_id', 'offer_type', 'nfor']


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



import matplotlib.pyplot as plt
fig_h = 10
fig_w = 18
fig, ax = plt.subplots(figsize=(fig_w, fig_h/1.5))
def_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']


x_axis = df_one_sku.calendar_date
x_axis.min()
x_axis.max()

ax.plot(x_axis, df_one_sku[sales_var], label=f'Sales {sku_A} (cannibal)',
        color=def_colours[0], linewidth=2.5, alpha=0.75)

# twin object for two different y-axis on the sample plot
ax2 = ax.twinx()
ax2.plot(x_axis, df_one_sku[price_var], label=f'Price {sku_A} (cannibal)',
        color=def_colours[1], linewidth=1.5, alpha=0.75)
ax2.set_ylabel('Price', fontsize=14)


plt.legend()
plt.xlabel('dates')
ax.set_ylabel('Store sales')
plt.grid(True)
plt.title(f'Sales of {sku_A}')
plt.show()