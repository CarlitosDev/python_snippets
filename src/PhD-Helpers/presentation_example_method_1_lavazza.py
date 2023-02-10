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




output_path = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/ThesisPresentationVIVA/datasets'
output_file = os.path.join(output_path, 'lavazza.pickle')
df_sku_meta_store = fu.readPickleFile(output_file)





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