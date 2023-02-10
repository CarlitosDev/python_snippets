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
#baseFile   = 'dataHousehold/dataBTID_Household_NovToDec2k16.pickle'
#baseFile   = 'dataGrocery1/dataBTID_Grocery 1_Xmas2k16.pickle'


baseFile   ='dataProvisions/dataBTID_Provisions_Xmas2k16.pickle'
baseFile   = 'dataGrocery2/dataBTID_Grocery 2_Xmas2k16.pickle'
baseFile   = 'dataGrocery1/dataBTID_Grocery 1_Xmas2k16.pickle'
#df_promos = pd.read_pickle(dataFile)


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


'''
df_promos['base_product_number_std'].value_counts()

62609350    894
62609056    882
70586933    784



'''

'''
datevars_to_extract = ['offer_end_date', 'offer_start_date']
for iVar in datevars_to_extract:
	df_promos[iVar] = pd.to_datetime(df_promos[iVar].apply(lambda x: str(x[0])), format='%Y-%m-%d')

import datetime
df_promos[iVar].apply(lambda d: datetime.strptime(d).date())

#pd.to_datetime(df_promos[iVar], format='%Y-%m-%d')

df_promos[iVar] = 
pd.to_datetime(df_promos[iVar].apply(lambda d: d[0]), format='%Y-%m-%d')

pd.to_datetime(df_promos[iVar].astype(str))
'''



this_keyword = 'coffee|lavazza|taylors|beans'
idx_products = df_promos.offer_description.str.contains(this_keyword, regex=True, flags=re.IGNORECASE)
if idx_products.any():
	print(f'Voila\n\nFound {idx_products.sum()} products')
	df_promos[idx_products].head()


this_keyword = 'nescafe|kenco'
idx_to_remove = df_promos.offer_description.str.contains(this_keyword, regex=True, flags=re.IGNORECASE)
idx_products[idx_to_remove] = False

df_selected_promos = df_promos[idx_products].copy()

df_selected_promos.iloc[32]

df_a = df_selected_promos.offer_description.value_counts().to_frame()
df_a['description'] = df_a.index
fu.to_random_excel_file(df_a, writeIndex=True)

this_keyword = 'lavazza'
idx_lavazza = df_selected_promos.offer_description.str.contains(this_keyword, regex=True, flags=re.IGNORECASE)
idx_lavazza.sum()

df_promos_lavazza = df_selected_promos[idx_lavazza]
df_promos_lavazza.iloc[2]
fu.to_random_excel_file(df_promos_lavazza)


this_keyword = 'TAYLORS COFFEE BEANS'
idx_taylor_beans = df_selected_promos.offer_description.str.contains(this_keyword, regex=True, flags=re.IGNORECASE)
idx_taylor_beans.sum()
df_taylor_beans = df_selected_promos[idx_taylor_beans]
fu.to_random_excel_file(df_taylor_beans)


this_keyword = 'lavazza|pods'
idx_lavazza_pods = df_selected_promos.offer_description.str.contains(this_keyword, regex=True, flags=re.IGNORECASE)
df_lavazza_pods = df_selected_promos[idx_lavazza_pods]
fu.to_random_excel_file(df_lavazza_pods)


df_coffee_beans = df_selected_promos[df_selected_promos.product_sub_group_code=='G61EU']
fu.to_random_excel_file(df_coffee_beans)



this_keyword = 'coffee pods'
idx_lavazza_pods = df_promos.offer_description.str.contains(this_keyword, regex=True, flags=re.IGNORECASE)
df_lavazza_pods = df_promos[idx_lavazza_pods]
fu.to_random_excel_file(df_lavazza_pods)

df_lavazza_pods.product_sub_group_code.unique()


this_keyword = 'capsules'
idx_lavazza_pods = df_promos.offer_description.str.contains(this_keyword, regex=True, flags=re.IGNORECASE)
df_lavazza_pods = df_promos[idx_lavazza_pods]
fu.to_random_excel_file(df_lavazza_pods)


psgc_coffee_pods = ['G61EN', 'G61ED', 'G61DF']
idx_coffee_pods = df_promos.product_sub_group_code.str.contains('|'.join(psgc_coffee_pods))
df_promos[idx_coffee_pods]




# Selected promo to forecast
promo_id = 'A32071995_50475945'
idx_promos_to_forecast = df_selected_promos['promo_id'] == promo_id

df_promos_to_forecast = df_selected_promos[idx_promos_to_forecast].copy()
df_promos_to_forecast.reset_index(inplace=True, drop=True)
fu.to_random_excel_file(df_promos_to_forecast)

idx_promo_to_forecast = 0
current_PSGC = df_promos_to_forecast.loc[idx_promo_to_forecast]['product_sub_group_code']


idx_all_SKUs_category = df_promos['product_sub_group_code'] == current_PSGC
df_current_PSGC = df_promos[idx_all_SKUs_category].copy().reset_index(drop=True)


# Select the variables prior to run the algo
numericalVars = ['total_stores', 'for_price','promoCurrencyDiscount',
'baseline_price','total_store_revenue', 'baseline_price', 'offer_price']
categoricalVars = ['meta_store_id', 'offer_type', 'nfor']
responseVariable = 'wk1_sales_all_stores'
all_input_vars = numericalVars + categoricalVars 
all_vars = all_input_vars + [responseVariable, 'promo_id']

df_current_PSGC[responseVariable] = df_current_PSGC[responseVariable].astype(float)

df_current_PSGC_OHE, mapping = pre.oneHotEncoding(df_current_PSGC[all_vars], categoricalVars)

all_input_vars_OHE = numericalVars + [mapping[this_var] for this_var in categoricalVars][0]



# Now I got to the point of training and test
promo_id = 'A32071995_50475945'
idx_promos_to_forecast = df_current_PSGC_OHE['promo_id'] != promo_id
df_train_set = df_current_PSGC_OHE[idx_promos_to_forecast].reset_index(drop=True)
df_test_set  = df_current_PSGC_OHE[~idx_promos_to_forecast].reset_index(drop=True)


from scipy.stats import zscore
df_current_PSGC_OHE_norm = df_current_PSGC_OHE[all_input_vars_OHE].apply(zscore)

X_train_norm = df_current_PSGC_OHE_norm[idx_promos_to_forecast].values
y_train = df_current_PSGC[idx_promos_to_forecast][responseVariable].values

X_train_norm.shape, y_train.shape


number_of_neighbours = 4
num_weak_learners = 5

numRecords, numFeatures = X_train_norm.shape
training_split = 0.25
testSize = round(numRecords*training_split)
numRecords, numFeatures, testSize


# Create a list of `num_weak_learners` bags
X_bags = []
y_bags = []

for k in range(0, num_weak_learners):
    idx_bags = np.random.choice(numRecords, testSize, replace=False)
    X_bags.append(X_train_norm[idx_bags, :])
    y_bags.append(y_train[idx_bags])

from mlToolbox.nextDoorNeighbours import NDN_weak_learner
idx_bag = 0
weak_learner = NDN_weak_learner(X_bags[idx_bag], y_bags[idx_bag], number_of_neighbours)
weak_learner.feat_weight



##
this_X_bag = X_bags[idx_bag]
this_y_bag = y_bags[idx_bag]

currentPromo    = np.zeros(testSize, dtype=bool)
remainingPromos = np.ones(testSize,  dtype=bool)


idx = 26
currentPromo[idx]    = True
remainingPromos[idx] = False


X_pool = this_X_bag[remainingPromos]
y_pool = this_y_bag[remainingPromos]


y_diff_abs = np.abs(np.subtract(y_pool, this_y_bag[currentPromo]))
idxSorted = np.argsort(y_diff_abs)[0:number_of_neighbours]

current_promo = this_y_bag[currentPromo][0]
print(f'Current {current_promo:.2f}')

weights = 1/y_diff_abs[idxSorted]
y_hat   = np.dot(weights, y_pool[idxSorted].T)/np.sum(weights)
print(f'weights {weights}')

print(f'Forecast is {y_hat:.2f}')


# more difficult to read but nicer
                       
weights_ = 1/(y_diff_abs[idxSorted]*np.sum(1/(y_diff_abs[idxSorted])))
print(f'weights_ {weights_}')
y_hat_   = np.dot(weights_, y_pool[idxSorted].T)

print(f'Forecast is {y_hat_:.2f}')

M = []
e = []
# this is 'M'
M.append(np.power(X_pool[idxSorted] - this_X_bag[currentPromo], 2))
# this is 'e'
e.append(np.power(weights_,2))
M = np.concatenate(M, axis=0).copy()
e = np.concatenate(e, axis=0).copy()
M.shape, e.shape

AtA = np.dot(M.T, M)
Aty = np.dot(M.T, e)
from mlToolbox.nnls_solvers import fnnls
fast_res = fnnls(AtA, Aty, epsilon=None, iter_max=None)










X_train, X_val, y_train, y_val = train_test_split(df_train_set[all_input_vars_OHE], df_train_set[responseVariable].values, \
	test_size=0.2, random_state=1)


forecasters = nextDoorForecasterV2.train(X_train, y_train, X_val, y_val, num_forecasters=3, 
_lambda=0.00, _training_split=0.25)


# missing the regulatisation
def calculate_feature_weights(this_X_bag, this_y_bag, training_split, number_of_neighbours):

	numRecords, numFeatures = this_X_bag.shape
	testSize = round(numRecords*training_split)

	currentPromo    = np.zeros(testSize, dtype=bool)
	remainingPromos = np.ones(testSize,  dtype=bool)

	M = []
	e = []

	for idx in range(0, numRecords):

		currentPromo[idx]    = True
		remainingPromos[idx] = False

		X_pool = this_X_bag[remainingPromos]
		y_pool = this_y_bag[remainingPromos]

		y_diff_abs = np.abs(np.subtract(y_pool, this_y_bag[currentPromo]))
		idxSorted = np.argsort(y_diff_abs)[0:number_of_neighbours]

		# more difficult to read but nicer
		weights = 1/(y_diff_abs[idxSorted]*np.sum(1/(y_diff_abs[idxSorted])))

		# current_promo = this_y_bag[currentPromo][0]
		# print(f'Current {current_promo:.2f}')
		# y_hat   = np.dot(weights_, y_pool[idxSorted].T)
		# print(f'Forecast is {y_hat:.2f}')

		# this is 'M'
		M.append(np.power(X_pool[idxSorted] - this_X_bag[currentPromo], 2))
		# this is 'e'
		e.append(np.power(weights,2))

		currentPromo[idx]    = False
		remainingPromos[idx] = True

	M = np.concatenate(M, axis=0).copy()
	e = np.concatenate(e, axis=0).copy()

	AtA = np.dot(M.T, M)
	Aty = np.dot(M.T, e)
	fast_res = fnnls(AtA, Aty, epsilon=None, iter_max=None)

