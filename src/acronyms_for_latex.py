'''
	Run it as
	python3 acronyms_for_latex.py | pbcopy
'''

'''
this_acronym = ''
this_meaning = ''
all_terms.update({this_acronym: this_meaning})

this_acronym = ''
this_meaning = ''
all_terms.update({this_acronym: this_meaning})
'''

def acronym_to_list(this_acronym, this_meaning):
  mn = this_meaning.split()
  print('\\textbf{' + this_acronym + '} & ', end='')
  for idx, this_char in enumerate(list(this_acronym)):
    print('\\textbf{' + this_char + '}' + mn[idx][1::] + ' ', sep=' ', end='')
  print('\\\\')



# Place them all here so I can sort them

all_terms = {}

this_acronym = 'CPG'
this_meaning = 'consumer packaged goods'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'FMCG'
this_meaning = 'fast Moving Consumer Goods'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'BOGOF'
this_meaning = 'buy one get one free'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'SCM'
this_meaning = 'Supply Chain Management'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'SC'
this_meaning = 'Supply Chain'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'CPFR'
this_meaning = 'Collaborative Planning Forecasting and Replenishment'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'SKU'
this_meaning = 'Storage keeping Unit'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'SCOR'
this_meaning = 'Supply Chain Operations Reference'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'PDF'
this_meaning = 'Probability density function'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'MDF'
this_meaning = 'mass density function'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'OHE'
this_meaning = 'one hot encoding'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'CART'
this_meaning = 'Classification And Regression Tree'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'MSE'
this_meaning = 'mean squared error'
all_terms.update({this_acronym: this_meaning})


this_acronym = 'GBDT'
this_meaning = 'gradient boosted decision trees'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'NNLS'
this_meaning = 'Non negative least squares'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'ML'
this_meaning = 'Machine Learning'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'DL'
this_meaning = 'Deep Learning'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'MOFC'
this_meaning = 'Makridakis Open Forecasting Center'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'NDA'
this_meaning = 'Non disclosure agreement'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'EDA'
this_meaning = 'Exploratory Data Analysis'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'IGD'
this_meaning = 'Institute Grocery Distribution'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'DRY'
this_meaning = 'Don\'t Repeat Yourself'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'KNN'
this_meaning = 'k nearest neighbours'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'ARIMA'
this_meaning = 'Autoregressive and Integrated Moving Average'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'SVM'
this_meaning = 'support vector machines'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'CE'
this_meaning = 'contrastive explanations'
all_terms.update({this_acronym: this_meaning})

this_acronym = 'POS'
this_meaning = 'Point of sales'
all_terms.update({this_acronym: this_meaning})

#this_acronym = 'NGBoost'
this_acronym = 'NGB'
this_meaning = 'Natural Gradient boosting'
all_terms.update({this_acronym: this_meaning})

#this_acronym = 'XGBoost'
this_acronym = 'XGB'
this_meaning = 'Extreme gradient boosting'
all_terms.update({this_acronym: this_meaning})

#this_acronym = 'LightGBM'
this_acronym = 'LGB'
this_meaning = 'Light gradient boosting'
all_terms.update({this_acronym: this_meaning})

# this_acronym = 'CatBoost'
this_acronym = 'CB'
this_meaning = 'Categorial boosting'
all_terms.update({this_acronym: this_meaning})

import collections
od = collections.OrderedDict(sorted(all_terms.items()))
for k,v in od.items():
  try:
    acronym_to_list(k, v)
  except:
    print(f'Error in {k}')



'''
import stringcase
a = 'StudentLevelTestProgress_id'
stringcase.snakecase(a)
b = 'CourseAlterKey'
stringcase.snakecase(b)

stringcase.capitalcase('atun de_a')
#'Atun de_a'

stringcase.snakecase(a)
#'student_level_test_progress_id'

stringcase.pascalcase(a)
#'StudentLevelTestProgressId'

stringcase.sentencecase(a)
#'Student level test progress id'

'''