# unique values and indexes of a list
# (below an approach to shrink another list l2 based on unique l1)
l1 = ['asno', 'pez', 'raton', 'asno']
l2 = ['asno1', 'pez1', 'raton1', 'asno2']
unique_l1 = sorted(set(l1))
indexes = [l1.index(x) for x in unique_l1]
l2_unique = [l2[idx] for idx in indexes]

# Sort lists and get their index
f1_weighted = []
f1_weighted.append(0.312)
f1_weighted.append(0.912)
f1_weighted.append(0.812)
# this
#idx_f1_weighted = sorted(range(len(f1_weighted)), key=lambda k: f1_weighted[k])[::-1][0]
# or this
idx_f1_weighted = sorted(range(len(f1_weighted)), key=lambda k: f1_weighted[k])[-1]
f1_weighted[idx_f1_weighted]


# Find the closest and smaller number to 287 in a list using map and filter
number = 287
numbers = [11, 71, 150, 285, 313, 373, 474]
fcn_closest = lambda number, numbers: max(filter(lambda x: x <= number, numbers))
fcn_closest(number, numbers)


# Feeling Pythonic
selected_departments = [ 'LIQUOR,WINE,BEER', 'GROCERY_II','GROCERY_I',
'MEATS', 'POULTRY', 'PREPARED_FOODS', 'PRODUCE', 'BREAD_BAKERY']
','.join(sorted([*map(lambda s: s.lower().replace('_', ' '), selected_departments)]))


# From list to string (SQL-clauses)
list_of_EFID = ['76a58ee3-ce5b-41e0-99a6-b17b99cb4cf9', 
'fc1e0a3d-96b3-4814-8a03-d2b5d786a059', '98abe356-1277-4c3b-a9b0-936c1284a2e8',
'f6ec44e7-93b8-47b6-aafb-c5cb29f0fcab', 'f839f407-286b-4bf1-86d7-edbfe72af785']
fcn_format = lambda this_efid: '\'' + this_efid + '\''
efid_filter_clause = '(' + ','.join(map(fcn_format, list_of_EFID)) + ')'



# List comprehension with two for loops

# Turn this thing into a list comprehension
d1 = {'features_batches': ['case_1', 'value1']}
d2 = {'features_batches': ['case_2', 'value2']}
forecasters_r = [d1, d2]

# Turn this thing into a list comprehension
all_feats_r= []
for forecaster in forecasters_r:
    for features in forecaster['features_batches']:
        all_feats_r.append(features)
# Voila. Put the final variable in the first place and copy and paste the for loops without altering the natural order
all_feats_lc = [features for forecaster in forecasters_r for features in forecaster['features_batches']]


# Indices of a sorted list
>>> s = [2, 3, 1, 4, 5]
>>> sorted(range(len(s)), key=lambda k: s[k])

# Zip creates an iterator. Useful for parallel executions
numList = [0, 1, 2]
engList = ['zero', 'one', 'two']
espList = ['cero', 'uno', 'dos']
Eng = list(zip(engList, espList, numList))
print(Eng)
# Populate the contents
for num, eng, esp in zip(numList, engList, espList):
    print(f'{num} is {eng} in English and {esp} in Spanish.')
# sort
Eng.sort()
# access elements
a, b, c = zip(*Eng)



# Add elements to a list
a = ['a', 'b']
c = a + ['c']

# Extend flattens a list - unline append
a = [['dsad', 'ifigfd'], 'ds', ['asdddd', 'dddasd']]
allInOne = []
for thisOne in a:
    allInOne.extend(thisOne)


# Use map to reuse functions passing list as arguments
add_fcn = lambda x, y: x+y

a = 6
b = 7
add_fcn(a,b)

# Get away with list comprehension
c = [7,4,2,7,8]
[add_fcn(a,x) for x in c]


# Complex lists
valid_group_names = [i_groupNames for i_groupNames in groupNames if dfGrouped.get_group(i_groupNames).shape[0]>1]



# Cast list of integers to one string (separated by commas)
ls = [i for i in range(10)]
ls_str = ','.join(map(str, ls))


# Pythonic tricks when iterating lists
a = [1,2,3,1,3,2,1,1]
b = [4 if x==1 else x+12 for x in a]
# if it's just one condition the clause goes at the end
valid_fields = [ iField + '_' + iVal for iVal in reco_config['facescan_' + iField + '_vals'] if iVal!='None']



# add list of strings to a string
response_type = 'calamar'
missing_fields = ['a', 'pez', 'raton']
response_type = '\n'.join(missing_fields) + '\n'

# Compare the elements of two lists regarless of the order of the elements
a = ['b','g']
b = ['g', 'b', 'f']
set(a) == set(b)

# From list of lists to a simple list:
from itertools import chain	
listOfTickers = [['AAPL', 'MSFT', 'TSCO.L', 'SBRY', 'BP', 'REP.MC']]
categories = list(chain.from_iterable(listOfTickers));



# Merge two lists (<> sizes)
a = ['d', 'g', 's']
b = ['b', 'c']
import itertools
flattened = list(itertools.chain.from_iterable([a,b]))


# Unnest a list (tempList):
fieldValuesList = [item for sublist in tempList for item in sublist]

# Remove an item (str) from a list (of str):
a = ['paco', 'manolo']
a.remove('paco')
print(a)


# Concatenate two lists - This one is really cool (very Pythonic):
a = ['d', 'h', 'g']
b = ['1', '2', '3']
c = [*map(str.__add__, a, b)]


# Index a list: Taking the last N (20) elements
latestFiles = fNames[-20:]




# Random choice of the elements of  a list:
import random
record_names = ['Alice', 'Bob', 'Charlie']
print(random.choice(record_names))

# To select more than 1 elemtn
# mind choices allows repetition
record_names = ['Alice', 'Bob', 'Charlie']
sampling = random.choices(record_names, k=2)


# this one does not replacement but it overwrites the list...
shuffled_list = record_names.copy()
random.shuffle(shuffled_list)


# this one does not replacement either
possible_attempts = random.sample(record_names, k=len(record_names))
print(possible_attempts)


# Random number of random elements from a list
# (random sampling without replacement)
shelf_capacity =[12, 4, 8, 6, 25]
num_shelves = len(shelf_capacity)
random.sample(shelf_capacity, 1+np.random.randint(num_shelves-1))



# Merge two lists:
newList = [x for t in zip(listA,listB) for x in t]



# Find values in a list:
# try out the value 'covergirl' in' operator
covergirlValues = [s for s in fNames if re.search('covergirl', s)];# with regEx
covergirlValues = [s for s in fNames if 'covergirl' in s];# no regex

# [Check out python_maps_reduce.py to see tricks with maps and lists]


# Lowercase all elements in list
varsToExcludeFromModel = ['SourceInsertDate', 'SourceUpdateDate', 'maxClassRoomAuditDate', 
'maxClassRoomPerformanceDate','maxClassRoomStartDate',	'minClassRoomStartDate', 
'DimOnlineTeacherKey', 'TeacherCenterName' , 	'TeacherAlterkey', 'TeacherCenterParentName', 
'TeacherCountry', 'TeacherKey', 'TeacherStatus', 'accessedDate', 'lastActivityDate']
srcFiles = list(map(lambda x: str.lower(x), varsToExcludeFromModel))


# Sort a list alphabetically:
presentVars = sorted(list(set(colNames).intersection(set(varsToRead))))


# Replace in list of strings:
srcFiles = list(map(lambda x: str.replace(x, tpParquet , ''), srcFiles))
ssrcFiles = [x.replace(tempFolder , '') for x in srcFiles]



# Print the components of a list
new_columns = ['uha', 'a tope']
print('Added columns {}'.format('\n'.join(map(str, new_columns))))



# Get the position of words in a bigger list
varNames = ['ad_interaction', 'tp_name', 'device_name']
colNames = ['month','day', 'hour', 'tp_name', 'device_name', 'visit_number', 'visit_page_number'];
[colNames.index(intersected) for intersected in [*set(varNames).intersection(set(colNames))]];



# Remove an element from a list (by element, by position use pop)
# internal_catalogues is a dictionary of DFs
this_catalogue = internal_catalogues[str.lower(this_product)]
cols = this_catalogue.columns
idx = cols.str.contains('product_') | cols.str.contains('reco_') | cols.str.contains('shade_')
col_names = cols[idx].tolist()
col_names.remove('product_key')



# Compare the contents of two lists
def compare_lists(list_A, list_B):
  set_A = set(list_A)
  set_B = set(list_B)
  print(f'list_A contains {len(set_A)} unique elements')
  print(f'list_B contains {len(set_B)} unique elements')
  set_intersection = set_A.intersection(set_B)
  interesected = list(set_intersection)
  only_A = [*set_A - set_intersection]
  only_B = [*set_B - set_intersection]
  print(f'the intersection contains {len(interesected)} elements')
  return {'interesected':interesected, 'only_A': only_A, 'only_B': only_B}



# Count elements in list
l = ["a","b","b"]
from collections import Counter
Counter(l)


# tricks to use if with dictionaries
ordered_items = [this_sentence['#text'] 
  for this_sentence in ActivityContentDict['Sequence']['ImgUrl'] 
    if this_sentence.get('#text', False)]