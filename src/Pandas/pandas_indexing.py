# Partial indexing
# let's read this file, set these 3 variables as Indexes
df_test = pd.read_csv(
	"/Users/carlos.aguilar/Documents/Kaggle Competition/Grocery Sales Forecasting/test.csv", usecols=[0, 1, 2, 3, 4],
	dtype={'onpromotion': bool},
	parse_dates=["date"]  # , date_parser=parser
).set_index(['store_nbr', 'item_nbr', 'date'])
# get the index values so we can perform search, etc operations
storeValues = df_test.index.get_level_values(0)
itemValues = df_test.index.get_level_values(1)
dateValues = df_test.index.get_level_values(2)
# get the data for this particular item
idxItem = itemValues == 310671;
df_test = df_test[idxItem]

# Access by the index value
df.loc['1000652']



# Stack and unstack multindex files
index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'), ('two', 'a'), ('two', 'b'), ('three', 'b')])
s = pd.Series(np.arange(1.0, 6.0), index=index)
s.head()
# To query the multiindexusing the index values
this_query = ('one', 'b')
s.loc[this_query]
# To avoid errors, check that the key exists before accessing it
this_query = ('MANOLO', 'b')
if this_query in s.index:
  s.loc[this_query]
# have a look to the 'labels' index
s.keys()
# Unstack by level 'i' is kind of 'the index at i will turn into columns'
# In the example there are only TWO levels
# unstack the by (1,b)
s.unstack(level=-1)
# unstack the by (one, two, three)
s.unstack(level=0)
# get the original back
df = s.unstack(level=0)
df.unstack()



# Can I reindex when it is a multiindex DF?
import numpy as np
index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'), ('two', 'a'), ('two', 'b'), ('three', 'b')])
df = pd.Series(np.arange(1.0, 6.0), index=index).to_frame('A')

('one', 'b') in df.index

new_index = ['a','b','c','d']
df2 = df.reindex(new_index, level=1, axis='index',fill_value=0)
print(df2)

df2 = df.unstack(level=-1).fillna(0)
print(df2)
df2 = df2.reindex(new_index, level=1, axis='columns',fill_value=0)
print(df2)



# Reindexing
# Reindex arranges the data according to the new index, introducing missing values if any index values are not present
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)



# Logical not of an index:
idxTest   = oneBPNS['promoToFrc']>0;
idxTrain  = np.logical_not(idxTest);




# create a DF with the indexes "store_nbr", "item_nbr", "date" and
# unstack the dates so the columns are the combination "onpromotion", "date"
promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
# Then, remove the "onpromotion" from the columns to just have the "dates"
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)

# Same idea applied to the test set
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
# Not sure...but I think by doing this i we kind of left join on the training set
# and set the products not present in the test set to false
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)

promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

# Set the date as columns and if the sales are not found, set them to zero
df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0) 
df_2017.columns = df_2017.columns.get_level_values(1)

# Reindex the items as in the training set
items = items.reindex(df_2017.index.get_level_values(1))


# string find / strfind
idx_production = df.map_type.str.contains('production')