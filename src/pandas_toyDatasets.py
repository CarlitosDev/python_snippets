# Fake sales
num_samples = 500

x_3_values = np.random.randint(0,200, size=(num_samples))
noisy_sales = 10 * np.random.random(size=(num_samples)) + x_3_values

df = pd.DataFrame({
    "x1" : np.random.randint(0, 100, size=(num_samples)),
    "x2" : np.random.randint(0, 200, size=(num_samples)),
    "x3" : x_3_values,
    "sales": noisy_sales
})








df = pd.DataFrame([{'A': 'foo', 'B': 'green', 'C': 11}, \
				{'A':'bar', 'B':'blue', 'C': 20}, \
				{'A':'foo', 'B':'blue', 'C': 20}])



# Note expectedTimes is deliberately left as strings
countryList = ['Japan','Morocco','Russia','Saudi Arabia','Taiwan']
expectedTimes = ['2018-02-09 20:30:00+09:00',
'2018-02-09 12:30:00+01:00','2018-02-09 13:30:00+02:00',
'2018-02-09 14:30:00+03:00','2018-02-09 19:30:00+08:00']
df = pd.DataFrame(
    {'start_date': pd.Timestamp('09/02/2018  06:30:00'),
    'country': pd.Categorical(countryList),
    'local_start_date': expectedTimes
    })



df2 = pd.DataFrame({ 'A' : 1.,
....:                      'B' : pd.Timestamp('20130102'),
....:                      'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
....:                      'D' : np.array([3] * 4,dtype='int32'),
....:                      'E' : pd.Categorical(["test","train","test","train"]),
....:                      'F' : 'foo' })


d = [{'col1': 12, 'col2': 17}, {'col1': 2, 'col2': 1}]
df = pd.DataFrame(data=d) 




# Two categorical variables
import pandas as pd
import random

# create data
reco_names = ['alpha', 'beta', 'gamma', 'delta', 
'epsilon', 'zeta', 'eta', 'theta', 'iota']

cuban_animals = ['Solenodon', 'Trogon', 'Eleuth', 'Hutia', 
'Ground Iguana', 'Tody', 'Land Crab']

num_records = 100;

all_recos  = []
all_animals = []
for idx in range(0, num_records):
    all_recos.append(random.choice(reco_names))
    all_animals.append(random.choice(cuban_animals))

df_categorical = pd.DataFrame({'reco_names': all_recos, 'cuban_animals': all_animals})
df_categorical.head()
# get the OHE version
df_OHE = pd.get_dummies(df_categorical)






dataframe = pd.DataFrame({
    "date_time": [ generate_random_date_in_last_year() for _ in range(10)],
    "animal": ['zebra','zebra','zebra','zebra','lion','lion','lion','lion','rhino','rhino',],
    "category": ['stripy'] * 4 + ['dangerous'] * 6,
    "name": ['Walter','Edmund','Gyles','John','Bartholomew','Frederyk','Raulf','Symond','Carlos','Arthur'],
    "weight": [80 +40 * r.random() for _  in range(10)],
    "favourite_integer" : [ r.randint(0,100) for _ in range(10)]
})


dataframe = pd.DataFrame({
    "weight": [80 +40 * np.random.random() for _  in range(10)],
    "favourite_integer" : [ np.random.randint(0,100) for _ in range(10)]
})



Use combinations to create a dataframe:
skuDateDuple = list(itertools.product(pd.date_range(minDate, maxDate, freq='D'), ['das', 'gaa']))
df = pd.DataFrame(skuDateDuple, columns=['date', 'productid'])
df['sales'] = 0




import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import date, timedelta

df = pd.DataFrame([{'unit_sales' :1, 'date' : pd.Timestamp('20130103'), 'store_nbr' : 2, 'onpromo': True }, 
                  {'unit_sales' : 7, 'date' : pd.Timestamp('20130106'), 'store_nbr' : 2, 'onpromo': False }, 
                  {'unit_sales' : 2, 'date' : pd.Timestamp('20130102'), 'store_nbr' : 1, 'onpromo': True  }])




# From Sebastian Raschka
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                ['red', 'L', 13.5, 'class2'],
                ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']