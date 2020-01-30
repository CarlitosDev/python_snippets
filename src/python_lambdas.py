Python - lambda: get the definition of an already declared lambda function
p2f = lambda x: float(x.strip('%'))/100

import inspect
inspect.getsource(p2f)

# Lambdas to pick random values from lists
get_one_random_value = lambda values: random.choice(values)

# Generate random values without replacement
np.random.choice(30, 3, replace=False)
get_slots = lambda max_vals: np.random.choice(max_vals, random.randint(0, max_vals), replace=False)


# inline lambda to intersect lists
intersect_lists = lambda list_A,list_B: [*set(list_A).intersection(set(list_B))]


# Set here some anonymous helpers
# helper to get the keys as a list
get_dict_keys = lambda self, d: [*d.keys()]
# helper to set the keys to lowercase
dict_keys_to_lower = lambda self, d: dict((k.lower(), v) for k,v in d.items())
# helper to set the keys to lowercase and add prefix
dict_keys_to_prefix_lower = lambda self, d, prefix: dict((prefix + '_' + k.lower(), v) for k,v in d.items())
# helper to lower the list elements
stringlist_to_lower = lambda self, l: [s.lower() for s in l]
# Merge lists
merge_lists = lambda self, list_a, list_b: list(itertools.chain.from_iterable([list_a, list_b]))


PD convert a percentage string (ie: 10.26%) into a float number:
p2f = lambda x: float(x.strip('%'))/100
colsToConvert  = ['#1 Combination %', '#2 Combination %', '#3 Combination %'];
convertersDict = dict(zip(colsToConvert, [p2f, p2f, p2f]))
df_market_basket_analysis = pd.read_csv(market_basket_FilePath, skiprows=0, converters=convertersDict);


Lambdas and Dataframes:

Apply a function to a DF column:
	p = lambda x: x*2
	df['newVar'] = df['sentiment'].apply(p)

Apply a function to a DF:
	cl1 = lambda x: str.replace(x, "___","_none_none_" )
	cl2 = lambda x: str.replace(x, "__","_none_" )
	df.applymap(cl1)
	df.applymap(cl2)
	# same idea but inline
	df['age_segment'] = df['age'].apply(lambda age: get_age_segment(age))



# >> x must be a pandas series
# Outlier detection based on MAD
mad_score = lambda x: np.abs((0.6745*(x-x.median()))/x.mad())


# Extract parts of a dictionary
this_key = 'role'
df['role_segment'] = df['event_properties'].apply(lambda d: d.get(this_key, []))


# If else condition
print_kinesis = lambda thisRecord: print(thisRecord) if thisRecord != [] else None