# Access fields with get - it does not throw an error if the field doesn't exist


# Merge two dictionaries
x = {'a':1, 'b': 2}
y = {'b':10, 'c': 11}
#This overwrites!
z = {**x, **y}
# And this is ugly, but works
temp_dict = {}
for iTopic in [*x.keys()]:
  temp_dict.setdefault(iTopic, []).append(x.get(iTopic,[]))
for iTopic in [*y.keys()]:
  temp_dict.setdefault(iTopic, []).append(y.get(iTopic,[]))


# Get all the keys but one (for more complex stuff use OrderedDict from collections)
json_contents_ordered = {key:value for key, value in json_contents.items() if key != 'descriptions'}





# Example of growing dictionary woth <> lenght lists
event_family = ['r1', 'r2', 'r1']
all_records = [{'a':'pestes', 'b':'leches'}, {'a':'sumos', 'b':'baterias'}, {'b': 'caravaggio', 'a':'comisario'}]
temp_dict = {}
for idx, iTopic in enumerate(event_family):
  temp_dict.setdefault(iTopic, []).append(all_records[idx])

# Pandas is such a good boy that sorts the order out: 
#pd.DataFrame.from_dict(temp_dict['r1'])



person = {'name': 'Phill', 'age': 22}
print('Name: ', person.get('name'))
print('Age: ', person.get('salary', 0))


# Classic key to index Matlab map
kinesis_topics = ['kus-kus', 'tzaciky']
key_to_index = dict(zip(kinesis_topics, range(len(kinesis_topics))))


# Dictionary appending lists with different lengths
temp_dict = {}
temp_dict.setdefault(unique_topics[0], []).append('apple')
temp_dict.setdefault(unique_topics[1], []).append('boots')
temp_dict.setdefault(unique_topics[0], []).append('cat')


# Example of growing dictionary woth <> lenght lists
event_family = ['pablo', 'pablo', 'pedro']
all_records = [{'s':'pestes'}, {'d':'sumos'}, {'c': 'caravaggio'}]
temp_dict = {}
for idx, iTopic in enumerate(event_family):
  temp_dict.setdefault(iTopic, []).append(all_records[idx])



# Dict comprehension
{str(i):i for i in [1,2,3,4,5]}



Create a dictionary from two lists:
	a = ['a', 'b']
	b = ['c', 'd']
	c = dict(zip(a,b))
	c['a']

Get a list with the values of a dict:
impression_1 = {'transactionId': 2137, 'CookieID':10, 'type':'impression'}
listOfValues = [value for key, value in impression_1.items()]
 
Get the keys of a dictionary:
	listOfKeys = list(currentImpression.keys())
	listOfKeys = [*currentImpression.keys()]



# helper to get the keys as a list
get_dict_keys = lambda d: [*d.keys()]
# helper to set the keys to lowercase
dict_keys_to_lower = lambda d: dict((k.lower(), v) for k,v in d.items())
# helper to set the keys to lowercase and add prefix
dict_keys_to_prefix_lower = lambda d, prefix: dict((prefix + '_' + k.lower(), v) for k,v in d.items())
    

# Update a dictionary - it will OVERWRITE the field '2'
d  = {1: "one", 2: "three"}
d1 = {2: "two"}
d.update(d1)

# or adding more values
d.update({'f': 432, 'j': "fgadsg"})


# delete, remove
d.pop('j', None)


# Dictionary with empty values
vars_to_keep = ['userId', 'type', 'event', 'originalTimestamp', 'traits']
temp_dict = dict.fromkeys(vars_to_keep,[])