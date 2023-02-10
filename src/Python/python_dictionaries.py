# Access fields with get - it does not throw an error if the field doesn't exist

# Nested dicts 101
all_assets = {}
meeting_token = 'a'
date = 1
mt_dt = all_assets.get(meeting_token, {date:{}})
mt_dt[date] = {'h':'entry1'}
all_assets.update({meeting_token: mt_dt})

date = 2
mt_dt = all_assets.get(meeting_token, {date:{}})
mt_dt[date] = {'g':'entry2'}
all_assets.update({meeting_token: mt_dt})

date = 3
mt_dt = all_assets.get(meeting_token, {date:{}})
mt_dt[date] = {'k':'entry3'}
all_assets.update({meeting_token: mt_dt})



# I din't know you can do this...
dict(width=0, height=3)



# Use generators to access dicts in one-liners
x = {'a':1, 'b': 2}
items = (f"{k}={v!r}" for k, v in x.items())
_items = "\n".join(items)
print(_items)



# dump dictionary keys to variables
dict_varname = 'lesson_subfolders_info'
for k in lesson_subfolders_info.keys():
  print(f'''{k}={dict_varname}.get(\'{k}\', None)''')
  

# create dictionary with empty values, just the keys
content_info = dict.fromkeys(['a', 'b'])

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



# Order the keys of a dictionary alphabetically
d = {'z':'sumos', 'b':'baterias'}
import collections
od = collections.OrderedDict(sorted(d.items()))

# Order dictionary by values
d = {'z':5, 'b':12, 'c':0}
import collections
import operator
od = collections.OrderedDict(sorted(d.items(), key=operator.itemgetter(1)))
od_descending = collections.OrderedDict(sorted(d.items(), key=operator.itemgetter(1))[::-1])


# Example of growing dictionary with <> lenght lists
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


# Example of growing dictionary with <> lenght lists
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




# pip3 install json_schema
from json_schema import generate
json_to_python = {'string': str, 'integer': int, 'number': float, 'boolean': bool, 'object': dict, 'array': list}



import json

def get_field_types(data_dict):
    field_types = {}
    data_dict = json.loads(json.dumps(data_dict), object_hook=lambda d: {int(k) if k.lstrip('-').isnumeric() else k: v for k, v in d.items()})
    for key, value in data_dict.items():
        field_types[key] = type(value)
    return field_types

data_dict = {"name": "John", "age": 30, "meta":{"a":4}}
print(get_field_types(data_dict)) 
# Output: {'name': <class 'str'>, 'age': <class 'int'>, 'meta': <class 'dict'>}


def get_field_types(data_dict):
    field_types = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            field_types[key] = get_field_types(value)
        else:
            field_types[key] = type(value)
    return field_types

data_dict = {"name": "John", "age": 30, "meta":{"a":4}}
print(get_field_types(data_dict))