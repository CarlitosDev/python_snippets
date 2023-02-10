'''

	Distances/Similarities between strings 


	https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance

'''



from fuzzywuzzy import fuzz
import numpy as np
import Levenshtein as lev


prn_result = lambda metric_name, value: print(f'{metric_name}: {value}')

def get_metrics(str1, str2):
	string1, string2 = str1.lower(), str2.lower()
	prn_result('Levenshtein dist', lev.distance(string1, string2))
	prn_result('fuzz.partial_ratio', fuzz.partial_ratio(string1, string2))
	prn_result('fuzz.ratio', fuzz.ratio(string1, string2))
	prn_result('Jaro dist', lev.jaro(string1, string2))
	prn_result('Jaro-Winkler dist', lev.jaro_winkler(string1, string2))


#%%
# In this example I want a high score but not too high
str1 = "GERMAN101"
str2 = "GERMAN102"
get_metrics(str1, str2)

#%%
# In this example I want a high score but not too high
str1 = "GERMAN101"
str2 = "GERMAN201"
get_metrics(str1, str2)

# In this example I want to capture that there are some similarities, as the level is the same
str1 = "GERMAN101"
str2 = "FINNISH101"
get_metrics(str1, str2)


# In this example I want to get a low score as the lessons are different
str1 = "GERMAN101"
str2 = "CHINESE109"
get_metrics(str1, str2)


# In this example I want to get a larger score than in the Chinese case
str1 = "GERMAN101"
str2 = "GERMAN-GAMMAR102"
get_metrics(str1, str2)



# Nothing in common
str1 = "ABCD"
str2 = "EFG"
get_metrics(str1, str2)

# One character in common
str1 = "ABCD"
str2 = "EFGA"
get_metrics(str1, str2)

# One character in common matching position
str1 = "ABCD"
str2 = "AEFG"
get_metrics(str1, str2)



# this one is a bit mind-blowing
spellings = ['aguilar', 'agilar', 'aguila', 'aguilarr']
lev.median(spellings)





# Look for partial matches in column names
get_lev_distance = lambda str_a, str_b: lev.ratio(str_a.lower(), str_b.lower())
score = df_all_tables_info.var_name.apply(lambda s: get_lev_distance(field_name,s))
df_all_tables_info[score>0.7].shape