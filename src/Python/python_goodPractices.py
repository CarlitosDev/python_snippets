# PEP8
autopep8 -i --max-line-length 120 PATHTOFILE



# Function annotations
# (This PEP introduces a syntax for adding arbitrary metadata annotations to Python functions [1].)
def kinetic_energy(m:'in KG', v:'in M/S')->'Joules': 
    return 1/2*m*v**2

kinetic_energy.__annotations__
#returns:
#{'m': 'in KG', 'v': 'in M/S', 'return': 'Joules'}

# This one is pretty cool
'{:,} {}'.format(kinetic_energy(20,3000), kinetic_energy.__annotations__['return'])
'90,000,000.0 Joules'


# f-strings in Python
v1 = 'variable'
v2 = 3284
thisStr = f'''This is a {v1} with a value {v2}
that could rufrom collections import dequen {v2} times'''
print(thisStr)

# Format with f-strings
a = 10.1234
f'{a:.2f}'
'10.12'




# If else in one line
data = [1,2,4,5]
data_missing = None if data is None else np.isnan(data)


# Asserts in one line
assert feature_dependence in feature_dependence_codes, "Invalid feature_dependence option!"