'''
source ~/.bash_profile && python3 -m pip install icecream

https://towardsdatascience.com/stop-using-print-to-debug-in-python-use-icecream-instead-79e17b963fcc

debugging_with_icecream.py


source ~/.bash_profile && python3 
'''

import icecream as ic

def plus_five(num):
    return num + 5


#ic.configureOutput(includeContext=True)
#AttributeError: module 'icecream' has no attribute 'configureOutput'?

# TypeError: 'module' object is not callable
ic(plus_five(4))