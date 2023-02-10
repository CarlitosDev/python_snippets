# Where is Python installed? Bin path to Python
which python3
brew install python3
brew upgrade python3

brew link --overwrite python3


pip3 --version
pip 19.0.3 from /Users/carlos.aguilar/Library/Python/3.7/lib/python/site-packages/pip (python 3.7)

# Find where a package is installed
pip3 show awscli

Name: awscli
Version: 1.16.133
Summary: Universal Command Line Environment for AWS.
Home-page: http://aws.amazon.com/cli/
Author: Amazon Web Services
Author-email: UNKNOWN
License: Apache License 2.0
Location: /Users/carlos.aguilar/Library/Python/3.7/lib/python/site-packages
Requires: s3transfer, botocore, colorama, rsa, docutils, PyYAML

# Some notes on installing python3 with Homebrew
1- which python3
gives the location of the installation (/usr/local/bin/python3)
2- As it might be a link, work out the actual location
ls -al /usr/local/bin/python3
3 - which gives in turn 
/usr/local/Cellar/python/3.7.3/bin/python3

4 - Add the bin folder to the path in ~/.bash_profile to run pip3 installed applications
/Users/carlos.aguilar/Library/Python/3.7/bin
export PATH=/Users/carlos.aguilar/Library/Python/3.7/bin:$PATH





import ast
ast.literal_eval

vars()[varName]
globals()[varName]
locals()[varName]




















##
joined 8 weeks ago
Data science space
In this talk I'd like to give some insight on why I think interpretability is important.
To do so, we will touch on the following points
- machines, interpretability, cases that connect to reality
- Jane and Joe will be our partners in this journey
- 
- machine decisions (not pneumatic hammers)
- typically, these machines reach their conclusions through a black-box approach.
- Tim Miller associate professor at Melbourne university

##





Three conversion flags are currently supported: '!s' which calls str() on the value, '!r' which calls repr() and '!a' which calls ascii()
a = (4,3,5,3)
print('you say:{!r}'.format(a))
print('you say:{!s}'.format(a))




# Slicing
productDescription = ['Top notes', 'Raspberry, Cloudberry', 'Heart notes', 'Cream, Daisy', 'Launch date', '2018']
productDescription[0::2]
productDescription[1::2]

# More on slicing
hiddenNames = [name[0:3] + 'xxxx@yyyy' + name[-4::] for name in ['jdcfdeph-cooper@outlook.com','dafs.fonda@gmail.com']]


# Floor division in Python 
a//b

# Get the name of the file that is been run
import os
print(__file__)
print(os.path.abspath(__file__))


# Open a file
import os
filePath = 'fsfasdfasdf'
os.startfile(filePath)


# Check Python version inside of python
import platform
print(platform.python_version())


# Call a method changing the order of the arguments. That might look silly 
# but methods like 'parcial()' for multiprocessing only allow to have a non-fixed
# argument that must be lefttish one.
def myTest(a,b):
	print('a={} and b={}'.format(a,b))

myTest('this is it', 'this is not')

myTestSorted = lambda b,a: myTest(a,b)
myTestSorted('this is it', 'this is not')


# Speed up counts via collections (histogram-like)
from collections import Counter
cnt = Counter(word for word in ['red', 'blue', 'red', 'green', 'blue', 'blue'])


Get a list of the functions contained in a module:
	import utils.adformUtils as adform
	dir(adform)

# Get the classes manually
a = dfPurchases[['CookieID', 'orderID', 'yyyymmdd']]
a.iloc[0,0].__class__
a.iloc[0,1].__class__
a.iloc[0,2].__class__
b = dfTPminimal[['CookieID', 'orderID', 'yyyymmdd']]
b.iloc[0,0].__class__
b.iloc[0,1].__class__
b.iloc[0,2].__class__


# Get the classes manually
a = dfPurchases[['CookieID', 'orderID', 'yyyymmdd']]
a.iloc[0,0].__class__
a.iloc[0,1].__class__
a.iloc[0,2].__class__
b = dfTPminimal[['CookieID', 'orderID', 'yyyymmdd']]
b.iloc[0,0].__class__
b.iloc[0,1].__class__
b.iloc[0,2].__class__




# Reload modules:
import imp
imp.reload(my.module)


Exceptions:
    # Couldn't love PANDAS more...
    try:
        dfAdform = pd.read_csv(adformPath, delimiter='\t', compression='gzip');
    except Exception as e:
        dfAdform = pd.DataFrame()
        print('Cannot read {}'.format(adformPath))


String concatenate:
	a = [x + '_' + currentLagVar for x in varsToKeep]



Iterate through a list of files in a folder and show a progress bar as they are read:
	import tqdm as tqdm
	for img in tqdm(os.listdir(TRAIN_DIR)):
		#do stuff



# Equivalent to Matlab's ismember. Search values in a DF
filterName  = 'countryCodeISO3'
filterValue = ['gbr', 'irl']
idxFilter   = dfPurchasesExt[filterName].isin(filterValue);
# Or using str functionality
filterValue = 'gbr'
idxFilter   = dfPurchasesExt[filterName].str.find(filterValue) > 0



Types. perform differenct actions based on the datatype:

    if isinstance(filterValue, list):
        idxFilter = dfAdform[filterName].isin(filterValue);
    elif isinstance(filterValue, int):
        idxFilter = dfAdform[filterName] == filterValue;
    elif isinstance(filterValue, np.ndarray):
        idxFilter = dfAdform[filterName].isin(filterValue);
    elif isinstance(filterValue, pd.core.series.Series):
        idxFilter = dfAdform[filterName].isin(filterValue.tolist());
    else:
        idxFilter = 0;
        print('WARNING: Cannot work out the input datatype')
			...

Pause Python:
	import time
	time.sleep(5.5)





#Copy and paste to/from the clipboard:
	import pyperclip
	pyperclip.copy('Hello world!')
	pyperclip.paste()

#Delete a folder:
	metaZipFolder  = 'data/raw/adform/masterdataset';
	metaTempFolder = '/Users/carlos.aguilar/Documents/Beamly/Personalisation/adForm data/metaTemp';
	if os.path.exists(metaTempFolder):
		os.system('rm -rf "{}"'.format(metaTempFolder))



#Check if a file exists:
	import os.path
	os.path.isfile(fname)


#Wait for a process to finish:
	from subprocess import check_output
    connectionStr = ''' osascript -e 'tell application "Viscosity" to connect "{}"' '''.format(vpnName);    
    out = check_output(connectionStr, shell=true);


#Find the index of a value in a list:
	thisString = 'Impression_74350.csv.gz'
	idx = fileName.index(thisString)
	fileName[idx]
If the list contains the value several times:
	mainIDx    = [i for i,x in enumerate(folderName) if x == mainFolder];


#Compare lists (using sets):
engineers   = set(['John', 'Jane', 'Jack', 'Janice'])
programmers = set(['Jack', 'Sam', 'Susan', 'Janice'])

# missing in programmers
diffSets = list(engineers - programmers)
# missing in engineers
diffSetsB = list(programmers - engineers)

# easy (gives 2)
set([2,3,4])-set([3,4,5])

# intersect the sets
set([2,3,4]).intersection(set([3,4,5]))

# Make sure the contents are the same
presentVars = set(['John', 'Jane', 'Janice'])
varNamesSet = set(['John', 'Jane', 'Susan', 'Janice'])
varNamesSet = set(['John', 'Jane', 'Janice'])

allPresent = list(varNamesSet - intersectedSet) == []


#Produce permutations over lists:

import itertools
print(list(itertools.permutations(['a','b','c'], 2)))
To get a 'n choose k':
print(list(itertools.combinations(['a','b','c'], 2)))
Produce products:
list(itertools.product(['Ada','Quinn','Violet'],['Comp','Math','Sci']))


Access the index when iterating:
	for idx,ai in enumerate(a):


# Use a progress bar in python:
totalNumber = 1000
pbar = pyprind.ProgBar(totalNumber)
for i in range(0, totalNumber):
    pbar.update()

# Use a progress bar in python:
from tqdm import *
import time
totalNumber = 2500
for index in tqdm(range(totalNumber), desc='Fuego en el fuego...'):
	time.sleep(0.001)



dir() – will display the defined symbols. Eg: >>>dir(str) – will only display the defined symbols. 
Built-in functions such as max(), min(), filter(), map(), etc is not apparent immediately as they are
available as part of standard module. dir(__builtins ) to view them.

zip() function- it will take multiple lists say list1, list2, etc and transform them into a single list of 
tuples by taking the corresponding elements of the lists that are passed as parameters. 

Every object holds unique id and it can be obtained by using id() method. Eg: id(obj-name) will return unique id of the given object.

File-related modules in Python:
	os and os.path – modules include functions for accessing the filesystem
	shutil – module enables you to copy and delete the files.
	 “with” statement makes the exception handling simpler by providing cleanup activities.



_ has 3 main conventional uses in Python:

* To hold the result of the last executed statement in an interactive interpreter session. This precedent was set by the standard CPython interpreter, and other interpreters have followed suit
For translation lookup in i18n (see the gettext documentation for example), as in code like: raise forms.ValidationError(_("Please enter a correct username"))

* As a general purpose "throwaway" variable name to indicate that part of a function result is being deliberately ignored, as in code like: label, has_label, _ = text.partition(':')

* The latter two purposes can conflict, so it is necessary to avoid using _ as a throwaway variable in any code block that also uses it for i18n translation (many folks prefer a double-underscore, __, as their throwaway variable for exactly this reason).





# Categorical to numerical
uniqueType = stores2.type.unique()
uniqueTypeDict = dict(zip(uniqueType, range(len(uniqueType))))
stores2['type'] = stores2['type'].replace(uniqueTypeDict)

# Where is Python installed? Bin path to Python
which python3

# Manually add code to the path
import sys
[print(iP) for iP in sys.path]
pythonModsRoot = '/Users/carlos.aguilar/Google Drive/PythonDev/Coding/BeamlyPython'
if pythonModsRoot not in sys.path:
    sys.path.append(pythonModsRoot)


Print without newline
print('Querying...', end='')



Help on a module
help(moduleName)

PIP (Python Package Index)

Pip normalyy installs in the systems package folder, to tell PIP to install in the personal user folder, simply add the modifier
pip3 install --user PACKAGENAME 
Also, we can run PIP from Python as 
python -m pip3 install --user PACKAGENAME 

List of the installed packages:

pip3 list

pip3 unistall PACKAGENAME 

Upgrade a package:
pip3 install PACKAGENAME --Upgrade

pip3 help

search package:

pip3 search PACKAGENAME



# create a UUID
import uuid
currentSession = str(uuid.uuid4())

# Alternatives
import secrets
secrets.token_hex(16)




uninstall a package
-------------------

python3 setup.py install --record files.txt
cat files.txt | xargs rm -rf
Then delete also the containing directory, e.g. /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/my_module-0.1.egg/ on macOS. It has no files, but Python will still import an empty module:


Install a package and keep track of the installed files
--------------------------------------------------------
python3 setup.py install --record files_installed.txt


