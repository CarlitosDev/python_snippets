Section 2: Creating a Package
------------------------------

Creating the package folder
Creating the __init__.py file
emacs PACKAGENAME/__init__.py
Inside of the __init__.py, type the name of the modules (to avoid issues with cases, etc)
__all__ = ['mod1', 'mod2']
Importing the new package

To get the list of folder that Python uses to find packages, do
import sys
print(sys.path)

Conventions for modules: should not start with numbers or capitals.

To add a package to the list of Python packages, 
sys.append(folderName)


Folders that don't contain an __init__.py file are 'namespace packages'. Namespace packages can be useful 
for a large collection of loosely-related packages 
(such as a large corpus of client libraries for multiple products from a single company). Each sub-package can now be separately installed, used, and versioned.


Relative import of a package with the trick: 
import .localModule

Absolute imports:
import utils.adformUtils as adform

If two modules import each other:
- move some of the code to a third module
- import one of the modules at the bottom of the other

To read the data files from a package:

from pkgutil import get_data
currentData = get_data(packagename, datafileName)
# if the data contains text:
currentData = get_data(packagename, datafileName).decode('utf8')