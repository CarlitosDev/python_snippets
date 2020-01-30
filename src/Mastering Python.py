Mastering Python

Help on a module
help(moduleName)

----------
PIP (Python Package Index)
----------

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

------------------------------
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

------------------------------
Section 3: Basic Best Practices
------------------------------
PEP 8 (Python Enhancement Proposals) and Writing Readable Code

Objects names according to how they are used, not to what they are.
Classes: CamelCase
Functions/Methods: Lowercase with underscore. Internal variables should be preceded by underscore.
Constants: capitals

Don't use tab characters for indentation, use 4 spaces.

3.2 Using Version Control

git init
git add fileName.py
git commit -a

Review the commits:
git log

In case we want to revert to a previous commit, find the id and reverts as
git checkout IDpreviousOne

Create branches:
git branch -t branchName
git checkout branchName

Merge into the master branch
git merge branchName
If overlapping changes:
git mergetool

To include code from other repositories, use git pull 

More on GIT: https://devguide.python.org/gitbootcamp/


3.3 Using venv to Create a Stable and Isolated Work Area

venv allows us to install a package without interfering with the current installation.

i - Create:
python3 -m venv venvExample
ii - Activate it so we can work on it:
source bin/activate
iii - Run any command. Install some private packages using pip:
pip install pillow

iv - to deactivate, simply type 'deactivate'

Getting the Most Out of docstrings Part 1 – PEP 257 and Sphinx:

Basic rules for docstrings:
- use triple quotes
- 1st line is a short description
- Blank-line and then a more detailed description

Using reStructuredText. Follow https://devguide.python.org/

Sphinx:
type sphinx-quickstart
It will ask for a root folder ('docs'). make sure the autodoc plugin is enabled.
sphinx-apidoc -o docs example
emacs conf.py
In the conf.py file add 
	import os
	import sys
	sys.path.append(os.path.abspath('..'))
make html


Getting the Most Out of docstrings Part 2 – doctest
To enable doctest to run python code, we actually write the test as if we were in the python shell, ie:
"""Doctest example
>>> for idx in range(5):
...		print(idx)
0
1
2
3
4

"""

python3 -m doctest -v fileName.py


------------------------------
Section 4: Creating a Command-line Utility
------------------------------
Making a Package Executable via python – m

Always use 
if __name__ == '__main__':
when running the main module as it avoids problems when importing the modules. Also, another cool thing 
is that we can just type 'python -m folderName' and Python will recognise the file with the __main__ tag and run it.
(On the example python /BeamlyRepos/venvExample)
python -m pipeline


Handling Command-line Arguments with argparse
Text-mode Interactivity
Executing Other Programs
Using Shell Scripts or Batch Files to Launch Programs

------------------------------
Section 5: Parallel Processing 
------------------------------
Using concurrent.futures
Using Multiprocessing
------------------------------
Section 6: Coroutines and Asynchronous I/O
------------------------------


Understanding Why Asynchronous I/O Isn't Like Parallel Processing
Using the asyncio Event Loop and Coroutine Scheduler
Futures
Making Asynchronous Tasks Interoperate
Communicating across the Network
------------------------------
Section 7: Metaprogramming 
------------------------------

Using Function Decorators
Using Function Annotations
Using Class Decorators
Using Metaclasses
Using Context Managers
Using Descriptors


------------------------------
Section 8: Unit Testing 
------------------------------

Understanding the Principles of Unit Testing
Using unittest
Using unittest.mock
Using unittest's Test Discovery
Using Nose for Unified Test Discovery and Reporting
