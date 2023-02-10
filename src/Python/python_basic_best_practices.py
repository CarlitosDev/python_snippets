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

My usual:

git add -A
git commit -m "this is the message"
git push origin master
 
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
sphinx-quickstart
It will ask for a root folder ('docs'). make sure the autodoc plugin is enabled.
sphinx-apidoc -o docs example
(sphinx-apidoc -o docs pandas -f)

emacs conf.py
In the conf.py file add 
	import os
	import sys
	sys.path.append(os.path.abspath('..'))
make html

From the Sphix-doc web:
sphinx-quickstart
sphinx-build -b html sourcedir builddir

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



from dis import dis
dis(_)

import builtins
builtins.__build_class__
<built-in function __build_class__>