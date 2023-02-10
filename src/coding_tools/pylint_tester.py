'''
	pylint_tester.py
	https://pypi.org/project/pylint/


	Pylint analyses your code without actually running it. 
	It checks for errors, enforces a coding standard, looks for code smells, 
	and can make suggestions about how the code could be refactored.


Pylint ships with three additional tools:

pyreverse (standalone tool that generates package and class diagrams.)
symilar (duplicate code finder that is also integrated in pylint)
epylint (Emacs and Flymake compatible Pylint)


Projects that you might want to use alongside pylint include:
- flake8 (faster and simpler checks with very few false positives), 
- mypy, pyright or pyre (typing checks), 
- bandit (security oriented checks), 
- black and isort (auto-formatting), 
- autoflake (automated removal of unused imports or variables), 
- pyupgrade (automated upgrade to newer python syntax) 
- pydocstringformatter (automated pep257).


From the L2R recommender:

tests: poetry run pytest
mypy: poetry run mypy .
flake8: poetry run flake8 .
black: poetry run black .
isort: poetry run isort .

David defines the following https://github.com/ef-carbon/l2r-optimizer/blob/master/setup.cfg

'''


pylint --recursive=y mydir mymodule mypackage

cd /Users/carlos.aguilar/Documents/EF_repos/data_science_utilities/src
pylint --recursive=y .



# generate class diagrams
pyreverse -o png -p yourpackage .


# Can't get this to work
pyreverse -o png -p ./EVC_API/classroomAI_DAT-73.py


pyreverse ./EVC_API/classroomAI_DAT-73.py -o png

