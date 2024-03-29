'''

Python virtual environments is to create an isolated environment for Python projects. 
Each project can have its own dependencies, regardless of what dependencies every other project has.

Each virtual environment has its own Python binary (which matches the version of the binary that was used to create this environment) 


'''


'''
To install different Python versions
'''
# 1
pyenv install 3.7.7
pyenv install 3.8.3
pyenv global 3.8.3
# 2. In the folder of the new project that I want to run
# It will add a file cat .python-version
pyenv local 3.7.7
# Now source ~/.bash_profile
# if it does not work, then do: 
eval "$(pyenv init -)"
python3 -V
# 3. 
python3 -m venv env
source env/bin/activate
# 4. Run my usual pip install
pip3 install --upgrade pip
pip3 install matplotlib --upgrade --ignore-installed
pip3 install pyarrow --upgrade --ignore-installed
pip3 install pandas --upgrade --ignore-installed
pip3 install numpy --upgrade --ignore-installed
pip3 install pygraphviz

pip3 install causalnex
deactivate




cd /Users/carlos.aguilar/Documents/EF_repos

mkdir superset-virtualenv && cd superset-virtualenv

python3 -m venv env
# It will not include any of your existing site packages.
'''
Here’s what each folder contains:

	bin: files that interact with the virtual environment
	include: C headers that compile the Python packages
	lib: a copy of the Python version along with a site-packages folder where each dependency is installed

'''


# Activate the environment
$ source env/bin/activate

# (deactivate simply 'deactivate')



############
# Example: Install Apache Superset in a virtual environment
############

# (1) Shell commmands
mkdir superset-virtualenv && cd superset-virtualenv
python3 -m venv env
source env/bin/activate


# (2) Shell+PIP commands 
pip install --upgrade setuptools pip

# Some more stuff for my connections
pip install psycopg2
pip install pymssql


# Install superset
pip install superset



# >> Workaround for the pandas issue as of 1st July 2019
'''

Replace the two occurrences of _maybe_box_datetimelike 
with maybe_box_datetimelike in superset/dataframe.py directly in the installed package. 
It's a workaround of course but apparently it works

In here:
cd ./env/lib/python3.7/site-packages/superset/dataframe.py


It will work with pandas 0.24

carlos

#pip install pandas --upgrade
'''

# Create an admin user (you will be prompted to set a username, first and last name before setting a password)
fabmanager create-admin --app superset



# Initialize the database
# >> Also: more workarounds - fixed
pip uninstall sqlalchemy
pip install sqlalchemy==1.2.18

superset db upgrade




# Load some data to play with
superset load_examples
# Create default roles and permissions
superset init
# To start a development web server on port 8088, use -p to bind to another port
superset runserver -d


# Internals:
# To connect to Apollo
mssql+pymssql://user.carlos.aguilar:efef@123!@10.162.85.20/Apollo



#######
# Install a repo in a venv
git clone git@github.com:efcloud/ml-smartmatch.git
python3 -m venv env
source env/bin/activate

python ./smartmatcher/setup.py install




#######
# Install SB
git clone git@github.com:efcloud/
python3 -m venv env
source env/bin/activate

cd ./src
python ./setup.py install
# (deactivate simply 'deactivate')


######
# For Autogluon in the seconds paper
python3 -m venv env
source env/bin/activate
pip install --upgrade pip

pip install pandas catboost numpy matplotlib sklearn xlsxwriter autogluon mxnet ngboost category_encoders seaborn tornado

cd ./src
python ./setup.py install
# (deactivate simply 'deactivate')