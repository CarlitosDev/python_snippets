python3.5.2 -m venv env-for-swagger

brew install pyenv
pyenv install 3.5.2

pyenv versions
pyenv local 3.5.2

Add to the ~/.bash_profile 
the line eval "$(pyenv init -)"

https://github.com/pyenv/pyenv/blob/master/COMMANDS.md


If there is an installation file, run python3 setup.py install and it will
install all the libraries for the especified Python version.