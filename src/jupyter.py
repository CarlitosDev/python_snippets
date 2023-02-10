Use Jupyter notebooks with parallel computing
pip3 install ipyparallel
ipcluster nbextension enable


Start it: jupyter notebook

jupyter nbconvert --to script 'my-notebook.ipynb'


# Auto generate notebook
import nbformat as nbf
nb = nbf.v4.new_notebook()
text = """\
# My first automatic Jupyter Notebook
This is an auto-generated notebook."""


with open('analyseDataForJoopAndTPS.py') as f:
	pythonCode = f.read()

nb['cells'] = [nbf.v4.new_markdown_cell(text),
               nbf.v4.new_code_cell(pythonCode)]
fname = 'test.ipynb'

with open(fname, 'w') as f:
    nbf.write(nb, f)


