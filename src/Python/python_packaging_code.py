'''
	python_packaging_code.py
	
	Following the official tutorial from https://packaging.python.org/tutorials/packaging-projects/

'''


# (1) create a __init__.py file where the source code is stored.
# (2) In the root folder, create a setup.py which should look like:



import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    begin_description = '\n\n# ds-lead-scores\n'
    end_description = '\n\n## Synopsis'

    idx_begin_description = long_description.find(begin_description) + len(begin_description)
    idx_end_description = long_description.find(end_description)
    project_description = long_description[idx_begin_description:idx_end_description]
    

setuptools.setup(
    name="ds-lead-scores",
    version="0.1.0",
    description="Class for scoring EF marketing leads",
    long_description=project_description,
    long_description_content_type="text/markdown",
    author='Carlos Aguilar',
    author_email='carlos.aguilar@ef.com',
    url="https://github.com/efcloud/ds-lead-scores",
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3"],
    python_requires='>=3.6',
)



# (3) upgrade setup tools
python3 -m pip install --user --upgrade setuptools wheel

# (4) Package it!
python3 setup.py sdist bdist_wheel