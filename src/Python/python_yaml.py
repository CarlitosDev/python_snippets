'''
pip3 install PyYAML --upgrade
'''





import yaml

yaml_string = '''
kind: job
metadata:
  name: xgb
  tag: ''
  hash: d28144e2b0205d379d
  project: ''
spec:
  command: src/python_yaml.py
  args: []
  env: []
  description: ''
  build:
    base_image: images/this_image
    source: './'
    commands:
    - pip install sklearn
    - pip install xgboost
    - pip install matplotlib
    code_origin: manolo.lolo
'''

# Read as a dict
d = yaml.safe_load(yaml_string)
d.keys()

# import pandas as pd
# pd.DataFrame.from_dict(d)