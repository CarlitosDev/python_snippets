'''
	Find a wheel within Pypi
'''

import requests
def find_python_wheel_url(package_name, package_version,
  required_python_version = 'cp38', required_system = 'linux', required_arc='x86_64'):
  base_url = 'https://pypi.org/pypi/' + package_name + '/' + package_version + '/json'
  response = requests.get(base_url)
  try:
    dc_response = response.json()
    for this_url in dc_response['urls']:
      if (this_url['python_version'] == required_python_version) & \
        (required_system in this_url['url']) &\
        (required_arc in this_url['url']):
        print(f'''wget {this_url['url']}''')
  except:
    print(package_name, 'Failed')

list_of_wheels = ['scikit_learn-0.24.2',
'scipy-1.6.3',
'numpy-1.20.2',
'pandas-1.2.4',
'install-1.3.4',
'transformers-4.5.1',
'nltk-3.6.2',
'tokenizers-0.10.2',
'regex-2021.4.4',
'sacremoses-0.0.45',
'joblib-1.0.1',
'sentencepiece-0.1.95',
'pytz-2021.1',
'blis-0.7.4',
'pydantic-1.7.3',
'spacy-3.0.6',
'pip-21.1.1', 'thinc-8.0.3']


for this_wheel in list_of_wheels:
  this_package_name, this_package_version = this_wheel.split('-')
  find_python_wheel_url(this_package_name, this_package_version)