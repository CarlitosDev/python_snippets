# Version control for the modules:
from distutils.version import LooseVersion
import sklearn as sk
if LooseVersion(sk.__version__) < '0.18':
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import GridSearchCV

# sklearn version:
import sklearn as sk
sk.__version__

# Pandas version:
import pandas as pd
pd.__version__

# Bokeh version:
import bokeh as bk
bk.__version__





# Confirm that we're using Python 3
import sys
assert sys.version_info.major is 3, 'Not running Python 3'