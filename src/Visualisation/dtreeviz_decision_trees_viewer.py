'''
dtreeviz_decision_trees_viewer.py

GitHub here:
https://github.com/parrt/dtreeviz


python3 -m pip3 install dtreeviz --upgrade
(it might install pyspark)
'''




from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *

regr = tree.DecisionTreeRegressor(max_depth=2)
boston = load_boston()
regr.fit(boston.data, boston.target)

viz = dtreeviz(regr,
               boston.data,
               boston.target,
               target_name='price',
               feature_names=boston.feature_names)

viz.view()