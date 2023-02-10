https://dcor.readthedocs.io/en/latest/index.html
https://dcor.readthedocs.io/en/latest/theory.html
https://github.com/vnmabus/dcor


Functional data analysis in Python 
https://github.com/GAA-UAM/scikit-fda
https://github.com/GAA-UAM


Library for experiments in Python
https://sacred.readthedocs.io/en/stable/index.html

#pip3 install dcor --upgrade


import dcor

distcorr = lambda column1, column2: dcor.distance_correlation(column1, column2)
rslt = var_importance_all_feats.apply(lambda col1: var_importance_all_feats.apply(lambda col2: distcorr(col1, col2)))
rslt -= np.eye(rslt.shape[0], rslt.shape[1])

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(rslt, annot=True, linewidths=.5, ax=ax)
plt.show()
