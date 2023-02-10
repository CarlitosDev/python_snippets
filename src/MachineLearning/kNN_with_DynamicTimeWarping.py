'''
kNN_with_DynamicTimeWarping.py

From this guy:
https://gist.github.com/nikolasrieble/8bd3a83e14c0b2fa66bfa2ddd8828717

Posted in here:
https://stackoverflow.com/questions/57015499/how-to-use-dynamic-time-warping-with-knn-in-python


KNN classification with custom metric (DTW Distance)
'''


import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#toy dataset 
X = np.random.random((100,10))
y = np.random.randint(0,2, (100))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#custom metric
def DTW(a, b):   
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0
    
    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

    return cumdist[an, bn]

#train
parameters = {'n_neighbors':[2, 4, 8]}
clf = GridSearchCV(KNeighborsClassifier(metric =DTW), parameters, cv=5)
clf.fit(X_train, y_train)

#evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

'''
	This tutorial also looks amazing
	https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping
	https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping/blob/master/K_Nearest_Neighbor_Dynamic_Time_Warping.ipynb

	A library to calculate DTW between time-series
	https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
'''