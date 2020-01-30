# To review this script as it is c&p 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from pandas.tools.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.externals import joblib

# These code lines assess the performances of different models with respect to the given data

models = []
models.append(('CRT', DecisionTreeRegressor(random_state = 42)))
models.append(('RDF', RandomForestRegressor(n_estimators = 1000, random_state = 42)))
models.append(('LNR', LinearRegression()))
models.append(('RID', Ridge(random_state = 42)))
models.append(('LAR', Lasso(random_state = 42)))
models.append(('MLR', MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000,random_state = 42)))

# Evaluating each model in turn by using a R2 score. Consider any negative value as R2 = 0

results = []
names = []
for name, model in models:
    
    model.fit(X_product_train,Y_product_train)
    Y_pred = model.predict(X_product_test)
    r2 = r2_score(Y_product_test, Y_pred)
     
    results.append(r2)
   
    names.append(name)
    msg = "%s: %f" % (name, r2)
   
    print(msg)