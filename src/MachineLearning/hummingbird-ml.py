from sklearn import datasets
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hummingbird.ml import convert
import time

# Create some random data for binary classification
X, y = datasets.make_classification(n_samples=1000, n_features=28)
# Create and train a model (scikit-learn RandomForestClassifier in this case)
skl_model = RandomForestClassifier(n_estimators=1000, max_depth=10)
skl_model.fit(X, y)



# Using Hummingbird to convert the model to PyTorch
model = convert(skl_model, 'pytorch')
print(type(model))

queryStart = time.time()
skl_model.predict(X)
queryEnd = time.time()
queryElapsed = queryEnd - queryStart
print('...done in {:.2f} sec!'.format(queryElapsed))


queryStart = time.time()
model.predict(X)
queryEnd = time.time()
queryElapsed = queryEnd - queryStart
print('...done in {:.2f} sec!'.format(queryElapsed))