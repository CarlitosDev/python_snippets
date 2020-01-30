Yellowbrick library to diagnose and visualise scikit regressors/classifiers - see the confussion matrix for example

http://www.scikit-yb.org/en/latest/
pip3 install yellowbrick
https://github.com/DistrictDataLabs/yellowbrick/blob/develop/examples/examples.ipynb

from yellowbrick.features import Rank2D

visualizer = Rank2D(features=features, algorithm='covariance')
visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data

Use dummy_classifier amd dummy_regressor as a way to start with a ML problem and benchmark against it.

PermutationImportance: permute the columns to understand which columns make the model worse.

Use tSNE to visualise error in prediction. Colour the dataset by error.



#####

from yellowbrick.classifier import DiscriminationThreshold
visualizer = DiscriminationThreshold(algorithm)
visualizer.fit(training_x,training_y)
visualizer.poof()