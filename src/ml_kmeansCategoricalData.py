import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes





df = pd.DataFrame([{'A': 'foo' , 'B': 'green' , 'C': 2}, \
				   {'A': 'bar' , 'B': 'blue'  , 'C': 2}, \
				   {'A': 'beer', 'B': 'red'   , 'C': 20}, \
				   {'A': 'bar' , 'B': 'green' , 'C': 20}, \
				   {'A': 'foo' , 'B': 'blue'  , 'C': 20}])

# Former 'as_matrix()'
npArray = df.values;
npArray[:, 2] = npArray[:, 2].astype(float)

kproto   = KPrototypes(n_clusters=2, init='Cao', verbose=2)
clusters = kproto.fit_predict(npArray, categorical=[0,1])

# Print cluster centroids of the trained model.
print(kproto.cluster_centroids_)
# Print training statistics
print(kproto.cost_)
print(kproto.n_iter_)

for s, c in zip(npArray[:, 2], clusters):
    print("Symbol: {}, cluster:{}".format(s, c))