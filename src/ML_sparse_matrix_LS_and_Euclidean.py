import numpy as np
import scipy.sparse as ss
from sklearn.metrics import pairwise_distances

# Create a sparse matrix
A = ss.lil_matrix((10, 10))
A[0,1] = 82
A[0,2] = 2
# randomly
idx       = np.random.choice(10, 4, replace=False)
A[idx, :] = np.random.random(size=(4,10))

# to see it in dense format
A.toarray()
# create a regular column vector
B = np.random.random(size=(1,10))



# >> solve LS in sparse matrices
from scipy.sparse.linalg import spsolve
x = spsolve(A.transpose(), B.transpose())

# >> Euclidean distances
euclidean = pairwise_distances(A, B)
