import numpy as np






# Euclidean Distance Matrix 
# ie: distances between students
# https://math.stackexchange.com/questions/1199380/what-is-the-intuition-behind-how-can-we-interpret-the-eigenvalues-and-eigenvec
student  = np.array([55,0,30])
a = student.reshape([-1,1])
M = np.abs(a - a.T)

w, v = np.linalg.eig(M)

# same as vh == u
#u, s, vh = np.linalg.svd(M)





import numpy as np
X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_