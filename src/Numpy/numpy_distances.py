import numpy as np


# Scaled Euclidean distances
# https://www.coursera.org/lecture/ml-clustering-and-retrieval/writing-scaled-euclidean-distance-using-weighted-inner-products-JMU6y


list_to_col_vector = lambda lst: np.array(lst).reshape(-1,1)
list_to_row_vector = lambda lst: np.array(lst).reshape(1,-1)

x_i = list_to_row_vector([2.0, 1.0])
x_q = list_to_row_vector([5.0, 3.0])

dif_xi_xq = x_i-x_q

# Plain Euclidean
plain_Euclidean = np.sqrt(np.dot(dif_xi_xq,dif_xi_xq.T)).squeeze()


# Scaled Euclidean
weights = [1.0, 1.0]
V = np.diag(weights)
scaled_Euclidean = np.sqrt(np.dot(np.dot(dif_xi_xq, V), dif_xi_xq.T)).squeeze()




# Euclidean Distance Matrix 
# ie: distances between students
# https://math.stackexchange.com/questions/1199380/what-is-the-intuition-behind-how-can-we-interpret-the-eigenvalues-and-eigenvec
student  = np.array([55,0,30])
a = student.reshape([-1,1])
M = np.abs(a - a.T)

w, v = np.linalg.eig(M)



# Distances between teachers and student
teachers = np.array([[1,5,6], [2,2,6], [1,4,1], [8,3,1]])
student  = np.array([0.5,7,1])

# repmap equivalent if broadcast doesn't work
m,n = teachers.shape
students = np.tile(student,(m,1))





# What I expect:
ts_diff = np.power(teachers[1]-student, 2.0)
scaled_diff = np.multiply(ts_diff, v_m)
scaled_dist = scaled_diff.sum()
np.sqrt(scaled_dist)


# Comando calculation
a = teachers-student
np.sqrt(np.dot(a, a.T))

# Scaled  calculation
v_minus_1 = np.linalg.inv(np.diag(np.power(v_m, 2.0)))
dist_matrix = np.sqrt(np.dot(np.dot(a,v_minus_1),a.T))

distances = np.diag(dist_matrix)







from scipy.spatial import distance
import pdist, cosine, jaccard

# This thing only works with 1D arrays
m,n = teachers.shape

#cosine similarity
cosine_distances = [1 - distance.cosine(student, this_teacher) for this_teacher in teachers]

# Bray-Curtis distance
braycurtis_distances = [1 - distance.braycurtis(student, this_teacher) for this_teacher in teachers]

#  distance
canberra_distances = [1 - distance.canberra(student, this_teacher) for this_teacher in teachers]

# chebyshev distance
chebyshev_distances = [1 - distance.chebyshev(student, this_teacher) for this_teacher in teachers]

# cityblock distance
cityblock_distances = [1 - distance.cityblock(student, this_teacher) for this_teacher in teachers]

# correlation distance
correlation_distances = [1 - distance.correlation(student, this_teacher) for this_teacher in teachers]


# jaccard distance
jaccard_distances = [1 - distance.jaccard(student, this_teacher) for this_teacher in teachers]

# seuclidean distance
seuclidean_distances = [1 - distance.seuclidean(student, this_teacher, [1,1,1]) for this_teacher in teachers]

#  distance
sqeuclidean_distances = [1 - distance.sqeuclidean(student, this_teacher, ) for this_teacher in teachers]


#a = np.array([2.0, 1.0])
#b = np.array([5.0, 3.0])
#np.sum([a,b])














## Adjust the Euclidean distance
from sklearn.metrics import pairwise_distances

list_to_col_vector = lambda lst: np.array(lst).reshape(-1,1)
list_to_row_vector = lambda lst: np.array(lst).reshape(1,-1)

x_i = list_to_row_vector([25.0, 1.0])
x_q = list_to_row_vector([58.0, 3.0])

dif_xi_xq = x_i-x_q

weights = [1.0, 1.0]
V = np.diag(weights)
scaled_Euclidean = np.sqrt(np.dot(np.dot(dif_xi_xq, V), dif_xi_xq.T)).squeeze()


X_train = np.array([[25.0, 34, 1.0, 0.1],[29.0, 66, 8.0, 0.25],[38, 300, 3.0, 0.4]])
X_test  = np.array([59.0, 100,  0.3, 0.12]).reshape(1,-1)

numFeatures = 4
num_neighbours = 2
feat_importances = np.ones(8)

x_weights = feat_importances[0:numFeatures]
x_ref_weights = feat_importances[numFeatures::]

# Prepare for the Euclidean distance
r_test   = X_test.shape[0]
t_test   = X_train.shape[0]
X_temp   = np.concatenate([X_test, X_train], axis=0)
x_normed = (X_temp - X_temp.min(0)) / np.maximum(X_temp.ptp(0), 1)

# Scale the test set. Select wether to use the reference or just the train set weights
X_test_scaled = np.multiply(x_normed[0:r_test, :], x_ref_weights)

# Scale the training set
X_train_scaled = np.multiply(x_normed[r_test::, :], x_weights)

euclidean = np.squeeze(pairwise_distances(X_train_scaled, X_test_scaled, metric='euclidean'))
idxSorted = np.argsort(euclidean)[0:num_neighbours]

cityblock = np.squeeze(pairwise_distances(X_train_scaled, X_test_scaled, metric='cityblock'))
idxSorted = np.argsort(euclidean)[0:num_neighbours]

# Plain Euclidean
dif_xi_xq = X_train_scaled[0, :]-X_test_scaled
plain_Euclidean = np.sqrt(np.dot(dif_xi_xq,dif_xi_xq.T)).squeeze()


###
from sklearn.preprocessing import StandardScaler
x_normed = StandardScaler().fit_transform(X_temp)




idxSorted = np.argsort(classProb)[0][::-1]




##
from sklearn.metrics import pairwise_distances
a = np.array([1,2,3]).reshape(1,-1)
b = np.array([0.5,0.4,0.3]).reshape(1,-1)
euclidean = np.squeeze(pairwise_distances(a,b))
np.sqrt(np.sum(np.power(a-b,2)))



import numpy as np

b = np.ones((10,10))
m,n = b.shape
a = np.ones((10,1))*2
a_exp = np.tile(a,(1,m))
a_exp.shape
ab = np.matmul(a_exp, b)



# Taxicab (Manhattan) distance

a = [1,1]
b = [1,3]
c = [1,4]
d = [2,1]
e = [3,3]
f = [3,4]
g = [4,1]

boxes = np.array([a,b,c,d,e,f,g])
np.sum(np.abs(boxes[1,:]-boxes[0,:]))
np.sum(np.abs(boxes[1,:]-boxes[6,:]))

from scipy.spatial.distance import cdist
# 
taxicab_dist = cdist(boxes, boxes, metric='cityblock')
taxicab_dist.shape

# Let's say because we read from left to right that the distance
# calculation should _favour_ items that connect from left to right
# So let's say that it is easier to travel from 'a' to 'c' 
# than from 'a' to 'b'
a = [1,1]
b = [2,1]
c = [1,2]
boxes = np.array([a,b,c])
taxicab_dist = cdist(boxes, boxes, metric='cityblock')


w = [1.5,1]
boxes_w = np.multiply(boxes, w)
taxicab_dist_w = cdist(boxes_w, boxes_w, metric='cityblock')

