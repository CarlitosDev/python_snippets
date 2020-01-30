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

