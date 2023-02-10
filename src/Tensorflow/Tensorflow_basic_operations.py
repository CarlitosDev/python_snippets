import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
import time
from sklearn import datasets
import pandas as pd


# Names in TF2.0
# tf.random.normal instead of tf.random_normal

# because of reproducible results
tf.set_random_seed(1)






centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
df   = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
                     
y = df['target'].values.transpose()
#y = pd.get_dummies(df['target']).as_matrix()#.transpose();


varsForModel = df.columns.tolist()
varsForModel.remove('target')
X = df[varsForModel].values

# options for initial weights
m = len(X[0])
v_m = np.ones([1,m])*1/m
v_zero = np.zeros([1,m])





# Matrix mult in TF
# creating session and evaluation
config = tf.ConfigProto()
# don't pre-allocate memory
config.gpu_options.allow_growth = True
# create a session with specified option

sess = tf.Session(config=config)
# global variable initialization
sess.run(tf.global_variables_initializer())

# In TF    
A = X
norm_a = tf.reduce_sum(tf.multiply(tf.square(A), v_m), 1)
norm_a_reshaped = tf.reshape(norm_a, [-1, 1])

norm_a_results = norm_a_reshaped.eval(session=sess)
norm_a_results.shape

scalar_product = tf.matmul(tf.multiply(A, v_m), A, False, True)
scalar_product_result = scalar_product.eval(session=sess)

D = norm_a - 2*scalar_product + norm_a
D_result = D.eval(session=sess)

D = norm_a - 2*scalar_product + norm_b

sess.close()



# Equivalent in NP
np.dot(v_m, np.square(A[0]))
#np.dot(v_m, np.square(A))
norm_a_np = np.squeeze(np.dot(np.square(A), v_m.T))
norm_a_np.shape


# tf.square(x) Computes square of x element-wise.
# 
# np.dot(v_m, np.square(A[0]))
#
#
#


# Distances between teachers and student
teachers = np.array([[1,5,6], [2,2,6], [1,4,1], [8,3,1]])
student  = np.array([0.5,7,1])



# Distances between teachers and student
teachers = np.array([[4,7], [5,1]])
student  = np.array([8,3])

nt, nf = teachers.shape

v_m = np.ones(nf)/nf
v_m = np.ones(nf)

scaled_teachers = np.multiply(teachers,v_m)
scalar_product = np.dot(scaled_teachers, teachers.T)


norm_teachers = np.linalg.norm(teachers, axis=1)
norm_student = np.linalg.norm(student)




np.dot(teachers,v_m.T)

np.dot(v_m, teachers.T)

np.dot(np.square(A), v_m.T)

teachers.dot(v_m.T)

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


# paper's calculation
scaled_teachers = np.multiply(teachers, v_m)
norm_teachers = np.linalg.norm(scaled_teachers, axis=1)

scaled_student = np.multiply(student, v_m)
norm_student = np.linalg.norm(scaled_student)

scalar_product = np.dot(scaled_teachers, student.T)

D = norm_teachers - 2*scalar_product + norm_student




student  = np.array([55,0,30])

a = student.reshape([-1,1])
np.dot(a, a.T)
np.abs(a - a.T)



student - student.T