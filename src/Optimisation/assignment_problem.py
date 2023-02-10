'''

assignment_problem using Jonker-Volgenant algorithm


cd /Users/carlos.aguilar/Documents/EF_repos
git clone https://github.com/gatagat/lap.git
cd lap

python3 setup.py build
python3 setup.py install




'''


'''
Get the example from here:
https://en.wikipedia.org/wiki/Hungarian_algorithm

	Clean bathroom	Sweep floors	Wash windows
Paul	$2	$3	$3
Dave	$3	$2	$3
Chris	$3	$3	$2

'''
import numpy as np
C = np.array([[2,3,3], [3,2,3],[3,3,2]])

C[2,:]

from lap import lapjv as lap_lapjv
cost, x, y = lapjv(C)

'''     t1  t2  t3
Paul	  $2	$3	$1
Dave	  $3	$2	$3
Chris	  $1	$3	$2
Richard $1  $1  $1
'''
# NxM = 4X3
C2 = np.array([[2,3,1], [3,2,3],[1,3,2],[1,1,1]])
cost, x, y = lap_lapjv(C2,extend_cost=True)

# y is a size-M array specifying to which row each column is assigned. 
# [2, 3, 0] >> Chris (1), Richard (1), Paul (1)

# x is [ 2, -1,  0,  1]
# So Paul does t3, Dave is not assigned, Chris does t1 and Richard does t2


# Use the Scipy's solver
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(C2)
cost = C2[row_ind, col_ind].sum()

# The Scipy's solver allows to maximise
# It could be used for example in the Au Pair problem
# We will have to work out all the (au pair, family) rankings as the cost matrix
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(maximize=True)


# Another project. This needs C2 to be square
# pip3 install lapjv
# from lapjv import lapjv as lapjv_2
# row_ind, col_ind, _= lapjv_2(C2)


