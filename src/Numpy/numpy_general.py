# Blunt ind2sub
# idx2sub
total_days = df_store.shape[0]
num_days = np.arange(total_days)
promo_sku_A = df_store[f'promotion_flag-{sku_id_A}-{store_name}']
num_days[promo_sku_A]



Numpy arrays are stored in contiguous memory space.


# Map values
x = np.array([1, 2, 3, 4, 5])
y = x[::-1]
y[0] = 12
mp = dict(zip(x, y))
np.vectorize(mp.get)(x)


# Hadamard product
np.multiply(a,b)


Array of boolean:
import numpy as np
np.ones((2, 2), dtype=bool)


# Get triangular matrices
mask = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
np.triu(mask,1)


Grow up arrays in an efficient way:
	M = []
	e = []
	numTimes = round(numRecords*0.5)

	for idx in range(0, numTimes):
		M.append(np.power(X[remainingPromos] - X[currentPromo], 2))
		e.append(np.power(Y[remainingPromos] - Y[currentPromo], 2))
	# concatenate horizontally		
	M = np.concatenate(M, axis=0).copy()
	e = np.concatenate(e, axis=0).copy()



(Have a look at this post https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
as this is very inefficient)
# create an empty vector and grow it dynamically
vector_q = np.empty([0.0])
vector_q = np.append(vector_q, 0.3)
vector_q = np.append(vector_q, 9.23)	
print(vector_q)

# It must be a way of using this better
vector_q = np.empty([0.0])
np.insert(vector_q, [0], 0.3)





# Get an array as a column-one
x = np.array([[1, 2, 3], [4, 5, 6]])
print(np.ravel(x))
[1 2 3 4 5 6]

# Random integers
np.random.randint(2, high=10, size=10)


# Some basic operations
myBID        = np.array([1.76197, 1.73945]);
numberStocks = np.array([20, 1980]);
totalShares  = np.sum(numberStocks);
grossBook    = np.dot(numberStocks, myBID);

# Numerical encoding:
# Create a label encoder and fit it
	from sklearn.preprocessing import LabelEncoder
	le_sex = LabelEncoder()
	le_sex.fit(df["sex"].unique())
	le_sex.classes_  # The fit results in two classes - M and F



# Save numpy data:
import numpy as np
np.save('filename', npVarName)



# Remove a value from a NP array
a = np.array([3,7,2])
b = 7
c = a[a != b]



# Get the indices after sorting an array:
np.argsort
# Descending order...
idxSorted = np.argsort(classProb)[0][::-1]

If result is an array nxM:
idxSorted = np.argsort(result[0], axis=-1)[::-1]


## From logical to float
baseKnowledge = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 1]])
np.all(baseKnowledge, axis=1).astype('float64')

# Index based on false/true
idx = np.all(baseKnowledge, axis=1)
weights = np.zeros(n)
weights[idx] = 543;


# Concatenate vectors
n = 2
weights = np.concatenate([np.zeros(n), np.ones(n)])

