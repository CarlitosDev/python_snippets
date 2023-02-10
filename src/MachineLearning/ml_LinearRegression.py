import numpy as np

dfInterpolation = dfBeamlyGRP[idxMobile].copy()
idxToRemove     = dfBeamlyGRP.total_products == 0
x = dfInterpolation[~idxToRemove].num_orders.as_matrix()
y = dfInterpolation[~idxToRemove].total_products.as_matrix()

A    = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]

print(m, c)

dfInterpolation.total_products[idxToRemove] = m*dfInterpolation[idxToRemove].num_orders + c