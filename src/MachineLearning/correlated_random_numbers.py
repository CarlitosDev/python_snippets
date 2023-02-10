#correlated random numbers
'''

Example for 2 signals

Good explanation:
https://stats.stackexchange.com/questions/38856/how-to-generate-correlated-random-numbers-given-means-variances-and-degree-of


https://quantcorner.wordpress.com/2018/02/09/generation-of-correlated-random-numbers-using-python/
https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html


'''

import numpy as np

num_samples = 100
price_mu = 4
price_sigma=2
sales_mu = 100
sales_sigma = 20

# 1 - Correlation matrix
rho = 0.65
corr_mat = np.array([[1.0, rho],
                    [rho, 1.0]])
# 2 - Decompose C
# Compute the (upper) Cholesky decomposition matrix
L = np.linalg.cholesky(corr_mat)

# 2.b - For a pair of signals, L is quite simple
# if rho is the desired correlation
# L = np.array([[1,rho], [0,np.sqrt(1-rho**2)]])
L = np.array([[1,rho], [0,np.sqrt(1-rho**2)]])

# 3 - Define random gaussian signals
noise = np.random.normal(0.0, 1.0, size=(500, 2))
correlated_signals = np.matmul(noise, L)

# 4 - Scale the signals
x1 = correlated_signals[:,0]*price_sigma + price_mu
x2 = correlated_signals[:,1]*sales_sigma + sales_mu


from scipy.stats import pearsonr
corr_0_1 , _ = pearsonr(x1, x2)
print(corr_0_1)

x1.mean()
x2.std()


import pandas as pd
df = pd.DataFrame({
  'baseline_sales': x2,
  'price': x1
})

import seaborn as sns; sns.set(style="white", color_codes=True)
g = sns.jointplot('price', 'baseline_sales', data=df, 
  kind="reg", color="b")
plt.grid()
plt.show()



