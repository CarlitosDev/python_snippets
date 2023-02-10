'''
	bernoulli_MultiArmedBandits.py

	This script comes from talking to David Chang who has a great post about Thompson sampling.

	https://chanind.github.io/2021/10/11/self-optimizing-ab-tests.html

	
	This post is also great:
	https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html
	
'''


'''
	So basically, we have 3 variations of our web and we count the conversions and trials for each.

	A and B are equal in terms of conversion rate (0.266), although the population of B is twice the population of A. 
	C has a larger population but slightly lower conversion rate (0.20) than A and B.
	
	By using Thompson sampling (see David's article about the alpha and beta parameters) we can estimate which variant
	will yield higher conversion rates.



'''


import numpy as np

mc_sampling_size = 1000

# branch A
num_conversions_A = 400
num_trials_A = 1500
num_conversions_A/num_trials_A
a = 1 + num_conversions_A
b = 1 + num_trials_A - num_conversions_A
branch_a = np.random.beta(a, b, size=mc_sampling_size)

# branch B
num_conversions_B = 800
num_trials_B = 3000
num_conversions_B/num_trials_B
a = 1 + num_conversions_B
b = 1 + num_trials_B - num_conversions_B
branch_b = np.random.beta(a, b, size=mc_sampling_size)

# branch C
# more people. Slightly smaller CR
num_conversions_C = 1600
num_trials_C = 8000
num_conversions_C/num_trials_C
a = 1 + num_conversions_C
b = 1 + num_trials_C - num_conversions_C
branch_C = np.random.beta(a, b, size=mc_sampling_size)

mean_branch_a = np.mean(branch_a)
mean_branch_b = np.mean(branch_b)
mean_branch_c = np.mean(branch_C)