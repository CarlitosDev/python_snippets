'''

	Distances between distributions

'''


'''
Kolmogorov-Smirnov

Are my samples drawn from the same distribution?
It’s a simple statistical test that based on a value alpha 
that allows to discriminate between hypothesis, 
being H0 that the samples come from the same distribution
'''




'''

Wasserstein distance to measure differences in the distribution. 

Unfortunately, KS does not capture small changes which apparently the Earth Movers distance does.
 In the 1D case, you can get them from the difference in th CDFs. There is another formulation using the quantile functions.

'''

import numpy as np
from scipy.stats import wasserstein_distance


bins_d1 = [0,1,2,3]
freq_d1 = [10,6,3,0]

bins_d2 = [0,1,2,3]
freq_d2 = [10,6,3,0]

print('Equal distributions')
print(wasserstein_distance(bins_d1,bins_d2, freq_d1, freq_d2))

print('Equal distributions but different binning')
print(wasserstein_distance(bins_d1,np.dot(bins_d2,2), freq_d1, freq_d2))

print(wasserstein_distance(freq_d1,np.dot(freq_d1,2)))
print(wasserstein_distance(freq_d1,np.dot(freq_d1,3)))


print('Similar distributions but one is a bit skewed towards the right tailed')
freq_d2_tail = freq_d1.copy()
print(wasserstein_distance(bins_d1, bins_d1, freq_d1, freq_d2_tail))
for _ in range(10):
	print('Increasing the long tail by one unit')
	freq_d2_tail[-1] += 1
	print(wasserstein_distance(bins_d1, bins_d1, freq_d1, freq_d2_tail))



print('Similar distributions but one is a bit skewed towards the right tailed')
freq_d2_tail = freq_d1.copy()
print(wasserstein_distance(bins_d1, bins_d1, freq_d1, freq_d2_tail))
for _ in range(10):
	print('Increasing the second position of the histogram by one unit')
	freq_d2_tail[1] += 1
	print(wasserstein_distance(bins_d1, bins_d1, freq_d1, freq_d2_tail))



print('Similar distributions but one is a bit skewed towards the right tailed')
freq_d2_tail = freq_d1.copy()
print(wasserstein_distance(bins_d1, bins_d1, freq_d1, freq_d2_tail))
for _ in range(10):
	print('Increasing the fist position of the histogram by two units')
	freq_d2_tail[0] += 2
	print(wasserstein_distance(bins_d1, bins_d1, freq_d1, freq_d2_tail))



'''

Some people use a normalised Wasserstein distance
https://www.groundai.com/project/normalized-wasserstein-distance-for-mixture-distributions-with-applications-in-adversarial-learning-and-domain-adaptation/1

It king of makes sense to me when comparing objects that are essentially the same but with different offsets.
Shall I just normalise the samples??


From here https://arxiv.org/abs/1902.00415
"...even if two mixture distributions have identical mixture components but different mixture proportions,
the Wasserstein distance between them will be large. This often leads to undesired results 
in distance-based learning methods for mixture distributions."

These guys provide with the code in here:
https://github.com/yogeshbalaji/Normalized-Wasserstein

'''