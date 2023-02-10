'''
  Tester for the library STUMPY which does some time series analysis
  Paper: https://joss.theoj.org/papers/eb91faaf9219d46c9acd373cfee8ac29
  GitHub: https://github.com/TDAmeritrade/stumpy

'''

import stumpy
import numpy as np

your_time_series = np.random.rand(10000)
window_size = 50  # Approximately, how many data points might be found in a pattern

'''
output – The first column consists of the matrix profile
        the second column consists of the matrix profile indices
        the third column consists of the left matrix profile indices
        the fourth column consists of the right matrix profile indices.

        * matrix profile represents the distances between
        all subsequences within a time series and their nearest neighbor
'''
matrix_profile = stumpy.stump(your_time_series, m=window_size)