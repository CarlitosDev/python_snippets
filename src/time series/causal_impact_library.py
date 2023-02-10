'''
causal_impact_library.py

In time-series, Google developed an also called Causal Impact (in R). 
The Python versions can be found here:
https://github.com/dafiti/causalimpact
Pip3 install pycausalimpact

'''

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from causalimpact import CausalImpact


np.random.seed(12345)
ar = np.r_[1, 0.9]
ma = np.array([1])
arma_process = ArmaProcess(ar, ma)
X = 100 + arma_process.generate_sample(nsample=100)
y = 1.2 * X + np.random.normal(size=100)
y[70:] += 5

data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])
pre_period = [0, 69]
post_period = [70, 99]

ci = CausalImpact(data, pre_period, post_period)
print(ci.summary())
print(ci.summary(output='report'))
ci.plot()



# Using seasonal components:
df = pd.DataFrame(data)
df = df.set_index(pd.date_range(start='20180101', periods=len(data)))
pre_period = ['20180101', '20180311']
post_period = ['20180312', '20180410']
ci = CausalImpact(df, pre_period, post_period, nseasons=[{'period': 7}])




#####

idx_promo_A = np.hstack([np.zeros(6,dtype=bool), np.ones(3,dtype=bool), False])

idx_promos = pd.Series(np.hstack([idx_promo_A, idx_promo_A]))

# Groups/sequences
seqs = (idx_promos.shift(1)!=idx_promos).cumsum()
promo_seqs = seqs[idx_promos]
non_promo_seqs = seqs[~idx_promos]

for value_promo_seqs in promo_seqs.unique():
  prev_seq = value_promo_seqs-1
  idx_pre_intervention = (seqs==prev_seq)
  idx_post_intervention = (seqs==value_promo_seqs)
  print(idx_pre_intervention, idx_post_intervention)



def split_promos_into_sequences(idx_promos: 'pd.Series'):
  # Groups/sequences
  seqs = (idx_promos.shift(1)!=idx_promos).cumsum()
  promo_seqs = seqs[idx_promos]
  # Indices
  idx_pre_intervention = []
  idx_post_intervention = []
  for value_promo_seqs in promo_seqs.unique():
    prev_seq = value_promo_seqs-1
    idx_pre_intervention.append(seqs==prev_seq)
    idx_post_intervention.append(seqs==value_promo_seqs)
  return idx_pre_intervention, idx_post_intervention




