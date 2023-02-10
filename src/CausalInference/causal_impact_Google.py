'''
CAUSAL IMPACT

In time-series, Google developed an also called Causal Impact (in R). 

The Python versions can be found here:
https://github.com/dafiti/causalimpact
Pip3 install pycausalimpact

It implements this one done in R by Google:  https://google.github.io/CausalImpact/CausalImpact.html

'''


## two ways of running CI
idx_regular_days = np.array([start_regular, end_regular]).tolist()
idx_promo_days   = np.array([start_promo, end_promo]).tolist()

ci_1 = CausalImpact(sales_sku_B, idx_regular_days, idx_promo_days)

data = pd.DataFrame({'y': sales_sku_B, 'X': sales_sku_A})
ci_2 = CausalImpact(sales_sku_B, idx_regular_days, idx_promo_days)

# It does not make a difference to include the 'cause' (X)



##########

'''

10.02.2022 -  Updates: the repo the Causal Impact implementation that I used in the paper (Dafiti) is now gone. There are several alternative/newer 
implementations:

https://github.com/jamalsenouci/causalimpact
pip3 install git+ssh://github.com/jamalsenouci/causalimpact.git


The wheel from Dafiti?
It seems that the creator (WillFuks) has moved pycausalimpact to TF
https://pypi.org/user/WillFuks/

pip3 install pycausalimpact
(this still works)

pip3 download pycausalimpact

Some tutos
https://github.com/cmp1/bsts-causal_impact/blob/main/Spot_IO_Causal_Impact_Vale_Dam_Accident_20210104.ipynb





'''


import pandas as pd
from causalimpact import CausalImpact


data = pd.read_csv('https://raw.githubusercontent.com/WillianFuks/tfcausalimpact/master/tests/fixtures/arma_data.csv')[['y', 'X']]
data.iloc[70:, 0] += 5

pre_period = [0, 69]
post_period = [70, 99]

ci = CausalImpact(data, pre_period, post_period)
print(ci.summary())
print(ci.summary(output='report'))
ci.plot()

# This will make usage of the algorithm Hamiltonian Monte Carlo which is State-of-the-Art for finding the Bayesian posterior of distributions. 
ci = CausalImpact(data, pre_period, post_period, model_args={'fit_method': 'hmc'})
print(ci.summary())


'''
Tutorial in PyData Sao Paulo by the author
https://www.youtube.com/watch?v=yN5jcKnKVrs
https://github.com/WillianFuks/pyDataSP-tfcausalimpact/blob/master/pyDataSP%20-%20tfcausalimpact.ipynb