Survival analysis is generally defined as a set of methods for analyzing data where the outcome variable is
the time until the occurrence of an event of interest. The event can be death, occurrence of 
a disease, marriage, divorce, etc. The time to event or survival time can be measured in days, weeks, years, etc.

Used to overcome situations such as how to average profiles for events that have not happened yet, such as death in a life insurer.
Survival and Hazard function
Packages in Python: lifelines and scikit-survival.

lifelines only linear interactions.

https://github.com/sebp/scikit-survival


Kaplan-Meier estimate of the survival function

Can we use survival analysis to model the expected time a person will work for Beamly?


Survival regression using Cox Proportional Hazards model
(min 16:08 https://www.youtube.com/watch?v=aKZQUaNHYb0&list=PLGVZCDnMOq0ovNxfxOqYcBcQOIny9Zvb-&index=47)


cph = CoxPHFitter()
cph.fit(data, 'duration_alive', event_col='observed_end')

CoxPHFitter() method


###

from lifelines.datasets import load_waltons
import matplotlib.pyplot as plt
df = load_waltons() 
df.head()

T = df['T']
E = df['E']

import lifelines as lfl
import numpy as np

from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)

kmf.survival_function_
kmf.cumulative_density_
kmf.median_
kmf.plot_survival_function()
plt.show()

kmf.plot_cumulative_density()
plt.show()


## All the models
fig, axes = plt.subplots(2, 3, figsize=(9, 5))

kmf = lfl.KaplanMeierFitter().fit(T, E, label='KaplanMeierFitter')
wbf = lfl.WeibullFitter().fit(T, E, label='WeibullFitter')
exf = lfl.ExponentialFitter().fit(T, E, label='ExponentalFitter')
lnf = lfl.LogNormalFitter().fit(T, E, label='LogNormalFitter')
llf = lfl.LogLogisticFitter().fit(T, E, label='LogLogisticFitter')
pwf = lfl.PiecewiseExponentialFitter([40, 60]).fit(T, E, label='PiecewiseExponentialFitter')

wbf.plot_survival_function(ax=axes[0][0])
exf.plot_survival_function(ax=axes[0][1])
lnf.plot_survival_function(ax=axes[0][2])
kmf.plot_survival_function(ax=axes[1][0])
llf.plot_survival_function(ax=axes[1][1])
pwf.plot_survival_function(ax=axes[1][2])

plt.show()




## Regression models

rossi = lfl.datasets.load_rossi()
rossi.head()

# Using Cox Proportional Hazards model
cph = lfl.CoxPHFitter()
cph.fit(rossi, 'week', event_col='arrest')
cph.print_summary()
cph.predict_median(rossi)
cph.predict_expectation(rossi)

cph.plot_covariate_groups('prio', values=np.arange(0, 15), cmap='coolwarm')
plt.show()





wft = lfl.WeibullAFTFitter()
wft.fit(rossi, 'week', event_col='arrest', ancillary_df=rossi)
wft.print_summary()
wft.predict_median(rossi)