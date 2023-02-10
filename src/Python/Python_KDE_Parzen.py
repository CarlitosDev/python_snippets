'''
Density estimation via the Parzen-window technique with a Gaussian kernel

From Raschka: https://sebastianraschka.com/Articles/2014_kernel_density_est.html#73-density-estimation-via-the-parzen-window-technique-with-a-gaussian-kernel

'''
from scipy.stats import kde
import numpy as np
import matplotlib.pyplot as plt

n = 100
m2 = np.random.normal(scale=0.5, size=n)

pdf_kde = kde.gaussian_kde(m2, bw_method='scott')

# Get the prob of a point being 0.8
print(pdf_kde.evaluate(0.8)[0])

# Get the probability betweem 2 points
low = 0.1
high = 0.4
pdf_kde.integrate_box_1d(low, high)

pdf_kde.integrate_box(low, high)


x = np.linspace(0,1,1000)
pdf_approx = pdf_kde.evaluate(x)

plt.plot(x, pdf_approx)
plt.show()



###
import seaborn as sns
sns.distplot(pdf_approx)
plt.show()




pdf_kde.pdf(0.8)



'''
from sklearn.neighbors import KernelDensity
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(m2)
log_dens = kde.score_samples(X_plot)
ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
        label="kernel = '{0}'".format(kernel))
''' 