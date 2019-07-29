import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from plot.ts import tsplot

np.random.seed(1)
n_samples = 1000

x = w = np.random.normal(size=n_samples)
for t in range(n_samples):
    x[t] = x[t-1] + w[t]

tsplot(x, lags=30, title='Random Walk Analysis Plots')
tsplot(np.diff(x), lags=30, title='Random Walk .diff Analysis Plots')
plt.show()