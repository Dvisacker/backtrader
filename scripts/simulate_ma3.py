import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from plot.ts import tsplot

n = int(500)
alphas = np.array([0.])
betas = np.array([0.3, 0.2, 0.1])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma3 = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n)
_ = tsplot(ma3, lags=30, title='MA3 Analysis Plots')

plt.show()