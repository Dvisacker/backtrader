import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from plot.ts import tsplot


# Simulate an ARMA(2, 2) model with alphas=[0.5,-0.25] and betas=[0.5,-0.3]
max_lag = 30
n = int(5000) # lots of samples to help estimates
burn = int(n/10) # number of samples to discard before fit
alphas = np.array([0.5, -0.25])
betas = np.array([0.5, -0.3])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

arma22 = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn)
tsplot(arma22, lags=max_lag, title='ARMA22 analysis plots')
plt.show()