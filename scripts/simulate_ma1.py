import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from plot.ts import tsplot

# Simulate an MA(1) process
n = int(1000)

# set the AR(p) alphas equal to 0
alphas = np.array([0.])
betas = np.array([0.6])

# add zero-lag and negate alphas
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma1 = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n)
_ = tsplot(ma1, lags=30, title='MA1 Analysis Plots')
plt.show()



# Fit the MA(1) model to our simulated time series
# Specify ARMA model with order (p, q)
# max_lag = 30
# mdl = smt.ARMA(ma1, order=(0, 1)).fit(
#     maxlag=max_lag, method='mle', trend='nc')
# print(mdl.summary())
# from statsmodels.stats.stattools import jarque_bera

# score, pvalue, _, _ = jarque_bera(mdl.resid)

# if pvalue < 0.10:
#     print 'We have reason to suspect the residuals are not normally distributed.'
# else:
#     print 'The residuals seem normally distributed.'