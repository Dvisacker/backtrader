import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels.api as sm
import numpy as np

def tsplot(y, lags=None, title='', figsize=(10, 10), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax, lw=1.)
        ts_ax.set_title(title)
        sm.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        sm.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax, lw=1., markersize=2, alpha=0.5)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        pp_ax.get_lines()[0].set_markersize(2)

        plt.tight_layout()
    return


n = int(1000)
alphas = np.array([.666, -.333])
betas = np.array([0.])

# Python requires us to specify the zero-lag value which is 1
# Also note that the alphas for the AR model must be negated
# We also set the betas for the MA equal to 0 for an AR(p) model
# For more information see the examples at statsmodels.org
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]
x = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n)

tsplot(x, lags=30, title='AR2 Analysis Plots')
plt.show()



# Fit an AR(p) model to simulated AR(2) process
# max_lag = 10
# mdl = smt.AR(ar2).fit(maxlag=max_lag, ic='aic', trend='nc')
# est_order = smt.AR(ar2).select_order( maxlag=max_lag, ic='aic', trend='nc')

# true_order = 2
# print('\ncoef estimate: %3.4f %3.4f | order estimate %s'%(mdl.params[0],mdl.params[1],est_order))

# N = 10
# AIC = np.zeros((N, 1))

# for i in range(N):
#     model = smt.AR(ar2)
#     model = model.fit(maxlag=(i+1))
#     AIC[i] = model.aic

# AIC_min = np.min(AIC)
# model_min = np.argmin(AIC)

# print 'Number of parameters in minimum AIC model %s' % (model_min+1)