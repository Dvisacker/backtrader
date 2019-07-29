import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from plot.ts import tsplot

np.random.seed(1)
randser = np.random.normal(size=1000)
tsplot(randser, lags=30, title='White noise')
plt.show()