import matplotlib.pyplot as plt
import numpy as np
from plot.ts import tsplot

# Simulating a GARCH(1, 1) process
np.random.seed(2)

a0 = 0.2
a1 = 0.5
b1 = 0.3

n = 10000
w = np.random.normal(size=n)
eps = np.zeros_like(w)
sigsq = np.zeros_like(w)

for i in range(1, n):
    sigsq[i] = a0 + a1*(eps[i-1]**2) + b1*sigsq[i-1]
    eps[i] = w[i] * np.sqrt(sigsq[i])

tsplot(eps, lags=30, title='GARCH Analysis Plots')
tsplot(eps**2, lags=30, title='GARCH Analysis Plots (squared)')
plt.show()