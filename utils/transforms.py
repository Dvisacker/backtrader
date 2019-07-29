import numpy as np
from scipy.stats import boxcox


def differentiate(X):
  return X.diff()

def detrend(X, window=12):
  return (X - X.close.rolling(window=window).mean()) / X.close.rolling(window=window).std()

def boxcox(X):
  transformed, lam = boxcox(X)
  return transformed, lam

def log(X):
  return np.log(X)

def fit_to_regression(X):
  x = [i for i in range(0, len(X))]
  y = X.values
  model = LinearRegression()
  model.fit(x,y)

  trend = model.predict(x)
  detrended = [y[i] - trend[i] for i in range(0, len(X))]
  return detrended

