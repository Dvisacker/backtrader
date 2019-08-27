import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm, tqdm_notebook


def get_weights(d, thres, max_size=10000):
  """
  Get coefficients for computing fractional derivatives
  """

  w = [1.]
  for k in range(1, max_size):
    w_ = -w[-1] / k * (d - k + 1)
    if abs(w_) <= thres:
      break
    w.append(w_)

  w = np.array(w)
  return w

def compute_fractional_differentiation(series, d, lag=1, thres=1e-3, max_size=10000):
  """
  Compute fractional differentiation
  """

  max_size = int(max_size / lag)
  w = get_weights(d, thres, max_size)

  width = len(w)
  series_ = series.fillna(method='ffill').dropna()
  rolling_array = []

  for i in range(width):
    rolling_array.append(series_.shift(i * lag).values)

  rolling_array = np.array(rolling_array)
  series_val = np.dot(rolling_array.T, w)
  series = pd.Series(index=series.index)
  timestamps = series.index[-len(series_val):]
  series.loc[timestamps] = series_val
  return series


def get_optimal_differentiation(series, lag=1, thres=1e-5, max_size=10000,
                                p_thres=1e-4, autolag=None, verbose=1, **kwargs):
  """
  Find minimum value of degree of stationarity differential
  """
  ds = np.array(np.linspace(0, 1, 100))

  opt_d = ds[-1]

  for d in tqdm(ds):
    diff = compute_fractional_differentiation(series, d=d, thres=thres, max_size=max_size)
    pval = adfuller(diff.dropna().values, autolag=autolag, **kwargs)[1]
    if pval < p_thres:
      opt_d = d
      break

  return opt_d