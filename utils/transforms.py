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



def resample_null_bars(bars):
  def custom_fill(row_name):
    return lambda row: row['close'] if np.isnan(row[row_name]) else row[row_name]

  def custom_null_fill(row_name):
      return lambda row: 0 if np.isnan(row[row_name]) else row[row_name]

  bars.resample('1min').last()
  bars.loc[:, 'close'] = bars.loc[:,'close'].ffill()
  bars['open'] = bars.apply(custom_fill('open'), axis=1)
  bars['high'] = bars.apply(custom_fill('high'), axis=1)
  bars['low'] = bars.apply(custom_fill('low'), axis=1)

  if 'ofi' in bars:
    bars['ofi'] = bars.apply(custom_null_fill('ofi'), axis=1)

  if 'tfi' in bars:
    bars['tfi'] = bars.apply(custom_null_fill('tfi'), axis=1)

  if 'vfi' in bars:
    bars['vfi'] = bars.apply(custom_null_fill('vfi'), axis=1)

  if 'volume' in bars:
    bars['volume'] = bars.apply(custom_null_fill('volume'), axis=1)

  if 'returns' in bars:
    bars['returns'] = bars.apply(custom_null_fill('returns'), axis=1)

  if 'midprice_returns' in bars:
    bars['midprice_returns'] = bars.apply(custom_null_fill('midprice_returns'), axis=1)

  return bars
