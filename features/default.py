import pandas as pd
import numpy as np

from .frac import compute_fractional_differentiation


def add_default_lags(X, y, features, options={}):
  lags = options.get("lags", 4)

  for pair, bars in features.items():
    for i in range(1, lags+1):
      X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)
      X['{}_volume_lag_{}'.format(pair, i)] = bars.volume.shift(i)
      X['{}_volume_weighted_return_{}'.format(pair, i)] = bars.returns.shift(i) * bars.volume.shift(i)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y

def add_default_and_diff_lags(X, y, features, options={}):
  lags = options.get("lags", 4)
  d = options.get("d", 0.40)

  for pair, bars in features.items():
    diff_return_bars = compute_fractional_differentiation(bars.close, d)
    diff_volume_bars = compute_fractional_differentiation(bars.volume, d)

    for i in range(1, lags+1):
      X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)
      X['{}_volume_lag_{}'.format(pair, i)] = bars.volume.shift(i)
      X['{}_volume_weighted_return_{}'.format(pair, i)] = bars.returns.shift(i) * bars.volume.shift(i)
      X['{}_diff_returns_lag_{}'.format(pair, i)] = diff_return_bars.shift(i)
      X['{}_diff_volume_lag_{}'.format(pair, i)] = diff_volume_bars.shift(i)
      X['{}_diff_volume_weighted_return_{}'.format(pair, i)] = diff_return_bars.shift(i) * diff_volume_bars.shift(i)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y


def add_previous_returns(X, y, features, options={}):
  period = options.get("period", 50)

  for pair, bars in features.items():
    X['{}_previous_returns_{}'.format(pair, period)] = bars.close.diff(period)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y


def add_returns(X, y, features, options={}):
  for pair, bars in features.items():
    X['{}_returns'.format(pair)] = bars.returns

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y

def add_lagged_returns(X, y, features, options={}):
  lags = options.get("lags", 4)

  for pair, bars in features.items():
    for i in range(1, lags+1):
      X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y

def add_returns_ma(X, y, features, options={}):
  period = options.get("period", 50)

  for pair, bars in features.items():
    X['{}_returns_ma_{}'.format(pair, period)] = bars.close.rolling(period).mean()

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y

def add_returns_ema(X, y, features, options={}):
  period = options.get("period", 50)

  for pair, bars in features.items():
    X['{}_returns_ema_{}'.format(pair, period)] = bars.close.rolling(period).mean()

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y

def add_returns_zscore(X, y, features, options={}):
  period = options.get("period", 50)
  min_periods = options.get("min_periods", 5)

  zscore = lambda x: (x - x.rolling(window=period, min_periods=min_periods).mean()) \
      / x.rolling(window=period, min_periods=min_periods).std()

  for pair, bars in features.items():
    X['{}_returns_zscore_{}'.format(pair, period)] = bars.close.apply(zscore)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y

def add_returns_rank(X, y, features, options={}):
  period = options.get("period", 50)
  min_periods = options.get("min_periods", 5)

  rank = lambda x: x.rolling(window=period, min_periods=min_periods) \
          .apply(lambda x: pd.Series(x).rank(pct=True)[0])

  for pair, bars in features.items():
    X['{}_returns_rank_{}'.format(pair, period)] = bars.close.apply(rank)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y


def add_signed_returns(X, y, features, options={}):

  for pair, bars in features.items():
    X['{}_signed_returns'.format(pair)] = bars.close.apply(np.sign)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y


def add_lagged_bitcoin_difference(X, y, features, options={}):
  lags = options.get("lags", 4)
  btc_bars = features['BTC/USD']

  for pair, bars in features.items():
    if (pair == 'BTC/USD'):
      continue

    for i in range(1, lags+1):
      X['[{}-BTC/USD]_returns_lag_{}'.format(pair, i)] = btc_bars.close.shift(i) - bars.close.shift(i)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y

def add_lagged_diff_returns(X, y, features, options={}):
  lags = options.get("lags", 4)
  d = options.get("d", 0.40)

  for pair, bars in features.items():
    diff_bars = compute_fractional_differentiation(bars.close, d)

    for i in range(1, lags + 1):
      X['{}_diff_returns_lag_{}'.format(pair, i)] = diff_bars.shift(i)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y


def add_volume_lags(X, y, features, options={}):
  lags = options.get("lags", 4)

  for pair, bars in features.items():
    for i in range(1, lags+1):
      X['{}_volume_lag_{}'.format(pair, i)] = bars.volume.shift(i)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y


def add_volume_mas(X, y, features, options={}):
  period = options.get("period", 50)

  for pair, bars in features.items():
    X['{}_volume_ma_{}'] = bars.close.rolling(period).mean()

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y

def add_lagged_diff_volumes(X, y, features, options={}):
  lags = options.get("lags", 4)
  d = options.get("d", 0.40)

  for pair, bars in features.items():
    diff_bars = compute_fractional_differentiation(bars.volume, d)

    for i in range(1, lags + 1):
      X['{}_diff_volumes_lag_{}'.format(pair, i)] = diff_bars.shift(i)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y

def add_lagged_volume_weighted_returns(X, y, features, options={}):
  lags = options.get("lags", 4)

  for pair, bars in features.items():
    for i in range(1, lags+1):
      X['{}_volume_weighted_return_lag_{}'.format(pair, i)] = bars.returns.shift(i) * bars.volume.shift(i)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y

def add_lagged_diff_weighted_volumes(X, y, features, options={}):
  lags = options.get("lags", 4)
  d = options.get("d", 0.40)

  for pair, bars in features.items():
    diff_weighted_volume_bars = compute_fractional_differentiation(bars.volume, d)
    diff_returns_bars = compute_fractional_differentiation(bars.close, d)

    for i in range(1, lags + 1):
      X['{}_diff_weighted_volumes_lag_{}'.format(pair, i)] = diff_weighted_volume_bars.shift(i) * diff_returns_bars.shift(i)

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)
  return X, y


def compute_default_features(main_pair, raw_features, options={}):
  lags = options.get("lags", 4)
  X = pd.DataFrame()
  for i in range(1, lags + 1):
    X['returns_lag_{}'.format(i)] = main_pair.returns.shift(i)

  if raw_features:
    for pair, bars in raw_features.items():
      for i in range(1, lags + 1):
        X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)

  X.dropna(inplace=True)
  y = main_pair['returns']
  X, y = X.align(y, join='inner', axis=0)

  return X, y



def compute_differentiated_features(main_pair, raw_features, options={}):
  lags = options.get("lags", 4)
  d = options.get("d", 0.40)
  prices = main_pair.close

  diff = compute_fractional_differentiation(prices, 1)

  diff_pairs = {}
  for pair, bars in raw_features.items():
    diff_pairs[pair] = compute_fractional_differentiation(bars.close, d)

  diff_returns = compute_fractional_differentiation(main_pair.close, d)

  X = pd.DataFrame()
  for i in range(1, lags + 1):
    X['diff_lag_{}'.format(i)] = diff.shift(i)

  for pair, bars in diff_pairs.items():
    for i in range(1, lags + 1):
      X['{}_diff_lag_{}'.format(pair, i)] = bars.shift(i)

  X.dropna(inplace=True)
  y = main_pair['returns']
  X, y = X.align(y, join='inner', axis=0)

  return X, y


