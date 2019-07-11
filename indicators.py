import pandas as pd
import numpy as np
import numexpr as ne

from numpy import broadcast_arrays
from scipy.stats import (
  linregress,
  pearsonr,
  spearmanr
)

def zscore(series):
    return (series - series.mean()) / np.std(series)

def moving_average(series, window):
    return pd.rolling_mean(series, window)

def moving_std(series, window):
    return pd.rolling_std(series, window)

def classic_momentum(series, window):
  return (series[-1] - series[-window]) / series[-window]

def quotient_momentum(series, window):
  return series[-1] / series[-window]

def mean_momentum(series, short_window, long_window):
  old_mean = np.mean(series[-long_window:-short_window])
  new_mean = np.mean(series[-short_window:])
  std = np.std(series[-long_window:])

  return (new_mean - old_mean) / std

def mean_volume_deviation(volumes, window):
  volumes_series = pd.Series(volumes)
  mean = volumes_series.rolling(window).mean()
  std = volumes_series.rolling(window).std()

  result = (volumes_series - mean) / std
  return result.iloc[-1]

def modified_momentum(high, low, close, momentum_window, atr_window, data_window):
  old_mean = np.mean(close[-data_window:-momentum_window])
  new_mean = np.mean(close[-momentum_window:-1])
  diff = new_mean - old_mean

  high = high[-data_window:-1]
  low = low[-data_window:-1]
  hmpc = np.abs()
  hml = high - low
  hmpc = np.abs(high - np.roll(close, 1, axis=0))
  lmpc = np.abs(low - np.roll(close, 1, axis=0))
  tr = np.maximum(hml, np.maximum(hmpc, lmpc))
  atr = np.mean(tr[-atr_window:], axis=0)

  return diff/atr


def average_volume_traded(prices, volumes):
  return np.mean(prices * volumes, axis = 0)

def vwap(prices, volumes, window):
  return np.sum(prices[-window:] * volumes[-window:], axis=0) / np.sum(volumes[-window:], axis=0)

def max_drawdown(data, window):
  prices = data[-window:]
  drawdowns = np.fmax.accumulate(prices, axis=0) - data
  drawdowns[np.isnan(drawdowns)] = np.NINF
  drawdown_end = np.nanargmax(drawdowns, axis=0)
  peak = np.max(data[:drawdown_end + 1])

  return (peak - prices[drawdown_end]) / prices[drawdown_end]

def average_dollar_volume(closes, volumes, window):
  closes = closes[-window:]
  volumes = volumes[-window:]
  return np.sum(closes * volumes) / len(closes)

def returns(prices, window):
  if window < 2:
    raise ValueError("'returns' expects a window length of at least 2")

  return (prices[-1] - prices[-window])

def rolling_pearson(data, target_data, window):
  return pearsonr(data[-window:], target_data[-window:])

def rolling_spearman(data, target_data, window):
  return spearmanr(data[-window:], target_data[-window:])

def bollinger_bands(data, window, k):
  difference = k * np.std(data, axis=0)
  middle = np.mean(data, axis=0)
  upper = middle + difference
  lower = middle - difference

  return lower, middle, upper


def aaron(lows, highs, window):
  high_date_index = np.argmax(highs, axis=0)
  low_date_index = np.argmin(lows, axis=0)
  up = ''
  down = ''

  ne.evaluate(
      '(100 * high_date_index) / (window - 1)',
      local_dict={
          'high_date_index': high_date_index,
          'window': window,
      },
      out=up,
  )

  ne.evaluate(
      '(100 * low_date_index) / (window - 1)',
      local_dict={
          'low_date_index': low_date_index,
          'window': window,
      },
      out=down,
  )

  return down, up


def rsi(data, window):
  diffs = np.diff(data, axis=0)
  ups = np.mean(np.clip(diffs, 0, np.inf), axis=0)
  downs = abs(np.mean(np.clip(diffs, -np.inf, 0), axis=0))

  results = ''

  ne.evaluate(
    "100 - (100 / (1 + (ups / downs)))",
    local_dict={'ups': ups, 'downs': downs},
    global_dict={},
    out=results,
  )

  return results


def rate_of_change_pct(data, window):
  last_close = data[-1]
  previous_close = data[-window]
  return ((last_close - previous_close) / previous_close) * 100

def atr(data, high, low, close, window):
  high = high[-window:-1]
  low = low[-window:-1]
  hml = high - low
  hmpc = np.abs(high - np.roll(close, 1, axis=0))
  lmpc = np.abs(low - np.roll(close, 1, axis=0))
  tr = np.maximum(hml, np.maximum(hmpc, lmpc))
  atr = np.mean(tr[1:], axis=0) #skip the first one as it will be NaN
  return atr
