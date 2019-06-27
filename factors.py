import pandas as pd
import numpy as np
import numexpr as ne

from numpy import broadcast_arrays

from scipy.stats import (
  linregress,
  pearsonr,
  spearmanr
)


def momentum(timeperiod):
  return lambda prices: (prices[-1] / prices[-timeperiod])

def average_volume_traded(volumes):
  return lambda prices, volumes: np.mean(prices * volumes, axis = 0)

def simple_moving_average(timeperiod):
  return lambda prices: np.mean(prices[-timeperiod:])

def vwap(timeperiod):
  return lambda prices, volumes: np.sum(prices[-timeperiod:] * volumes[-timeperiod:], axis=0) / np.sum(volumes[-timeperiod:], axis=0)

def max_drawdown(timeperiod):
  def max_drawdown_factor(prices):
    prices = prices[-timeperiod:]
    drawdowns = np.fmax.accumulate(prices, axis=0) - prices
    drawdowns[np.isnan(drawdowns)] = np.NINF
    drawdown_end = np.nanargmax(drawdowns, axis=0)
    peak = np.max(prices[:drawdown_end + 1])

    return (peak - prices[drawdown_end]) / prices[drawdown_end]

  return max_drawdown_factor

def average_dollar_volume(timeperiod):
  def average_dollar_volume_factor(closes, volumes):
    closes = closes[-timeperiod:]
    volumes = volumes[-timeperiod:]
    return np.sum(closes * volumes) / len(closes)

  return average_dollar_volume_factor

def returns(timeperiod):
  def returns_factor(prices):
    if timeperiod < 2:
      raise ValueError("'returns' expects a window length of at least 2")

    return (prices[-1] - prices[-timeperiod])

  return returns_factor

def rolling_pearson(timeperiod):
  def rolling_pearson_factor(data, target_data):
    return pearsonr(data[-timeperiod:], target_data[-timeperiod:])

  return rolling_pearson_factor

def rolling_spearman(timeperiod):
  def rolling_spearman_factor(data, target_data):
    return spearmanr(data[-timeperiod:], target_data[-timeperiod:])

  return rolling_spearman_factor

def bollinger_bands(timeperiod, k):
  def bollinger_bands_factor(data):
    difference = k * np.std(data, axis=0)
    middle = np.mean(data, axis=0)
    upper = middle + difference
    lower = middle - difference

    return lower, middle, upper

  return bollinger_bands_factor


def aaron(timeperiod):
  def aaron_factor(lows, highs):
    highs = highs[-timeperiod:]
    lows = lows[-timeperiod:]
    high_date_index = np.argmax(highs, axis=0)
    low_date_index = np.argmin(lows, axis=0)
    up = ''
    down = ''

    ne.evaluate(
        '(100 * high_date_index) / (timeperiod - 1)',
        local_dict={
            'high_date_index': high_date_index,
            'timeperiod': timeperiod,
        },
        out=up,
    )

    ne.evaluate(
        '(100 * low_date_index) / (timeperiod - 1)',
        local_dict={
            'low_date_index': low_date_index,
            'timeperiod': timeperiod,
        },
        out=down,
    )

    return down, up

  return aaron_factor


def rsi(timeperiod):
  def rsi_factor(data):
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

  return rsi_factor


def rate_of_change_pct(timeperiod):
  def rate_of_change_pct_factor(data):
    last_close = data[-1]
    previous_close = data[-timeperiod]
    return ((last_close - previous_close) / previous_close) * 100

  return rate_of_change_pct_factor
