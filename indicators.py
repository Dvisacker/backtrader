import numpy as np
import talib

def beta(high, low, timeperiod=5):
  return talib.BETA(high, low, timeperiod)

def pearson_correlation(high, low, timeperiod=30):
  return talib.CORREL(high, low, timeperiod)

def linear_reg(close, timeperiod=14):
  return talib.LINEARREG(close, timeperiod)

def std_dev(x, timeperiod=5, nbdev=1):
  return talib.STDDEV(x, timeperiod, nbdev)

def rsi(x, rsi_period):
  return talib.RSI(x, rsi_period)

def mavg(x, short_window, long_window):
  short_sma = talib.SMA(x, timeperiod=short_window)
  long_sma = talib.SMA(x, timeperiod=long_window)

  return short_sma - long_sma

def macd_trigger_line(x, short_window, long_window, trigger_window):
  macd_line, signal_line, _ = talib.MACD(x, fastperiod=short_window, slowperiod=long_window, signalperiod=trigger_window)
  return macd_line - signal_line
