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


