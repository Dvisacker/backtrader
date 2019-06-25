import numpy as np
import talib

def crossover_value(x,y):
  return(x[-1] > y and x[-2] < y)

def crossunder_value(x,y):
  return (x[-1] < y and x[-2] > y)

def crossover(x,y):
  return(x[-1] > y[-1] and x[-2] < y[-2])

def crossunder(x,y):
  return (x[-1] < y[-1] and x[-2] > y[-2])

def macd_crossover(x, short_window, long_window, trigger_window):
  macd_line, signal_line, _ = talib.MACD(x, fastperiod=short_window, slowperiod=long_window, signalperiod=trigger_window)
  return crossover(macd_line, signal_line)

def macd_crossunder(x, short_window, long_window, trigger_window):
  macd_line, signal_line, _ = talib.MACD(x, fastperiod=short_window, slowperiod=long_window, signalperiod=trigger_window)
  return crossunder(macd_line, signal_line)

def rsi_crossover(x, timeperiod, upper_threshold):
  rsi = talib.RSI(x, timeperiod=timeperiod)
  return crossover_value(rsi, upper_threshold)

def rsi_crossunder(x, timeperiod, lower_threshold):
  rsi = talib.RSI(x, timeperiod=timeperiod)
  return crossunder_value(rsi, lower_threshold)

def mavg_crossover(x, short_window, long_window):
  short_sma = talib.SMA(x, timeperiod=short_window)
  long_sma = talib.SMA(x, timeperiod=long_window)
  return crossover(short_sma, long_sma)

def mavg_crossunder(x, short_window, long_window):
  short_sma = talib.SMA(x, timeperiod=short_window)
  long_sma = talib.SMA(x, timeperiod=long_window)
  return crossunder(short_sma, long_sma)

def atr_crossover(high, low, close, timeperiod, threshold):
  atr = talib.ATR(high, low, close, timeperiod=timeperiod)
  return crossover(atr, threshold)

def atr_crossunder(high, low, close, timeperiod, threshold):
  atr = talib.ATR(high, low, close, timeperiod=timeperiod)
  return crossunder(atr, threshold)


def consecutive_ups(x):
  i = 0
  while True:
    a = x[-1 + i]
    b = x[-2 + i]

    if b > a:
      break

    i += 1

def consecutive_downs(x):
  i = 0
  while True:
    a = x[-1 + i]
    b = x[-2 + i]

    if a < b:
      break

    i += 1





