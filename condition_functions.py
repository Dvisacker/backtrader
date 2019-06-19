import numpy as np
import talib

SHORT_TIMEPERIODS = [
  10,
  20,
  30,
  40,
  50,
  60,
  70,
  80,
  90,
  100
]

LONG_TIMEPERIODS = [
  100,
  200,
  300,
  400,
  500,
  600,
  700,
  800,
  900,
  1000
]

RSI_LOW_THRESHOLDS = [
  10,
  15,
  20,
  25,
  30,
  35,
  40,
  45
]

RSI_HIGH_THRESHOLDS = [
  55,
  60,
  65,
  70,
  75,
  80,
  85,
  90
]



def crossover_value(x,y):
  return(x[-1] > y and x[-2] < y)

def crossunder_value(x,y):
  return (x[-1] < y and x[-2] > y)

def crossover(x,y):
  return(x[-1] > y[-1] and x[-2] < y[-2])

def crossunder(x,y):
  return (x[-1] < y[-1] and x[-2] > y[-2])

def and_function(condition_1, condition_2):
  return lambda x: condition_1(x) and condition_2(x)

def or_function(condition_1, condition_2):
  return lambda x: condition_1(x) or condition_2(x)

def macd_crossover_function(short_window, long_window, trigger_window):
  def macd_crossover(x):
    macd_line, signal_line, _ = talib.MACD(x, fastperiod=short_window, slowperiod=long_window, signalperiod=trigger_window)
    return crossover(macd_line, signal_line)

  return macd_crossover

def macd_crossunder_function(short_window, long_window, trigger_window):
  def macd_crossunder(x):
    macd_line, signal_line, _ = talib.MACD(x, fastperiod=short_window, slowperiod=long_window, signalperiod=trigger_window)
    return crossunder(macd_line, signal_line)

  return macd_crossunder

def rsi_crossover_function(timeperiod, upper_threshold):
    def rsi_crossover(x):
      rsi = talib.RSI(x, timeperiod=timeperiod)
      return crossover_value(rsi, upper_threshold)

    return rsi_crossover

def rsi_crossunder_function(timeperiod, lower_threshold):
    def rsi_crossunder(x):
      rsi = talib.RSI(x, timeperiod=timeperiod)
      return crossunder_value(rsi, lower_threshold)

    return rsi_crossunder

def mavg_crossover_function(short_window, long_window):
    def mavg_crossover(x):
        short_sma = talib.SMA(x, timeperiod=short_window)
        long_sma = talib.SMA(x, timeperiod=long_window)
        return crossover(short_sma, long_sma)

    return mavg_crossover

def mavg_crossunder_function(short_window, long_window):
    def mavg_crossunder(x):
        short_sma = talib.SMA(x, timeperiod=short_window)
        long_sma = talib.SMA(x, timeperiod=long_window)
        return crossunder(short_sma, long_sma)

    return mavg_crossunder

def atr_crossover_function(timeperiod, threshold):
    def atr_crossover(high, low, close):
        atr = talib.ATR(high, low, close, timeperiod=timeperiod)
        return crossover(atr, threshold)

    return atr_crossover

def atr_crossunder_function(timeperiod, threshold):
    def atr_crossunder(high, low, close):
        atr = talib.ATR(high, low, close, timeperiod=timeperiod)
        return crossunder(atr, threshold)

    return atr_crossunder

def consecutive_ups_function():
    def consecutive_ups(x):
      i = 0
      while True:
        a = x[-1 + i]
        b = x[-2 + i]

        if b > a:
          break

        i += 1

    return consecutive_ups

def consecutive_downs_function():
    def consecutive_downs(x):
      i = 0
      while True:
        a = x[-1 + i]
        b = x[-2 + i]

        if a < b:
          break

        i += 1

    return consecutive_downs



