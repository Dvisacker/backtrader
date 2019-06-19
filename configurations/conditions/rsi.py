from condition_functions import *

conditions = []

for tp in SHORT_TIMEPERIODS:
  for (low_thres, high_thres) in zip(RSI_LOW_THRESHOLDS, RSI_HIGH_THRESHOLDS):
    conditions.append({
      "long": rsi_crossunder_function(tp, low_thres),
      "short": rsi_crossover_function(tp, high_thres),
      "name": "rsi-period={},thresholds=({},{})".format(tp, low_thres, high_thres)
    })