import pandas as pd
import pdb


def compute_signals(y, y_pred, classified=False):
  if classified == True:
    signals = compute_classified_predictions_to_signals(y, y_pred)
  else:
    signals = compute_regressed_predictions_to_signals(y, y_pred)

  return signals

def compute_regressed_predictions_to_signals(y, y_pred):
  up1 = y[y_pred > y_pred.mean() + 1 * y_pred.std()]
  down1 = y[y_pred < y_pred.mean() - 1 * y_pred.std()]
  signals_up = pd.Series(1, index=up1.index)
  signals_down = pd.Series(-1, index=down1.index)
  signals = pd.concat([signals_up, signals_down]).sort_index()

  return signals


def compute_classified_predictions_to_signals(y, y_pred):
  y_pred = pd.Series(y_pred, index=y.index)

  signals_up = y_pred[y_pred == 1]
  signals_down = y_pred[y_pred == -1]
  signals = pd.concat([signals_up, signals_down]).sort_index()

  return signals

