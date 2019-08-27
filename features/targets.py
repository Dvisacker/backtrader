import pandas as pd

def one_step_forward_returns(target_data):
  y = target_data.close.pct_change(-1)
  y.dropna(inplace=True)
  return y

def five_step_forward_returns(target_data):
  y=  target_data.close.pct_change(-5)
  y.dropna(inplace=True)
  return y

def ten_step_forward_returns(target_data):
  y = target_data.close.pct_change(-10)
  y.dropna(inplace=True)
  return y

def classify_target(X, y, options={}):
  nb_std = options.get("nb_std", 1)

  std = y.std()
  avg = y.mean()
  y = pd.Series(0, y.index)
  y[y > avg + nb_std * std] = 1
  y[y < avg - nb_std * std] = -1
  return X, y

